import datetime as dt
import warnings

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from sonora.breakout.evaluation import EvaluationPeriod, evaluate_scores, rank_scores
from sonora.breakout.modeling import (
    MODEL_FEATURES,
    fit_logistic_regression,
    run_logistic_walk_forward,
    score_logistic_regression,
)

KEYS = ["user_id", "artist_id", "scoring_date"]


def training():
    return pl.DataFrame(
        {
            "listening_share_7d": [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8],
            "previous_listening_count": [0, 1, 2, 4, 8, 16, 32, 64],
            "target_60d": [False, False, False, True, False, True, True, True],
            "has_full_horizon": [True] * 8,
            "unrelated_metadata": [1000] * 8,
        }
    )


def scoring():
    return pl.DataFrame(
        {
            "user_id": ["u", "u", "u", "v"],
            "artist_id": ["a", "b", "c", "a"],
            "scoring_date": [dt.date(2022, 1, 1)] * 4,
            "listening_share_7d": [0.1, 0.1, np.nextafter(0.1, 1.0), 0.1],
            "previous_listening_count": [2] * 4,
        }
    )


def baseline():
    return (
        scoring()
        .select(KEYS)
        .with_columns(pl.Series("score", [-2.0, -1.0, -3.0, -1.0]))
    )


class ShareScores:
    def decision_function(self, inputs):
        return inputs[:, 0]


def test_preprocessing_logs_only_previous_count_and_fits_all_training_rows():
    rows = training()
    model = fit_logistic_regression(rows)
    transformed = np.column_stack(
        [
            rows["listening_share_7d"].to_numpy(),
            np.log1p(rows["previous_listening_count"].to_numpy()),
        ]
    )
    assert_allclose(
        model.named_steps["transform"].transform(
            rows.select(MODEL_FEATURES).to_numpy()
        ),
        transformed,
    )
    assert_allclose(model.named_steps["scale"].mean_, transformed.mean(axis=0))
    assert model.named_steps["scale"].n_samples_seen_ == rows.height


@pytest.mark.parametrize("interaction", [False, True])
def test_scoring_keeps_fitted_preprocessing_and_ignores_target_metadata(interaction):
    model = fit_logistic_regression(training(), interaction=interaction)
    means = model.named_steps["scale"].mean_.copy()
    scales = model.named_steps["scale"].scale_.copy()
    rows = scoring().with_columns(pl.lit(1_000_000).alias("previous_listening_count"))
    result = score_logistic_regression(model, rows, baseline())
    altered = rows.with_columns(
        pl.lit(True).alias("target_60d"), pl.lit(-1000).alias("unrelated_metadata")
    )
    other = score_logistic_regression(model, altered, baseline())
    assert_allclose(model.named_steps["scale"].mean_, means)
    assert_allclose(model.named_steps["scale"].scale_, scales)
    assert_allclose(result["model_score"].to_numpy(), other["model_score"].to_numpy())


def test_interaction_uses_share_times_logged_count_before_training_only_scaling():
    rows = training()
    model = fit_logistic_regression(rows, interaction=True)
    share = rows["listening_share_7d"].to_numpy()
    logged = np.log1p(rows["previous_listening_count"].to_numpy())
    terms = np.column_stack([share, logged, share * logged])
    assert_allclose(model[:2].transform(rows.select(MODEL_FEATURES).to_numpy()), terms)
    assert_allclose(model.named_steps["scale"].mean_, terms.mean(axis=0))
    assert model.named_steps["scale"].n_samples_seen_ == rows.height
    additive = fit_logistic_regression(rows)
    assert (
        model.named_steps["classifier"].get_params()
        == additive.named_steps["classifier"].get_params()
    )
    assert model.named_steps["classifier"].n_features_in_ == 3


def test_exact_ties_follow_baseline_but_even_small_model_differences_take_priority():
    rows = scoring()
    result = score_logistic_regression(ShareScores(), rows, baseline().reverse())
    assert result["artist_id"].to_list() == ["c", "b", "a", "a"]
    assert result["score"].to_list() == [-1.0, -2.0, -3.0, -1.0]
    assert result["baseline_score"].to_list() == [-3.0, -1.0, -2.0, -1.0]
    assert rank_scores(result, rows)["rank"].to_list() == [1, 2, 3, 1]
    shuffled = score_logistic_regression(ShareScores(), rows.reverse(), baseline())
    assert result.equals(shuffled)


def test_walk_forward_fits_independent_pipelines_using_only_available_training_rows():
    dates = (
        [dt.date(2021, 9, 1)] * 4
        + [dt.date(2022, 1, 2)] * 2
        + [dt.date(2022, 7, 2)] * 2
    )
    data = pl.DataFrame(
        {
            "user_id": ["u"] * 8,
            "artist_id": [f"a{i}" for i in range(8)],
            "scoring_date": dates,
            "horizon_end_date": [day + dt.timedelta(days=60) for day in dates],
            "target_60d": [False, True] * 4,
            "has_full_horizon": [True] * 8,
            "listening_share_7d": [0.01, 0.2, 0.03, 0.3, 0.02, 0.8, 0.01, 0.7],
            "previous_listening_count": [2, 20, 1, 30, 1, 80, 1, 70],
        }
    )
    expected = data.tail(4)
    baseline_scores = expected.select(KEYS).with_columns(
        pl.Series("score", [-2.0, -1.0, -2.0, -1.0])
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False),
        EvaluationPeriod("2022 H2", dt.date(2022, 7, 1), dt.date(2022, 7, 2), True),
    ]
    first, second = run_logistic_walk_forward(data, periods, baseline_scores)
    assert first.model is not second.model
    assert first.fold.training_rows == 4
    assert second.fold.training_rows == 6
    for result in [first, second]:
        rows = result.fold.training
        assert_allclose(
            result.model.named_steps["scale"].mean_,
            [
                rows["listening_share_7d"].mean(),
                np.log1p(rows["previous_listening_count"].to_numpy()).mean(),
            ],
        )
        assert result.model.named_steps["scale"].n_samples_seen_ == rows.height
    combined = pl.concat([first.scores, second.scores])
    events = pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["a5"],
            "breakout_period_start": [dt.date(2022, 2, 1)],
            "first_important_period_end": [dt.date(2022, 2, 7)],
            "confirmation_date": [dt.date(2022, 3, 2)],
        }
    )
    evaluation = evaluate_scores(
        combined,
        expected,
        events,
        oos_start=periods[0].scoring_start,
        oos_end=periods[-1].scoring_end,
    )
    assert evaluation.rankings.height == expected.height
    assert evaluation.event_results.height == 1


@pytest.mark.parametrize("value", [False, None])
def test_training_rejects_incomplete_horizons_even_when_labels_are_known(value):
    rows = training().with_columns(
        pl.lit(value, dtype=pl.Boolean).alias("has_full_horizon")
    )
    with pytest.raises(ValueError, match="fully observed horizons"):
        fit_logistic_regression(rows)


def test_training_rejects_unknown_labels_and_single_class_data():
    with pytest.raises(ValueError, match="Boolean labels"):
        fit_logistic_regression(
            training().with_columns(pl.lit(None, dtype=pl.Boolean).alias("target_60d"))
        )
    with pytest.raises(ValueError, match="both target classes"):
        fit_logistic_regression(
            training().with_columns(pl.lit(False).alias("target_60d"))
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("listening_share_7d", 1.1),
        ("previous_listening_count", -1),
        ("previous_listening_count", float("inf")),
        ("listening_share_7d", float("nan")),
        ("previous_listening_count", None),
    ],
)
def test_invalid_features_are_rejected_before_training(column, value):
    with pytest.raises(ValueError, match="features|Listening share"):
        fit_logistic_regression(training().with_columns(pl.lit(value).alias(column)))


def test_missing_baseline_candidates_are_rejected():
    with pytest.raises(ValueError, match="Missing scores"):
        score_logistic_regression(ShareScores(), scoring(), baseline().head(3))


def test_empty_scoring_preserves_the_evaluator_schema():
    result = score_logistic_regression(
        ShareScores(), scoring().head(0), baseline().head(0)
    )
    assert result.is_empty()
    assert result.schema["score"] == pl.Float64
    assert rank_scores(result, scoring().head(0)).is_empty()


def test_nonconverged_training_is_reported_as_a_failure(monkeypatch):
    def fail_to_converge(self, *args, **kwargs):
        warnings.warn("Iteration limit reached", ConvergenceWarning)

    monkeypatch.setattr(Pipeline, "fit", fail_to_converge)
    with pytest.raises(ConvergenceWarning):
        fit_logistic_regression(training())
