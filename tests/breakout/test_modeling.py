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
    RECENCY_FEATURES,
    fit_boosted_trees,
    fit_logistic_regression,
    model_features,
    run_boosted_tree_walk_forward,
    run_logistic_walk_forward,
    score_model,
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


def share_scores():
    model = fit_logistic_regression(training())
    model.decision_function = lambda inputs: inputs[:, 0]
    return model


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
    result = score_model(model, rows, baseline())
    altered = rows.with_columns(
        pl.lit(True).alias("target_60d"), pl.lit(-1000).alias("unrelated_metadata")
    )
    other = score_model(model, altered, baseline())
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
    result = score_model(share_scores(), rows, baseline().reverse())
    assert result["artist_id"].to_list() == ["c", "b", "a", "a"]
    assert result["score"].to_list() == [-1.0, -2.0, -3.0, -1.0]
    assert result["baseline_score"].to_list() == [-3.0, -1.0, -2.0, -1.0]
    assert rank_scores(result, rows)["rank"].to_list() == [1, 2, 3, 1]
    shuffled = score_model(share_scores(), rows.reverse(), baseline())
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
        score_model(share_scores(), scoring(), baseline().head(3))


def test_empty_scoring_preserves_the_evaluator_schema():
    result = score_model(share_scores(), scoring().head(0), baseline().head(0))
    assert result.is_empty()
    assert result.schema["score"] == pl.Float64
    assert rank_scores(result, scoring().head(0)).is_empty()


def test_nonconverged_training_is_reported_as_a_failure(monkeypatch):
    def fail_to_converge(self, *args, **kwargs):
        warnings.warn("Iteration limit reached", ConvergenceWarning)

    monkeypatch.setattr(Pipeline, "fit", fail_to_converge)
    with pytest.raises(ConvergenceWarning):
        fit_logistic_regression(training())


def recency_training():
    return training().with_columns(
        pl.Series(
            "prior_listening_gap_days",
            [None, 8.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
        )
    )


def test_recency_median_and_scaling_use_training_only_and_keep_missing_artists():
    rows = recency_training()
    model = fit_logistic_regression(rows, recency=True)
    transform = model.named_steps["transform"]
    assert transform.named_transformers_["prior_gap"].statistics_[0] == 40
    terms = np.column_stack(
        [
            rows["listening_share_7d"].to_numpy(),
            np.log1p(rows["previous_listening_count"].to_numpy()),
            [40, 8, 10, 20, 40, 80, 160, 320],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert_allclose(
        transform.transform(rows.select(RECENCY_FEATURES).to_numpy()), terms
    )
    assert_allclose(model.named_steps["scale"].mean_, terms.mean(axis=0))
    assert model_features(model) == RECENCY_FEATURES
    assert (
        model.named_steps["classifier"].get_params()
        == fit_logistic_regression(rows).named_steps["classifier"].get_params()
    )
    means = model.named_steps["scale"].mean_.copy()
    candidates = scoring().with_columns(
        pl.Series("previous_listening_count", [0, 0, 2, 0]),
        pl.Series("prior_listening_gap_days", [None, None, 10_000.0, None]),
    )
    result = score_model(model, candidates, baseline())
    assert result.height == candidates.height
    assert np.isfinite(result["model_score"].to_numpy()).all()
    assert transform.named_transformers_["prior_gap"].statistics_[0] == 40
    assert_allclose(model.named_steps["scale"].mean_, means)


def test_recency_indicator_is_present_when_missing_history_first_appears_at_scoring():
    rows = recency_training().with_columns(
        pl.col("previous_listening_count").clip(lower_bound=1),
        pl.col("prior_listening_gap_days").fill_null(8),
    )
    model = fit_logistic_regression(rows, recency=True)
    assert model.named_steps["classifier"].n_features_in_ == 4
    candidates = scoring().with_columns(
        pl.lit(0).alias("previous_listening_count"),
        pl.lit(None, dtype=pl.Float64).alias("prior_listening_gap_days"),
    )
    terms = model.named_steps["transform"].transform(
        candidates.select(RECENCY_FEATURES).to_numpy()
    )
    assert terms.shape == (4, 4)
    assert terms[:, 3].tolist() == [1, 1, 1, 1]
    assert score_model(model, candidates, baseline()).height == 4


@pytest.mark.parametrize("value", [None, 7, float("inf"), float("nan")])
def test_invalid_prior_gap_or_missing_existing_history_is_rejected(value):
    rows = recency_training().with_columns(
        pl.Series(
            "prior_listening_gap_days",
            [None, value, 10, 20, 40, 80, 160, 320],
            dtype=pl.Float64,
        )
    )
    with pytest.raises(ValueError, match="Gaps must"):
        fit_logistic_regression(rows, recency=True)


def test_recency_requires_an_observed_training_gap_and_excludes_interaction():
    rows = recency_training().with_columns(
        pl.lit(0).alias("previous_listening_count"),
        pl.lit(None, dtype=pl.Float64).alias("prior_listening_gap_days"),
    )
    with pytest.raises(ValueError, match="observed training gap"):
        fit_logistic_regression(rows, recency=True)
    with pytest.raises(ValueError, match="separate model comparisons"):
        fit_logistic_regression(recency_training(), interaction=True, recency=True)


def test_trees_use_raw_inputs_native_missingness_and_fixed_training_rounds():
    rows = recency_training()
    model = fit_boosted_trees(rows)
    classifier = model.named_steps["classifier"]
    assert list(model.named_steps) == ["classifier"]
    assert model_features(model) == RECENCY_FEATURES
    assert classifier.n_features_in_ == 3
    assert classifier.n_iter_ == 100
    assert classifier.do_early_stopping_ is False
    assert classifier.max_depth == 3
    assert classifier.max_leaf_nodes == 7
    assert classifier.min_samples_leaf == 100
    candidates = scoring().with_columns(
        pl.lit(0).alias("previous_listening_count"),
        pl.lit(None, dtype=pl.Float64).alias("prior_listening_gap_days"),
    )
    result = score_model(model, candidates, baseline())
    assert result.height == candidates.height
    assert np.isfinite(result["model_score"].to_numpy()).all()
    assert result["score"].to_list() == [-1, -2, -3, -1]
    with pytest.raises(ValueError, match="observed training gap"):
        fit_boosted_trees(
            rows.with_columns(
                pl.lit(0).alias("previous_listening_count"),
                pl.lit(None, dtype=pl.Float64).alias("prior_listening_gap_days"),
            )
        )


@pytest.mark.parametrize(
    "column,value",
    [("has_full_horizon", False), ("target_60d", None), ("target_60d", True)],
)
def test_tree_training_rejects_unknown_or_unresolved_labels_and_single_class(
    column, value
):
    rows = recency_training().with_columns(
        pl.lit(value, dtype=pl.Boolean).alias(column)
    )
    with pytest.raises(ValueError, match="Training requires"):
        fit_boosted_trees(rows)


def test_tree_walk_forward_uses_the_same_resolved_training_boundaries():
    rows = recency_training().with_columns(
        pl.Series("user_id", ["u"] * 8),
        pl.Series("artist_id", [f"a{i}" for i in range(8)]),
        pl.Series(
            "scoring_date",
            [dt.date(2021, 9, 1)] * 4
            + [dt.date(2022, 1, 1)] * 2
            + [dt.date(2022, 7, 1)] * 2,
        ),
    )
    rows = rows.with_columns(
        (pl.col("scoring_date") + dt.timedelta(days=60)).alias("horizon_end_date")
    )
    periods = [
        EvaluationPeriod("first", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False),
        EvaluationPeriod("second", dt.date(2022, 7, 1), dt.date(2022, 7, 1), True),
    ]
    candidates = rows.filter(pl.col("scoring_date") >= dt.date(2022, 1, 1))
    baseline_scores = candidates.select(
        *KEYS, pl.col("listening_share_7d").alias("score")
    )
    expected = run_logistic_walk_forward(rows, periods, baseline_scores)
    actual = run_boosted_tree_walk_forward(rows, periods, baseline_scores)
    for tree, logistic in zip(actual, expected, strict=True):
        assert tree.fold.training.equals(logistic.fold.training)
        assert tree.fold.scoring.equals(logistic.fold.scoring)
        assert tree.model.named_steps["classifier"].n_iter_ == 100
