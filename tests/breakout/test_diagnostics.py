import datetime as dt
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from sonora.breakout.diagnostics import (
    diagnose_lost_detections,
    review_detection_changes,
)
from sonora.breakout.evaluation import EvaluationPeriod, evaluate_scores
from sonora.breakout.modeling import (
    MODEL_FEATURES,
    fit_logistic_regression,
    run_logistic_walk_forward,
    score_logistic_regression,
)

KEYS = ["user_id", "artist_id", "scoring_date"]


@pytest.fixture
def comparison():
    rows = [
        ("u", f"training-{i}", dt.date(2021, 12, 1), share, count, i >= 4)
        for i, (share, count) in enumerate(
            zip(
                [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
                [0, 1, 2, 3, 100, 101, 102, 103],
                strict=True,
            )
        )
    ]
    rows.extend(
        [
            ("u", "later-negative", dt.date(2022, 4, 30), 0.15, 200, False),
            ("u", "later-positive", dt.date(2022, 4, 30), 0.35, 300, True),
        ]
    )
    confirmation = dt.date(2022, 8, 28)
    dates = [
        dt.date(2022, 6, 28),
        dt.date(2022, 6, 30),
        dt.date(2022, 7, 1),
        confirmation,
    ]
    for user in ["u", "other"]:
        for day in dates:
            for artist, share, count in [
                ("first", 0.95, 1000),
                ("second", 0.9, 500),
                ("lost", 0.8, 0),
                ("promoted", 0.2, 100),
            ]:
                rows.append(
                    (
                        user,
                        artist,
                        day,
                        share,
                        count,
                        artist == "lost"
                        and day < confirmation <= day + dt.timedelta(days=60),
                    )
                )
    data = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
            "listening_share_7d": pl.Float64,
            "previous_listening_count": pl.Int64,
            "target_60d": pl.Boolean,
        },
        orient="row",
    ).with_columns(
        (pl.col("scoring_date") + dt.timedelta(days=60)).alias("horizon_end_date"),
        pl.lit(True).alias("has_full_horizon"),
    )
    candidates = data.filter(pl.col("scoring_date") >= dates[0])
    scores = candidates.select(*KEYS, pl.col("listening_share_7d").alias("score"))
    periods = [
        EvaluationPeriod("2022 H1", dates[0], dt.date(2022, 6, 30), True),
        EvaluationPeriod("2022 H2", dt.date(2022, 7, 1), confirmation, True),
    ]
    folds = run_logistic_walk_forward(data, periods, scores)
    events = pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["lost"],
            "breakout_period_start": [dt.date(2022, 8, 1)],
            "first_important_period_end": [dt.date(2022, 8, 7)],
            "confirmation_date": [confirmation],
        }
    )
    baseline, model = [
        evaluate_scores(
            stream, candidates, events, oos_start=dates[0], oos_end=confirmation
        )
        for stream in [scores, pl.concat([result.scores for result in folds])]
    ]
    assert baseline.event_summary["detected_events"].item() == 1
    assert model.event_summary["detected_events"].item() == 0
    return baseline, model, folds


def test_window_user_and_promoted_artist_are_identified_with_original_fold_scores(
    comparison, monkeypatch
):
    baseline, model, folds = comparison
    means = [result.model.named_steps["scale"].mean_.copy() for result in folds]

    def cannot_refit(*args, **kwargs):
        raise AssertionError("Diagnosis must not refit")

    monkeypatch.setattr(Pipeline, "fit", cannot_refit)
    diagnostic = diagnose_lost_detections(baseline, model, folds)
    summary = diagnostic.summary.row(0, named=True)
    assert summary["valid_opportunities"] == 2
    assert summary["required_top3_appearances"] == 2
    assert summary["baseline_top3_days"] == 2
    assert summary["model_top3_days"] == 0
    assert diagnostic.daily_ranks["scoring_date"].to_list() == [
        dt.date(2022, 6, 30),
        dt.date(2022, 7, 1),
    ]
    promoted = diagnostic.displacements
    assert promoted.height == 2
    assert promoted["user_id"].unique().to_list() == ["u"]
    assert promoted["promoted_artist_id"].unique().to_list() == ["promoted"]
    assert promoted["promoted_baseline_rank"].to_list() == [4, 4]
    assert promoted["promoted_model_rank"].to_list() == [3, 3]
    assert (
        promoted["promoted_listening_share_7d"].max()
        < promoted["listening_share_7d"].min()
    )
    assert promoted["previous_count_advantage"].min() > 0
    assert promoted["model_advantage"].min() > 0
    np.testing.assert_allclose(
        promoted["share_advantage"] + promoted["previous_count_advantage"],
        promoted["model_advantage"],
    )
    assert not np.array_equal(means[0], means[1])
    for result, mean in zip(folds, means, strict=True):
        rows = diagnostic.daily_ranks.filter(pl.col("period") == result.fold.period)
        np.testing.assert_allclose(
            rows["model_score"],
            result.model.decision_function(rows.select(MODEL_FEATURES).to_numpy()),
        )
        np.testing.assert_array_equal(result.model.named_steps["scale"].mean_, mean)


def test_no_lost_detections_returns_typed_empty_views(comparison):
    baseline, model, folds = comparison
    full = diagnose_lost_detections(baseline, model, folds)
    empty = diagnose_lost_detections(baseline, baseline, folds)
    for name in ["summary", "daily_ranks", "displacements"]:
        assert getattr(empty, name).is_empty()
        assert getattr(empty, name).schema == getattr(full, name).schema


def test_missing_fitted_fold_cannot_silently_drop_diagnostic_days(comparison):
    baseline, model, folds = comparison
    with pytest.raises(ValueError, match="exactly one fitted fold"):
        diagnose_lost_detections(baseline, model, folds[:1])


def test_changed_model_cannot_explain_a_different_recorded_score(comparison):
    baseline, model, folds = comparison
    folds[0].model.named_steps["classifier"].coef_[0, 1] += 1.0
    with pytest.raises(ValueError, match="do not match"):
        diagnose_lost_detections(baseline, model, folds)


def test_mismatched_daily_candidates_are_rejected(comparison):
    baseline, model, folds = comparison
    model = replace(model, rankings=model.rankings.slice(1))
    with pytest.raises(ValueError, match="same candidates"):
        diagnose_lost_detections(baseline, model, folds)


@pytest.mark.parametrize("same_day", [False, True])
def test_event_review_keeps_both_losses_and_unchanged_events_with_null_miss_context(
    comparison, same_day
):
    baseline, model, folds = comparison
    reviewed = review_detection_changes(
        baseline, baseline if same_day else model, folds
    )
    assert reviewed.height == 1
    row = reviewed.row(0, named=True)
    assert row["change"] == ("Same day" if same_day else "Lost detection")
    assert row["first_eligible_date"] == dt.date(2022, 6, 30)
    assert row["history_at_entry"] == "No earlier listens"
    assert row["entry_previous_count"] == 0
    assert row["baseline_detection_previous_count"] == 0
    assert row["model_detection_previous_count"] == (0 if same_day else None)


def test_interaction_contributions_reconstruct_original_fold_scores(comparison):
    baseline, _, folds = comparison
    results = []
    for original in folds:
        model = fit_logistic_regression(original.fold.training, interaction=True)
        scores = score_logistic_regression(
            model,
            original.fold.scoring,
            baseline.rankings.select(*KEYS, "score").join(
                original.fold.scoring.select(KEYS), on=KEYS, how="semi"
            ),
        )
        results.append(replace(original, model=model, scores=scores))
    model = evaluate_scores(
        pl.concat([r.scores for r in results]),
        pl.concat([r.fold.scoring for r in results]),
        baseline.event_results,
        oos_start=folds[0].fold.scoring_start,
        oos_end=folds[-1].fold.scoring_end,
    )
    reviewed = review_detection_changes(baseline, model, results)
    assert reviewed.height == 1
    diagnostic = diagnose_lost_detections(baseline, model, results)
    assert diagnostic.daily_ranks.height > 0
    values = diagnostic.daily_ranks
    np.testing.assert_allclose(
        values["share_contribution"]
        + values["previous_count_contribution"]
        + values["interaction_contribution"]
        + values["intercept"],
        values["model_score"],
    )
