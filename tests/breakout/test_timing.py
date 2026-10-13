import datetime as dt
from dataclasses import replace

import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from sonora.breakout.evaluation import EvaluationPeriod, evaluate_scores
from sonora.breakout.evaluation.splits import build_walk_forward_folds
from sonora.breakout.modeling import LogisticFoldResult
from sonora.breakout.timing import AFTER_START, BEFORE_START, audit_detection_timing


@pytest.fixture
def timing_comparison():
    rows = [
        ("u", "training", dt.date(2021, 9, 1), 0.0, 10, True, 1.0),
        ("u", "training", dt.date(2021, 9, 5), 0.2, 10, True, 1.0),
        ("other", "training", dt.date(2021, 9, 1), 0.0, 10, True, 1.0),
    ]
    schedules = [
        ("u", 10, ["a", "partial", "x", "y", "z"]),
        ("u", 18, ["x", "y", "z", "a"]),
        ("u", 19, ["a", "x", "y", "z", "b"]),
        ("u", 20, ["x", "y", "a", "z"]),
        ("u", 21, ["b", "x", "y", "a", "z"]),
        ("other", 19, ["x", "y", "z", "a"]),
        ("other", 21, ["a", "x", "y", "z"]),
    ]
    for user, day, artists in schedules:
        for rank, artist in enumerate(artists, start=1):
            positive = artist in ["a", "b", "partial"]
            rows.append(
                (
                    user,
                    artist,
                    dt.date(2022, 1, day),
                    0.0 if day < 20 else 0.1,
                    10,
                    positive,
                    -float(rank),
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
            "score": pl.Float64,
        },
        orient="row",
    ).with_columns(
        (pl.col("scoring_date") + dt.timedelta(days=60)).alias("horizon_end_date"),
        pl.lit(True).alias("has_full_horizon"),
    )
    periods = [
        EvaluationPeriod("first", dt.date(2022, 1, 1), dt.date(2022, 1, 20), True),
        EvaluationPeriod("second", dt.date(2022, 1, 21), dt.date(2022, 3, 1), True),
    ]
    folds = [
        LogisticFoldResult(fold, Pipeline([]), fold.scoring)
        for fold in build_walk_forward_folds(data, periods)
    ]
    events = pl.DataFrame(
        [
            ("u", "a", dt.date(2022, 1, 20), dt.date(2022, 1, 26), dt.date(2022, 3, 2)),
            ("u", "b", dt.date(2022, 1, 20), dt.date(2022, 1, 26), dt.date(2022, 3, 2)),
            (
                "other",
                "a",
                dt.date(2022, 1, 18),
                dt.date(2022, 1, 24),
                dt.date(2022, 3, 2),
            ),
            (
                "u",
                "partial",
                dt.date(2022, 1, 10),
                dt.date(2022, 1, 12),
                dt.date(2022, 1, 15),
            ),
            (
                "u",
                "training",
                dt.date(2021, 9, 5),
                dt.date(2021, 9, 11),
                dt.date(2021, 9, 15),
            ),
            (
                "other",
                "training",
                dt.date(2021, 9, 10),
                dt.date(2021, 9, 16),
                dt.date(2021, 9, 20),
            ),
        ],
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "breakout_period_start": pl.Date,
            "first_important_period_end": pl.Date,
            "confirmation_date": pl.Date,
        },
        orient="row",
    )
    candidates = pl.concat([fold.fold.scoring for fold in folds])
    evaluation = evaluate_scores(
        candidates.select("user_id", "artist_id", "scoring_date", "score"),
        candidates,
        events,
        oos_start=periods[0].scoring_start,
        oos_end=periods[-1].scoring_end,
    )
    return folds, events, evaluation


def test_stage_boundary_fold_reuse_and_users_are_kept_separate(
    timing_comparison, monkeypatch
):
    folds, events, evaluation = timing_comparison

    def cannot_fit(*args, **kwargs):
        raise AssertionError("Timing audit must not fit a model")

    monkeypatch.setattr(Pipeline, "fit", cannot_fit)
    audit = audit_detection_timing({"baseline": evaluation}, folds, events)
    days = audit.event_days
    assert days.filter(pl.col("artist_id") == "partial").is_empty()
    assert days.filter(
        (pl.col("user_id") == "u") & (pl.col("scoring_date") == dt.date(2022, 1, 20))
    )["stage"].to_list() == [AFTER_START]
    assert days.filter(pl.col("user_id") == "other")["stage"].unique().to_list() == [
        AFTER_START
    ]
    assert (
        days.select("approach", "user_id", "artist_id", "scoring_date").n_unique()
        == days.height
    )
    for period in ["first", "second"]:
        training = audit.training_summary.filter(
            (pl.col("period") == period) & (pl.col("user_id") == "u")
        )
        assert training["positive_days"].to_list() == [1, 1]
        assert training["positive_day_share"].to_list() == [0.5, 0.5]
    assert audit.training_summary.filter(pl.col("user_id") == "other")[
        "positive_day_share"
    ].to_list() == [1.0, 1.0]
    before = audit.ranking_summary.filter(
        (pl.col("user_id") == "u") & (pl.col("stage") == BEFORE_START)
    ).row(0, named=True)
    assert before["positive_days"] == 4
    assert before["top3_days"] == 2
    assert before["top3_day_rate"] == pytest.approx(2 / 4)
    assert before["mean_event_top3_day_rate"] == pytest.approx((2 / 3 + 0) / 2)
    assert before["zero_share_fraction"] == 1.0


def test_same_candidates_and_event_cohort_are_required(timing_comparison):
    folds, events, evaluation = timing_comparison
    with pytest.raises(ValueError, match="same scoring candidates"):
        audit_detection_timing(
            {
                "baseline": evaluation,
                "model": replace(evaluation, rankings=evaluation.rankings.slice(1)),
            },
            folds,
            events,
        )
    with pytest.raises(ValueError, match="same event cohort"):
        audit_detection_timing(
            {
                "baseline": evaluation,
                "model": replace(
                    evaluation,
                    event_results=evaluation.event_results.with_columns(
                        pl.lit(False).alias("is_comparable")
                    ),
                ),
            },
            folds,
            events,
        )
    with pytest.raises(ValueError, match="exactly one fold"):
        audit_detection_timing({"baseline": evaluation}, folds + folds, events)


def test_missing_positive_event_and_invalid_horizon_are_rejected(timing_comparison):
    folds, events, evaluation = timing_comparison
    training = folds[0].fold.training.with_columns(pl.lit("missing").alias("artist_id"))
    changed = replace(folds[0], fold=replace(folds[0].fold, training=training))
    with pytest.raises(ValueError, match="Every positive row"):
        audit_detection_timing({"baseline": evaluation}, [changed, folds[1]], events)
    training = folds[0].fold.training.with_columns(
        pl.col("scoring_date").alias("horizon_end_date")
    )
    changed = replace(folds[0], fold=replace(folds[0].fold, training=training))
    with pytest.raises(ValueError, match="within their horizon"):
        audit_detection_timing({"baseline": evaluation}, [changed, folds[1]], events)


def test_no_comparable_events_returns_typed_empty_ranking_views(timing_comparison):
    folds, events, evaluation = timing_comparison
    full = audit_detection_timing({"baseline": evaluation}, folds, events)
    evaluation = replace(
        evaluation,
        event_results=evaluation.event_results.with_columns(
            pl.lit(False).alias("is_comparable")
        ),
    )
    empty = audit_detection_timing({"baseline": evaluation}, folds, events)
    for table in ["ranking_summary", "event_days"]:
        assert getattr(empty, table).is_empty()
        assert getattr(empty, table).schema == getattr(full, table).schema
