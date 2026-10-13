"""Retrospective timing of positive labels and held-out breakout rankings."""

from dataclasses import dataclass

import polars as pl

from sonora.breakout.evaluation.evaluate import EvaluationResult
from sonora.breakout.evaluation.events import EVENT_COLUMNS, EVENT_KEYS, TOP_K
from sonora.breakout.features import CANDIDATE_KEYS
from sonora.breakout.modeling import MODEL_FEATURES, LogisticFoldResult

BEFORE_START = "Before start"
AFTER_START = "Started; awaiting confirmation"


@dataclass(frozen=True)
class TimingAudit:
    """Fold training composition and comparable-event ranking diagnostics."""

    training_summary: pl.DataFrame
    ranking_summary: pl.DataFrame
    event_days: pl.DataFrame


def audit_detection_timing(
    evaluations: dict[str, EvaluationResult],
    fold_results: list[LogisticFoldResult],
    events: pl.DataFrame,
) -> TimingAudit:
    """Separate positive days before start from start through confirmation-1.

    Training summaries describe each expanding fold separately. Ranking views
    use only fully comparable, scoreable events and their eligible days inside
    the existing evaluation window. Stage dates are retrospective outcomes;
    they are never supplied to the fitted models. Rates describe artist-days,
    not event detection, and per-event means give each represented event equal
    weight within its stage. Users are summarized separately.
    """
    if not evaluations or not fold_results:
        raise ValueError("Timing audit requires evaluations and fitted folds")
    event_dates = events.select(EVENT_COLUMNS)
    if event_dates.select(EVENT_KEYS).n_unique() != event_dates.height:
        raise ValueError("Timing events must be unique per user and artist")
    if any(event_dates[column].null_count() for column in EVENT_COLUMNS):
        raise ValueError("Timing event identity and dates cannot contain nulls")
    scoring = pl.concat([result.fold.scoring for result in fold_results])
    keys = scoring.select(CANDIDATE_KEYS).sort(CANDIDATE_KEYS)
    if keys.n_unique() != keys.height:
        raise ValueError("Timing scoring rows must belong to exactly one fold")
    reference = next(iter(evaluations.values())).event_results
    comparison_columns = [
        *EVENT_COLUMNS,
        "is_comparable",
        "is_scoreable",
        "evaluation_window_start",
        "evaluation_window_end",
    ]
    reference = reference.select(comparison_columns).sort(EVENT_KEYS)
    if not reference.select(EVENT_COLUMNS).equals(
        event_dates.join(reference.select(EVENT_KEYS), on=EVENT_KEYS, how="semi").sort(
            EVENT_KEYS
        )
    ):
        raise ValueError("Timing events must match the evaluation event dates")
    cohort = reference.filter(pl.col("is_comparable") & pl.col("is_scoreable"))
    scoring_days = (
        _positive_timing_rows(
            scoring.join(cohort.select(EVENT_KEYS), on=EVENT_KEYS, how="semi"),
            cohort.select(EVENT_COLUMNS),
        )
        .join(
            cohort.select(
                *EVENT_KEYS, "evaluation_window_start", "evaluation_window_end"
            ),
            on=EVENT_KEYS,
            validate="m:1",
        )
        .filter(
            pl.col("scoring_date").is_between(
                pl.col("evaluation_window_start"), pl.col("evaluation_window_end")
            )
        )
    )
    training = []
    for result in fold_results:
        rows = _positive_timing_rows(result.fold.training, event_dates)
        training.append(
            rows.group_by("user_id", "stage")
            .agg(*_positive_summary())
            .with_columns(
                (
                    pl.col("positive_days")
                    / pl.col("positive_days").sum().over("user_id")
                ).alias("positive_day_share"),
                pl.lit(result.fold.period).alias("period"),
            )
        )
    ranked_days = []
    for name, evaluation in evaluations.items():
        if not keys.equals(
            evaluation.rankings.select(CANDIDATE_KEYS).sort(CANDIDATE_KEYS)
        ):
            raise ValueError("Timing evaluations must use the same scoring candidates")
        if not reference.equals(
            evaluation.event_results.select(comparison_columns).sort(EVENT_KEYS)
        ):
            raise ValueError("Timing evaluations must use the same event cohort")
        ranked_days.append(
            scoring_days.join(
                evaluation.rankings.select(*CANDIDATE_KEYS, "rank"),
                on=CANDIDATE_KEYS,
                validate="1:1",
            ).with_columns(pl.lit(name).alias("approach"))
        )
    days = pl.concat(ranked_days)
    groups = ["approach", "user_id", "stage"]
    event_rates = (
        days.group_by(*groups, "artist_id")
        .agg((pl.col("rank") <= TOP_K).mean().alias("top3_day_rate"))
        .group_by(groups)
        .agg(pl.col("top3_day_rate").mean().alias("mean_event_top3_day_rate"))
    )
    rankings = (
        days.group_by(groups)
        .agg(
            *_positive_summary(),
            (pl.col("rank") <= TOP_K).sum().alias("top3_days"),
            (pl.col("rank") <= TOP_K).mean().alias("top3_day_rate"),
            pl.col("rank").median().alias("median_rank"),
        )
        .join(event_rates, on=groups, validate="1:1")
        .sort(groups)
    )
    return TimingAudit(
        training_summary=pl.concat(training).sort("period", "user_id", "stage"),
        ranking_summary=rankings,
        event_days=days.sort("approach", *EVENT_KEYS, "scoring_date"),
    )


def _positive_timing_rows(rows: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    positive = (
        rows.filter(pl.col("target_60d"))
        .select(*CANDIDATE_KEYS, "horizon_end_date", *MODEL_FEATURES)
        .join(events, on=EVENT_KEYS, how="left", validate="m:1")
    )
    if positive["breakout_period_start"].null_count():
        raise ValueError("Every positive row requires its breakout event")
    if positive.filter(
        (pl.col("confirmation_date") <= pl.col("scoring_date"))
        | (pl.col("confirmation_date") > pl.col("horizon_end_date"))
    ).height:
        raise ValueError("Positive rows must precede confirmation within their horizon")
    return positive.with_columns(
        pl.when(pl.col("scoring_date") < pl.col("breakout_period_start"))
        .then(pl.lit(BEFORE_START))
        .otherwise(pl.lit(AFTER_START))
        .alias("stage")
    )


def _positive_summary() -> list[pl.Expr]:
    return [
        pl.len().alias("positive_days"),
        pl.col("artist_id").n_unique().alias("breakout_artists"),
        (pl.col("listening_share_7d") == 0).sum().alias("zero_share_days"),
        (pl.col("listening_share_7d") == 0).mean().alias("zero_share_fraction"),
        pl.col("listening_share_7d").median().alias("median_share"),
        pl.col("previous_listening_count").median().alias("median_previous_count"),
    ]
