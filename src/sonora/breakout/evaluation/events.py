"""Opportunity-aware evaluation of distinct breakout events."""

import datetime as dt

import polars as pl

from sonora.breakout.evaluation.rankings import RANKING_KEYS
from sonora.breakout.targets import DEFAULT_HORIZON_DAYS

EVENT_KEYS = ("user_id", "artist_id")
EVENT_DATES = (
    "breakout_period_start",
    "first_important_period_end",
    "confirmation_date",
)
EVENT_COLUMNS = (*EVENT_KEYS, *EVENT_DATES)
TOP_K = 3


def evaluate_events(
    events: pl.DataFrame,
    rankings: pl.DataFrame,
    *,
    oos_start: dt.date,
    oos_end: dt.date,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pl.DataFrame:
    """Evaluate each event once against the continuous OOS ranking stream."""
    _validate_events(events)
    _validate_rankings(rankings)
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if oos_end < oos_start:
        raise ValueError("oos_end must not precede oos_start")

    base = events.select(EVENT_COLUMNS).with_columns(
        (pl.col("confirmation_date") - pl.duration(days=horizon_days)).alias(
            "evaluation_window_start"
        ),
        (pl.col("confirmation_date") - pl.duration(days=1)).alias(
            "evaluation_window_end"
        ),
    )
    opportunities = (
        base.join(rankings, on=EVENT_KEYS, how="inner", validate="1:m")
        .filter(
            pl.col("scoring_date").is_between(
                pl.col("evaluation_window_start"),
                pl.col("evaluation_window_end"),
            )
        )
        .group_by(EVENT_KEYS)
        .agg(
            pl.col("scoring_date").n_unique().alias("valid_opportunities"),
            pl.col("scoring_date")
            .filter(pl.col("scoring_date") < pl.col("breakout_period_start"))
            .n_unique()
            .alias("pre_start_opportunities"),
            pl.col("scoring_date")
            .filter(pl.col("rank") <= TOP_K)
            .n_unique()
            .alias("top3_appearances"),
            pl.col("scoring_date")
            .filter(pl.col("rank") <= TOP_K)
            .unique()
            .sort()
            .alias("_top3_dates"),
        )
    )
    result = (
        base.join(opportunities, on=EVENT_KEYS, how="left", validate="1:1")
        .with_columns(
            pl.col(
                "valid_opportunities",
                "pre_start_opportunities",
                "top3_appearances",
            )
            .fill_null(0)
            .cast(pl.UInt32)
        )
        .with_columns(
            pl.min_horizontal(pl.lit(2), "valid_opportunities")
            .cast(pl.UInt32)
            .alias("required_top3_appearances")
        )
        .with_columns(
            pl.when(
                (pl.col("required_top3_appearances") > 0)
                & (pl.col("top3_appearances") >= pl.col("required_top3_appearances"))
            )
            .then(
                pl.col("_top3_dates").list.get(
                    pl.col("required_top3_appearances") - 1,
                    null_on_oob=True,
                )
            )
            .otherwise(None)
            .alias("detection_date")
        )
        .with_columns(
            (
                (pl.col("evaluation_window_start") >= pl.lit(oos_start))
                & (pl.col("evaluation_window_end") <= pl.lit(oos_end))
            ).alias("is_comparable"),
            (pl.col("valid_opportunities") > 0).alias("is_scoreable"),
            (pl.col("pre_start_opportunities") > 0).alias("is_anticipatable"),
            pl.col("detection_date").is_not_null().alias("detected"),
            (
                pl.col("confirmation_date").dt.year().cast(pl.String)
                + pl.when(pl.col("confirmation_date").dt.month() <= 6)
                .then(pl.lit(" H1"))
                .otherwise(pl.lit(" H2"))
            ).alias("confirmation_period"),
        )
        .with_columns(
            pl.when(~pl.col("is_scoreable"))
            .then(pl.lit("Unscoreable"))
            .when(~pl.col("detected"))
            .then(pl.lit("Missed"))
            .when(pl.col("detection_date") < pl.col("breakout_period_start"))
            .then(pl.lit("Anticipatory"))
            .when(pl.col("detection_date") <= pl.col("first_important_period_end"))
            .then(pl.lit("Early"))
            .otherwise(pl.lit("Late"))
            .alias("timing_stage"),
            (pl.col("breakout_period_start") - pl.col("detection_date"))
            .dt.total_days()
            .alias("days_to_breakout_start"),
            (pl.col("confirmation_date") - pl.col("detection_date"))
            .dt.total_days()
            .alias("days_to_confirmation"),
        )
        .drop("_top3_dates")
        .select(
            *EVENT_KEYS,
            *EVENT_DATES,
            "evaluation_window_start",
            "evaluation_window_end",
            "valid_opportunities",
            "pre_start_opportunities",
            "required_top3_appearances",
            "top3_appearances",
            "is_comparable",
            "is_scoreable",
            "is_anticipatable",
            "detected",
            "detection_date",
            "timing_stage",
            "days_to_breakout_start",
            "days_to_confirmation",
            "confirmation_period",
        )
        .sort("confirmation_date", "user_id", "artist_id")
    )
    return result


def _validate_events(events: pl.DataFrame) -> None:
    missing = set(EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if events.select(EVENT_KEYS).n_unique() != events.height:
        raise ValueError("Breakout events must be unique per user and artist")
    if any(events[column].null_count() for column in EVENT_COLUMNS):
        raise ValueError("Event identity and date columns cannot contain nulls")
    if any(events.schema[column] != pl.Date for column in EVENT_DATES):
        raise ValueError("Breakout event dates must use Date values")
    invalid = events.filter(
        (pl.col("breakout_period_start") > pl.col("first_important_period_end"))
        | (pl.col("first_important_period_end") >= pl.col("confirmation_date"))
    )
    if invalid.height:
        raise ValueError("Breakout event dates are out of order")


def _validate_rankings(rankings: pl.DataFrame) -> None:
    required = {*RANKING_KEYS, "rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    if rankings.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if rankings.select(RANKING_KEYS).n_unique() != rankings.height:
        raise ValueError("Ranking rows must be unique per candidate and day")
    if any(rankings[column].null_count() for column in required):
        raise ValueError("Ranking keys and ranks cannot contain nulls")
