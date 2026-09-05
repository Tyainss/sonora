"""Daily listening counts and explicit history limits for breakout analysis."""

import polars as pl

BOUND_COLUMNS = (
    "user_id",
    "first_date",
    "warmup_end",
    "score_start",
    "last_complete_date",
)


def aggregate_daily_listening(
    events: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Count listening by UTC day, once per user and once per artist."""
    timestamp = events.collect_schema()["listened_at"]
    if not isinstance(timestamp, pl.Datetime) or timestamp.time_zone != "UTC":
        raise ValueError("listened_at must contain UTC timestamps")
    daily = events.with_columns(pl.col("listened_at").dt.date().alias("date"))
    return (
        daily.group_by("user_id", "date")
        .agg(pl.len().alias("user_scrobbles"))
        .collect(),
        daily.group_by("user_id", "artist_id", "date")
        .agg(pl.len().alias("artist_scrobbles"))
        .collect(),
    )


def historical_user_bounds(daily_user: pl.DataFrame) -> pl.DataFrame:
    """Apply the historical study's first-year exclusion and 21-day buffer.

    This is an analysis preset, not a production rule for users with short
    histories. Detection and target building accept caller-supplied bounds.
    """
    return (
        daily_user.group_by("user_id")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_observed_date"),
        )
        .with_columns(
            (pl.col("first_date") + pl.duration(days=365)).alias("warmup_end"),
            (pl.col("first_date") + pl.duration(days=386)).alias("score_start"),
            (pl.col("last_observed_date") - pl.duration(days=1)).alias(
                "last_complete_date"
            ),
        )
        .sort("user_id")
    )


def validate_user_bounds(bounds: pl.DataFrame, *, period_days: int = 7) -> None:
    missing = set(BOUND_COLUMNS) - set(bounds.columns)
    if missing:
        raise ValueError(f"Missing user bounds: {sorted(missing)}")
    if bounds["user_id"].n_unique() != bounds.height:
        raise ValueError("Each user must have exactly one set of bounds")
    if any(bounds[column].null_count() for column in BOUND_COLUMNS):
        raise ValueError("User bounds cannot contain null values")
    if any(bounds.schema[column] != pl.Date for column in BOUND_COLUMNS[1:]):
        raise ValueError("User bounds must use Date columns")
    if bounds.filter(
        (pl.col("warmup_end") <= pl.col("first_date"))
        | (
            pl.col("score_start")
            < pl.col("warmup_end") + pl.duration(days=3 * period_days)
        )
    ).height:
        raise ValueError(
            "Scoring must start after the initial history period and its full follow-up"
        )


def available_user_bounds(bounds: pl.DataFrame) -> pl.DataFrame:
    """Keep users whose declared historical exclusion can be fully observed."""
    return bounds.filter(
        pl.col("score_start") <= pl.col("last_complete_date") + pl.duration(days=1)
    )
