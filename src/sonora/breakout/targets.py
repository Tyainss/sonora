"""Point-in-time eligibility and daily first-breakout labels."""

import polars as pl

from sonora.breakout.history import available_user_bounds, validate_user_bounds

DEFAULT_RECENCY_DAYS = 90
DEFAULT_HORIZON_DAYS = 60


def build_listening_gaps(
    daily_artist: pl.DataFrame,
    events: pl.DataFrame,
    established: pl.DataFrame,
    user_bounds: pl.DataFrame,
) -> pl.DataFrame:
    """Retain earlier history while splitting listening into non-overlapping ranges."""
    validate_user_bounds(user_bounds)
    if (
        daily_artist.select("user_id", "artist_id", "date").unique().height
        != daily_artist.height
    ):
        raise ValueError(
            "Daily artist listening must have unique user, artist, date rows"
        )
    return (
        daily_artist.join(
            available_user_bounds(user_bounds), on="user_id", validate="m:1"
        )
        .filter(
            pl.col("date").is_between(
                pl.col("first_date"), pl.col("last_complete_date")
            )
        )
        .with_columns(
            pl.col("date")
            .min()
            .over("user_id", "artist_id")
            .alias("first_listen_date"),
            pl.col("date")
            .shift(-1)
            .over(["user_id", "artist_id"], order_by="date")
            .alias("next_listen"),
            (pl.col("last_complete_date") + pl.duration(days=1)).alias(
                "known_confirmation_through"
            ),
            (pl.col("last_complete_date") + pl.duration(days=1)).alias("score_end"),
        )
        .join(
            established.select("user_id", "artist_id"),
            on=["user_id", "artist_id"],
            how="anti",
        )
        .join(
            events.select("user_id", "artist_id", "confirmation_date"),
            on=["user_id", "artist_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.when(pl.col("confirmation_date") <= pl.col("known_confirmation_through"))
            .then(pl.col("confirmation_date"))
            .otherwise(None)
            .alias("confirmation_date")
        )
    )


def eligible_segments(gaps: pl.DataFrame, recency_days: int) -> pl.DataFrame:
    """Return disjoint eligible ranges; a listen today only affects tomorrow."""
    if recency_days < 1:
        raise ValueError("recency_days must be positive")
    return gaps.with_columns(
        pl.max_horizontal(pl.col("date") + pl.duration(days=1), "score_start").alias(
            "eligible_start"
        ),
        pl.min_horizontal(
            pl.col("date") + pl.duration(days=recency_days),
            "next_listen",
            pl.col("confirmation_date") - pl.duration(days=1),
            "score_end",
        ).alias("eligible_end"),
    ).filter(pl.col("eligible_start") <= pl.col("eligible_end"))


def make_daily_targets(
    daily_artist: pl.DataFrame,
    events: pl.DataFrame,
    established: pl.DataFrame,
    user_bounds: pl.DataFrame,
    *,
    recency_days: int = DEFAULT_RECENCY_DAYS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pl.LazyFrame:
    """Build eligible daily rows, retaining unknown outcomes at the history end.

    The saved target is explicitly a 60-day target. Smaller comparison horizons
    belong in analysis; they must not be written under the same column name.
    """
    if horizon_days != DEFAULT_HORIZON_DAYS:
        raise ValueError("target_60d requires a 60-day horizon")
    gaps = build_listening_gaps(daily_artist, events, established, user_bounds)
    return (
        eligible_segments(gaps, recency_days)
        .lazy()
        .with_columns(
            pl.date_ranges("eligible_start", "eligible_end", interval="1d").alias(
                "scoring_date"
            )
        )
        .explode("scoring_date", empty_as_null=False)
        .with_columns(
            (pl.col("scoring_date") + pl.duration(days=horizon_days)).alias(
                "horizon_end_date"
            )
        )
        .with_columns(
            (pl.col("horizon_end_date") <= pl.col("known_confirmation_through")).alias(
                "has_full_horizon"
            )
        )
        .with_columns(
            pl.when(
                (pl.col("confirmation_date") > pl.col("scoring_date"))
                & (pl.col("confirmation_date") <= pl.col("horizon_end_date"))
            )
            .then(True)
            .when(pl.col("has_full_horizon"))
            .then(False)
            .otherwise(None)
            .cast(pl.Boolean)
            .alias("target_60d")
        )
        .select(
            "user_id",
            "artist_id",
            "scoring_date",
            "first_listen_date",
            pl.col("date").alias("last_listen_date"),
            "horizon_end_date",
            "target_60d",
            "has_full_horizon",
        )
        .sort("user_id", "scoring_date", "artist_id")
    )
