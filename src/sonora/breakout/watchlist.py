"""Point-in-time breakout watchlists from curated listening history."""

from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from sonora.breakout.baselines import score_recent_listening_share
from sonora.breakout.events import detect_first_breakouts
from sonora.breakout.history import aggregate_daily_listening, validate_user_bounds
from sonora.breakout.targets import eligible_candidates_for_date
from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

WATCHLIST_SCHEMA = {
    "user_id": pl.String,
    "scoring_date": pl.Date,
    "rank": pl.Int64,
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "listening_share_7d": pl.Float64,
    "artist_scrobbles_7d": pl.Int64,
    "user_scrobbles_7d": pl.Int64,
    "latest_listened_at": pl.Datetime("us", "UTC"),
    "prior_scrobbles": pl.Int64,
}


def get_breakout_watchlist(
    user_id: str,
    scoring_date: date,
    *,
    user_bounds: pl.DataFrame,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> pl.DataFrame:
    """Return up to three eligible artists ranked by prior seven-day listening share.

    History is cut at midnight UTC on scoring_date. The declared
    last_complete_date must cover the previous day. User history bounds
    are supplied by the caller.
    """
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a nonempty string")
    if not isinstance(scoring_date, date) or isinstance(scoring_date, datetime):
        raise TypeError("scoring_date must be a date")

    validate_user_bounds(user_bounds)
    selected = user_bounds.filter(pl.col("user_id") == user_id)
    if selected.height != 1:
        raise ValueError(f"No history bounds available for user {user_id!r}")
    record = selected.row(0, named=True)
    if scoring_date < record["score_start"]:
        raise ValueError("scoring_date precedes the user's scoring start")
    if scoring_date > record["last_complete_date"] + timedelta(days=1):
        raise ValueError("Listening history is not declared complete through D-1")

    cutoff = datetime.combine(scoring_date, time.min, tzinfo=UTC)
    listening = (
        pl.scan_parquet(paths.curated_listening_events)
        .filter((pl.col("user_id") == user_id) & (pl.col("listened_at") < cutoff))
        .select("user_id", "artist_id", "listened_at")
        .collect()
    )
    daily_user, daily_artist = aggregate_daily_listening(listening.lazy())
    if daily_user.is_empty() or daily_user["date"].min() != record["first_date"]:
        raise ValueError("User history must begin on the declared first_date")

    # Detection and eligibility see only completed days before scoring_date.
    bounded = selected.with_columns(
        pl.lit(scoring_date - timedelta(days=1))
        .cast(pl.Date)
        .alias("last_complete_date")
    )
    confirmed, established = detect_first_breakouts(daily_user, daily_artist, bounded)
    candidates = eligible_candidates_for_date(
        daily_artist, confirmed, established, bounded, scoring_date
    )
    if candidates.is_empty():
        return pl.DataFrame(schema=WATCHLIST_SCHEMA)

    ranked = score_recent_listening_share(listening, candidates)
    artists = pl.read_parquet(paths.curated_artists).select(
        "artist_id", "canonical_name"
    )
    if artists["artist_id"].n_unique() != artists.height or any(
        artists[column].null_count() for column in ("artist_id", "canonical_name")
    ):
        raise ValueError("Canonical artists must have unique, non-null identities")
    result = (
        ranked.head(3)
        .with_columns((-pl.col("score")).cast(pl.Int64).alias("rank"))
        .join(artists, on="artist_id", how="left", validate="m:1")
        .sort("rank")
    )
    if result["canonical_name"].null_count():
        raise ValueError("Every ranked artist must have a canonical name")
    columns = [
        pl.col(name).cast(dtype).alias(name) for name, dtype in WATCHLIST_SCHEMA.items()
    ]
    return result.select(columns)
