"""Point-in-time listening features for eligible breakout candidates."""

import polars as pl

CANDIDATE_KEYS = ("user_id", "artist_id", "scoring_date")
RECENT_SHARE_DAYS = 7


def build_breakout_features(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Return the two raw model features for each supplied eligible artist-day.

    listening_share_7d uses D-7 through D-1. previous_listening_count counts
    artist listens strictly before D-7. Eligibility belongs to the caller;
    target labels and feature transformations are not part of this output.
    """
    return (
        build_listening_evidence(events, candidates)
        .with_columns(
            (pl.col("prior_scrobbles") - pl.col("artist_scrobbles_7d")).alias(
                "previous_listening_count"
            )
        )
        .select(*CANDIDATE_KEYS, "listening_share_7d", "previous_listening_count")
    )


def build_listening_evidence(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
    *,
    rolling_days: tuple[int, ...] = (RECENT_SHARE_DAYS,),
) -> pl.DataFrame:
    """Build listening counts, recency and shares for supplied candidate rows.

    A window of N days includes D-N through D-1 in UTC. Shares use all of
    the user's listening, including artists outside the candidate set.
    Candidates must be unique and have listening history before D.
    """
    _validate_candidates(candidates)
    event_frame = events.collect() if isinstance(events, pl.LazyFrame) else events
    _validate_events(event_frame)
    timestamp_dtype = event_frame.schema["listened_at"]

    daily = event_frame.select("user_id", "artist_id", "listened_at").with_columns(
        pl.col("listened_at").dt.date().alias("date")
    )
    candidate_days = candidates.select(
        "user_id",
        "artist_id",
        pl.col("scoring_date").alias("date"),
    )
    candidate_user_days = candidate_days.select("user_id", "date").unique()

    user_timeline = (
        pl.concat(
            [
                daily.group_by("user_id", "date").agg(pl.len().alias("user_scrobbles")),
                candidate_user_days.with_columns(
                    pl.lit(0, dtype=pl.UInt32).alias("user_scrobbles")
                ),
            ]
        )
        .group_by("user_id", "date")
        .agg(pl.col("user_scrobbles").sum())
        .sort("user_id", "date")
    )
    user_evidence = candidate_user_days
    for days in rolling_days:
        recent = (
            user_timeline.rolling(
                index_column="date",
                period=f"{days}d",
                closed="left",
                group_by="user_id",
            )
            .agg(pl.col("user_scrobbles").sum().alias(f"user_scrobbles_{days}d"))
            .join(candidate_user_days, on=["user_id", "date"], how="semi")
        )
        user_evidence = user_evidence.join(
            recent,
            on=["user_id", "date"],
            validate="1:1",
        )
    user_evidence = user_evidence.rename({"date": "scoring_date"})

    artist_timeline = (
        pl.concat(
            [
                daily.group_by("user_id", "artist_id", "date").agg(
                    pl.len().alias("artist_scrobbles"),
                    pl.col("listened_at").max().alias("latest_listened_at"),
                ),
                candidate_days.with_columns(
                    pl.lit(0, dtype=pl.UInt32).alias("artist_scrobbles"),
                    pl.lit(None, dtype=timestamp_dtype).alias("latest_listened_at"),
                ),
            ]
        )
        .group_by("user_id", "artist_id", "date")
        .agg(
            pl.col("artist_scrobbles").sum(),
            pl.col("latest_listened_at").max(),
        )
        .sort("user_id", "artist_id", "date")
        .with_columns(
            pl.col("latest_listened_at")
            .forward_fill()
            .shift(1)
            .over(["user_id", "artist_id"], order_by="date")
            .alias("latest_prior_listened_at"),
            pl.col("artist_scrobbles")
            .cum_sum()
            .shift(1)
            .over(["user_id", "artist_id"], order_by="date")
            .fill_null(0)
            .alias("prior_scrobbles"),
        )
    )
    artist_evidence = artist_timeline.select(
        "user_id",
        "artist_id",
        "date",
        "latest_prior_listened_at",
        "prior_scrobbles",
    ).join(candidate_days, on=["user_id", "artist_id", "date"], how="semi")
    for days in rolling_days:
        recent = (
            artist_timeline.rolling(
                index_column="date",
                period=f"{days}d",
                closed="left",
                group_by=["user_id", "artist_id"],
            )
            .agg(
                pl.col("artist_scrobbles").sum().alias(f"artist_scrobbles_{days}d"),
                (pl.col("artist_scrobbles") > 0)
                .sum()
                .alias(f"artist_active_days_{days}d"),
            )
            .join(candidate_days, on=["user_id", "artist_id", "date"], how="semi")
        )
        artist_evidence = artist_evidence.join(
            recent,
            on=["user_id", "artist_id", "date"],
            validate="1:1",
        )
    artist_evidence = artist_evidence.rename(
        {
            "date": "scoring_date",
            "latest_prior_listened_at": "latest_listened_at",
        }
    )

    scored = (
        candidates.select(CANDIDATE_KEYS)
        .join(user_evidence, on=["user_id", "scoring_date"], validate="m:1")
        .join(artist_evidence, on=CANDIDATE_KEYS, validate="1:1")
    )
    if scored.height != candidates.height:
        raise ValueError("Every candidate must receive listening evidence")
    if scored["latest_listened_at"].null_count():
        raise ValueError("Every candidate must have prior listening history")
    return scored.with_columns(
        pl.when(pl.col(f"user_scrobbles_{days}d") > 0)
        .then(pl.col(f"artist_scrobbles_{days}d") / pl.col(f"user_scrobbles_{days}d"))
        .otherwise(0.0)
        .alias(f"listening_share_{days}d")
        for days in rolling_days
    )


def _validate_candidates(candidates: pl.DataFrame) -> None:
    missing = set(CANDIDATE_KEYS) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if candidates.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if any(candidates[column].null_count() for column in CANDIDATE_KEYS):
        raise ValueError("Candidate keys cannot contain nulls")
    if candidates.select(CANDIDATE_KEYS).n_unique() != candidates.height:
        raise ValueError("Candidates contain duplicate rows")


def _validate_events(events: pl.DataFrame) -> None:
    required = {"user_id", "artist_id", "listened_at"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing listening-event columns: {sorted(missing)}")
    timestamp = events.schema["listened_at"]
    if not isinstance(timestamp, pl.Datetime) or timestamp.time_zone != "UTC":
        raise ValueError("listened_at must contain UTC timestamps")
    if any(events[column].null_count() for column in required):
        raise ValueError("Listening-event identity and timestamps cannot contain nulls")
