"""Simple point-in-time behavioural rankings for breakout candidates."""

import polars as pl

BASELINE_KEYS = ("user_id", "artist_id", "scoring_date")
RECENT_SHARE_DAYS = 7
MOMENTUM_DAYS = 14


def score_recent_listening_share(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Rank candidates by prior 7-day listening share with behavioural tie-breaks."""
    scored = _build_listening_evidence(
        events,
        candidates,
        rolling_days=(RECENT_SHARE_DAYS,),
    ).with_columns(
        pl.when(pl.col("user_scrobbles_7d") > 0)
        .then(pl.col("artist_scrobbles_7d") / pl.col("user_scrobbles_7d"))
        .otherwise(0.0)
        .alias("listening_share_7d")
    )

    _validate_complete_order(
        scored,
        ["listening_share_7d", "latest_listened_at", "prior_scrobbles"],
        "7-day listening share, latest-listen recency, and prior listening "
        "do not fully order candidates",
    )

    return _rank_scores(
        scored,
        ["listening_share_7d", "latest_listened_at", "prior_scrobbles"],
        [
            "listening_share_7d",
            "artist_scrobbles_7d",
            "user_scrobbles_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
    )


def score_listening_share_momentum(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Rank candidates by week-over-week change in listening share."""
    scored = (
        _build_listening_evidence(
            events,
            candidates,
            rolling_days=(RECENT_SHARE_DAYS, MOMENTUM_DAYS),
        )
        .with_columns(
            (
                pl.col("artist_scrobbles_14d") - pl.col("artist_scrobbles_7d")
            ).alias("artist_scrobbles_previous_7d"),
            (
                pl.col("user_scrobbles_14d") - pl.col("user_scrobbles_7d")
            ).alias("user_scrobbles_previous_7d"),
        )
        .with_columns(
            pl.when(pl.col("user_scrobbles_7d") > 0)
            .then(pl.col("artist_scrobbles_7d") / pl.col("user_scrobbles_7d"))
            .otherwise(0.0)
            .alias("listening_share_7d"),
            pl.when(pl.col("user_scrobbles_previous_7d") > 0)
            .then(
                pl.col("artist_scrobbles_previous_7d")
                / pl.col("user_scrobbles_previous_7d")
            )
            .otherwise(0.0)
            .alias("listening_share_previous_7d"),
        )
        .with_columns(
            (
                pl.col("listening_share_7d")
                - pl.col("listening_share_previous_7d")
            ).alias("listening_share_momentum")
        )
    )

    _validate_complete_order(
        scored,
        [
            "listening_share_momentum",
            "listening_share_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
        (
            "Listening-share momentum and behavioural tie-breaks do not fully "
            "order candidates"
        ),
    )

    return _rank_scores(
        scored,
        [
            "listening_share_momentum",
            "listening_share_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
        [
            "listening_share_momentum",
            "listening_share_7d",
            "listening_share_previous_7d",
            "artist_scrobbles_7d",
            "artist_scrobbles_previous_7d",
            "user_scrobbles_7d",
            "user_scrobbles_previous_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
    )


def score_recent_active_days(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Rank candidates by active listening days in the prior 7 days."""
    scored = _build_listening_evidence(
        events,
        candidates,
        rolling_days=(RECENT_SHARE_DAYS,),
    ).with_columns(
        pl.when(pl.col("user_scrobbles_7d") > 0)
        .then(pl.col("artist_scrobbles_7d") / pl.col("user_scrobbles_7d"))
        .otherwise(0.0)
        .alias("listening_share_7d")
    )

    _validate_complete_order(
        scored,
        [
            "artist_active_days_7d",
            "listening_share_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
        (
            "Recent active days and behavioural tie-breaks do not fully "
            "order candidates"
        ),
    )

    return _rank_scores(
        scored,
        [
            "artist_active_days_7d",
            "listening_share_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
        [
            "artist_active_days_7d",
            "listening_share_7d",
            "artist_scrobbles_7d",
            "user_scrobbles_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
    )


def _build_listening_evidence(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
    *,
    rolling_days: tuple[int, ...],
) -> pl.DataFrame:
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
                daily.group_by("user_id", "date").agg(
                    pl.len().alias("user_scrobbles")
                ),
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
    artist_evidence = (
        artist_timeline.select(
            "user_id",
            "artist_id",
            "date",
            "latest_prior_listened_at",
            "prior_scrobbles",
        )
        .join(candidate_days, on=["user_id", "artist_id", "date"], how="semi")
    )
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
        candidates.select(BASELINE_KEYS)
        .join(user_evidence, on=["user_id", "scoring_date"], validate="m:1")
        .join(artist_evidence, on=BASELINE_KEYS, validate="1:1")
    )
    if scored.height != candidates.height:
        raise ValueError("Every candidate must receive baseline evidence")
    if scored["latest_listened_at"].null_count():
        raise ValueError("Every candidate must have prior listening history")
    return scored


def _validate_complete_order(
    scored: pl.DataFrame,
    order_columns: list[str],
    message: str,
) -> None:
    unresolved = (
        scored.group_by("user_id", "scoring_date", *order_columns)
        .len()
        .filter(pl.col("len") > 1)
    )
    if unresolved.height:
        raise ValueError(message)


def _rank_scores(
    scored: pl.DataFrame,
    order_columns: list[str],
    output_columns: list[str],
) -> pl.DataFrame:
    return (
        scored.sort(
            ["user_id", "scoring_date", *order_columns],
            descending=[False, False, *([True] * len(order_columns))],
        )
        .with_columns(
            (
                -pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over(
                    "user_id", "scoring_date"
                )
            )
            .cast(pl.Float64)
            .alias("score")
        )
        .select(*BASELINE_KEYS, *output_columns, "score")
    )


def _validate_candidates(candidates: pl.DataFrame) -> None:
    missing = set(BASELINE_KEYS) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if candidates.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if any(candidates[column].null_count() for column in BASELINE_KEYS):
        raise ValueError("Candidate keys cannot contain nulls")
    if candidates.select(BASELINE_KEYS).n_unique() != candidates.height:
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
