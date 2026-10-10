"""Simple point-in-time behavioural rankings for breakout candidates."""

import polars as pl

from sonora.breakout.features import (
    CANDIDATE_KEYS,
    RECENT_SHARE_DAYS,
    build_listening_evidence,
)

MOMENTUM_DAYS = 14


def score_recent_listening_share(
    events: pl.DataFrame | pl.LazyFrame,
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Rank candidates by prior 7-day listening share with behavioural tie-breaks."""
    scored = build_listening_evidence(
        events,
        candidates,
        rolling_days=(RECENT_SHARE_DAYS,),
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
        build_listening_evidence(
            events,
            candidates,
            rolling_days=(RECENT_SHARE_DAYS, MOMENTUM_DAYS),
        )
        .with_columns(
            (pl.col("artist_scrobbles_14d") - pl.col("artist_scrobbles_7d")).alias(
                "artist_scrobbles_previous_7d"
            ),
            (pl.col("user_scrobbles_14d") - pl.col("user_scrobbles_7d")).alias(
                "user_scrobbles_previous_7d"
            ),
        )
        .with_columns(
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
                pl.col("listening_share_7d") - pl.col("listening_share_previous_7d")
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
    scored = build_listening_evidence(
        events,
        candidates,
        rolling_days=(RECENT_SHARE_DAYS,),
    )

    _validate_complete_order(
        scored,
        [
            "artist_active_days_7d",
            "listening_share_7d",
            "latest_listened_at",
            "prior_scrobbles",
        ],
        ("Recent active days and behavioural tie-breaks do not fully order candidates"),
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
        .select(*CANDIDATE_KEYS, *output_columns, "score")
    )
