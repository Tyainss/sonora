"""Write the settled daily targets and their source details."""

import hashlib
import json
import os
from pathlib import Path

import polars as pl

from sonora.breakout.events import detect_first_breakouts
from sonora.breakout.history import (
    aggregate_daily_listening,
    available_user_bounds,
    validate_user_bounds,
)
from sonora.breakout.targets import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RECENCY_DAYS,
    make_daily_targets,
)
from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths


def build_breakout_targets(
    *, user_bounds: pl.DataFrame, paths: DataPaths = DEFAULT_DATA_PATHS
) -> Path:
    """Save events, daily rows, and reproducible settings using explicit user bounds.

    Recent unknown labels remain null. Training must select has_full_horizon.
    All outputs are checked before replacement; metadata is replaced last.
    """
    validate_user_bounds(user_bounds)
    destinations = [
        paths.breakout_events,
        paths.daily_breakout_targets,
        paths.breakout_build_metadata,
    ]
    partials = [p.with_suffix(p.suffix + ".partial") for p in destinations]
    if any(p.exists() for p in partials):
        raise FileExistsError(
            "An unfinished breakout output exists; inspect it before rebuilding"
        )
    with paths.curated_listening_events.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    daily_user, daily_artist = aggregate_daily_listening(
        pl.scan_parquet(paths.curated_listening_events)
    )
    if set(daily_user["user_id"]) != set(user_bounds["user_id"]):
        raise ValueError("Bounds must cover exactly the users in the listening data")
    observed = daily_user.group_by("user_id").agg(pl.col("date").min().alias("_first"))
    if (
        user_bounds.join(observed, on="user_id")
        .filter(pl.col("first_date") != pl.col("_first"))
        .height
    ):
        raise ValueError("Bounds must preserve the first observed listening day")
    events, established = detect_first_breakouts(daily_user, daily_artist, user_bounds)
    targets = make_daily_targets(daily_artist, events, established, user_bounds)
    paths.breakout_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Exclusive creation prevents accidentally replacing an unfinished build.
        for partial in partials:
            partial.touch(exist_ok=False)
            created.append(partial)
        events.write_parquet(partials[0])
        targets.sink_parquet(partials[1])
        saved = pl.read_parquet(partials[1])
        if (
            saved.select("user_id", "artist_id", "scoring_date").unique().height
            != saved.height
        ):
            raise ValueError("Duplicate daily target rows")
        if saved.filter(
            (pl.col("last_listen_date") >= pl.col("scoring_date"))
            | (pl.col("first_listen_date") > pl.col("last_listen_date"))
            | (
                (pl.col("scoring_date") - pl.col("last_listen_date")).dt.total_days()
                > DEFAULT_RECENCY_DAYS
            )
            | (pl.col("has_full_horizon") & pl.col("target_60d").is_null())
            | (~pl.col("has_full_horizon") & pl.col("target_60d").eq(False))
        ).height:
            raise ValueError("Invalid dates or incomplete-history labels")
        ready = set(available_user_bounds(user_bounds)["user_id"])
        users = []
        for bounds in user_bounds.sort("user_id").to_dicts():
            user = bounds["user_id"]
            rows = saved.filter(pl.col("user_id") == user)
            users.append(
                {
                    **bounds,
                    "status": "built" if user in ready else "not_enough_history",
                    "rows": rows.height,
                    "full_horizon_rows": rows["has_full_horizon"].sum(),
                    "positive_rows": rows["target_60d"].sum(),
                    "unknown_rows": rows["target_60d"].null_count(),
                    "events": events.filter(pl.col("user_id") == user).height,
                    "established_artists": established.filter(
                        pl.col("user_id") == user
                    ).height,
                }
            )
        metadata = {
            "schema_version": 1,
            "source": {
                "path": str(paths.curated_listening_events),
                "sha256": source_hash,
            },
            "settings": {
                "recency_days": DEFAULT_RECENCY_DAYS,
                "horizon_days": DEFAULT_HORIZON_DAYS,
                "period_days": 7,
                "min_active_days": 2,
                "importance_quantile": 0.9,
                "importance_lookback_days": 730,
                "share_decimal_places": 2,
                "second_period_start_gap_days": [7, 14],
                "timezone": "UTC",
                "target_interval": "scoring_date < confirmation_date <= scoring_date + 60 days",
                "training_filter": "has_full_horizon = true",
            },
            "users": users,
            "outputs": {
                p.name: hashlib.sha256(tmp.read_bytes()).hexdigest()
                for p, tmp in zip(destinations[:2], partials[:2], strict=True)
            },
        }
        partials[2].write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        with paths.curated_listening_events.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != source_hash:
                raise ValueError("Listening data changed during the build")
        for partial, destination in zip(partials, destinations, strict=True):
            os.replace(partial, destination)
    finally:
        for partial in created:
            partial.unlink(missing_ok=True)
    return paths.daily_breakout_targets
