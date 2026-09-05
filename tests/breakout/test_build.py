import datetime as dt
import hashlib
import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from sonora.breakout import aggregate_daily_listening, historical_user_bounds
from sonora.breakout.build import build_breakout_targets
from sonora.data.paths import DataPaths


def source_at(paths):
    base = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    rows = [("a", "artist", base + dt.timedelta(days=n)) for n in range(500)]
    rows += [("short", "artist", base)]
    source = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "listened_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    paths.curated_dir.mkdir(parents=True)
    source.write_parquet(paths.curated_listening_events)
    daily, _ = aggregate_daily_listening(source.lazy())
    return historical_user_bounds(daily)


def test_build_roundtrip_metadata_and_unfinished_output(tmp_path):
    paths = DataPaths(tmp_path)
    bounds = source_at(paths)
    assert (
        build_breakout_targets(user_bounds=bounds, paths=paths)
        == paths.daily_breakout_targets
    )
    rows = pl.read_parquet(paths.daily_breakout_targets)
    assert rows.schema["target_60d"] == pl.Boolean
    metadata = json.loads(paths.breakout_build_metadata.read_text())
    assert metadata["settings"]["recency_days"] == 90
    assert metadata["settings"]["horizon_days"] == 60
    assert metadata["users"][1]["status"] == "not_enough_history"
    assert (
        metadata["outputs"][paths.daily_breakout_targets.name]
        == hashlib.sha256(paths.daily_breakout_targets.read_bytes()).hexdigest()
    )
    build_breakout_targets(user_bounds=bounds, paths=paths)
    assert_frame_equal(rows, pl.read_parquet(paths.daily_breakout_targets))
    partial = paths.daily_breakout_targets.with_suffix(".parquet.partial")
    partial.write_text("unfinished")
    with pytest.raises(FileExistsError):
        build_breakout_targets(user_bounds=bounds, paths=paths)
    assert partial.read_text() == "unfinished"
    assert_frame_equal(rows, pl.read_parquet(paths.daily_breakout_targets))


def test_build_requires_all_users_and_conservative_end(tmp_path):
    paths = DataPaths(tmp_path)
    bounds = source_at(paths)
    with pytest.raises(ValueError, match="exactly"):
        build_breakout_targets(user_bounds=bounds.head(1), paths=paths)
    with pytest.raises(ValueError, match="latest listen"):
        build_breakout_targets(
            user_bounds=bounds.with_columns(
                pl.col("last_observed_date").alias("last_complete_date")
            ),
            paths=paths,
        )


def test_daily_aggregation_requires_utc():
    source = pl.DataFrame(
        {
            "user_id": ["a"],
            "artist_id": ["x"],
            "listened_at": [dt.datetime(2020, 1, 1, tzinfo=dt.UTC)],
        }
    )
    source = source.with_columns(pl.col("listened_at").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="UTC"):
        aggregate_daily_listening(source.lazy())
