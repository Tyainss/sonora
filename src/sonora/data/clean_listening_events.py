import os
from pathlib import Path

import polars as pl

from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_REQUIRED_RAW_COLUMNS = {
    "artist_name",
    "track_name",
    "album_name",
    "timestamp_unix",
    "mbid",
    "raw",
}
_REQUIRED_CLEAN_COLUMNS = ("artist_name", "track_name", "listened_at")


def clean_listening_events(
    scrobbles: pl.LazyFrame,
    *,
    user_id: str,
) -> pl.LazyFrame:
    """Apply safe structural cleaning to raw Last.fm scrobbles."""
    _validate_user_id(user_id)
    _validate_raw_schema(scrobbles)

    artist_mbid = (
        pl.col("raw")
        .struct.field("artist")
        .struct.field("mbid")
        .str.strip_chars()
        .replace("", None)
        .alias("artist_mbid")
    )
    album_mbid = (
        pl.col("raw")
        .struct.field("album")
        .struct.field("mbid")
        .str.strip_chars()
        .replace("", None)
        .alias("album_mbid")
    )

    return scrobbles.select(
        pl.lit(user_id).alias("user_id"),
        pl.from_epoch("timestamp_unix", time_unit="s")
        .dt.replace_time_zone("UTC")
        .alias("listened_at"),
        _clean_string("artist_name").alias("artist_name"),
        _clean_string("track_name").alias("track_name"),
        _clean_string("album_name").alias("album_name"),
        artist_mbid,
        _clean_string("mbid").alias("track_mbid"),
        album_mbid,
        pl.lit("lastfm").alias("source"),
    ).unique(maintain_order=True)


def build_listening_events_clean(
    *,
    user_id: str,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> Path:
    """Build the clean interim listening-events dataset from the raw snapshot."""
    input_path = paths.raw_lastfm_scrobbles
    output_path = paths.listening_events_clean

    if not input_path.is_file():
        raise FileNotFoundError(f"Raw Last.fm snapshot not found: {input_path}")

    cleaned = clean_listening_events(
        pl.scan_ndjson(input_path),
        user_id=user_id,
    ).collect()
    _validate_clean_events(cleaned)
    _write_parquet_atomic(cleaned, output_path)
    return output_path


def _clean_string(column: str) -> pl.Expr:
    return pl.col(column).str.strip_chars().replace("", None)


def _validate_user_id(user_id: str) -> None:
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty Sonora identifier")


def _validate_raw_schema(scrobbles: pl.LazyFrame) -> None:
    columns = set(scrobbles.collect_schema().names())
    missing = sorted(_REQUIRED_RAW_COLUMNS - columns)
    if missing:
        raise ValueError(f"Raw Last.fm snapshot is missing required columns: {missing}")


def _validate_clean_events(events: pl.DataFrame) -> None:
    invalid = events.select(
        pl.col(column).is_null().sum().alias(column)
        for column in _REQUIRED_CLEAN_COLUMNS
    ).row(0, named=True)
    invalid = {column: count for column, count in invalid.items() if count}

    if invalid:
        details = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise ValueError(
            f"Clean listening events contain missing required values: {details}"
        )


def _write_parquet_atomic(events: pl.DataFrame, output_path: Path) -> None:
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        raise FileExistsError(
            "A partial clean listening-events build already exists; "
            "inspect or remove it before retrying."
        )

    try:
        events.write_parquet(partial_path)
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
