import os
from pathlib import Path

import polars as pl

from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_CURATED_COLUMNS = [
    "user_id",
    "listened_at",
    "artist_id",
    "track_id",
    "album_name",
    "source",
]
_REQUIRED_VALUES = ("user_id", "listened_at", "artist_id", "track_id", "source")


def build_curated_listening_events(
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> Path:
    """Build the canonical listening-event fact table."""
    _require_inputs(paths)

    clean_events = pl.read_parquet(paths.listening_events_clean).with_row_index(
        "_event_row_id"
    )
    artists = pl.read_parquet(paths.curated_artists)
    artist_aliases = pl.read_parquet(paths.curated_artist_aliases)
    tracks = pl.read_parquet(paths.curated_tracks)
    track_aliases = pl.read_parquet(paths.curated_track_aliases)

    expected_rows = clean_events.height
    resolved = clean_events.join(
        artist_aliases,
        left_on="artist_name",
        right_on="observed_artist_name",
        how="left",
        validate="m:1",
    )
    _validate_event_rows(resolved, expected_rows=expected_rows)
    _require_resolved_ids(resolved, "artist_id", label="artist")

    resolved = resolved.join(
        track_aliases,
        left_on=["artist_id", "track_name"],
        right_on=["artist_id", "observed_track_name"],
        how="left",
        validate="m:1",
    )
    _validate_event_rows(resolved, expected_rows=expected_rows)
    _require_resolved_ids(resolved, "track_id", label="track")

    resolved = resolved.join(
        artists.select("artist_id").with_columns(pl.lit(True).alias("_artist_known")),
        on="artist_id",
        how="left",
        validate="m:1",
    ).join(
        tracks.select(
            "track_id",
            pl.col("artist_id").alias("_track_artist_id"),
        ),
        on="track_id",
        how="left",
        validate="m:1",
    )
    _validate_event_rows(resolved, expected_rows=expected_rows)
    _validate_references(resolved)

    curated = resolved.sort("_event_row_id").select(_CURATED_COLUMNS)
    _validate_curated_events(curated, expected_rows=expected_rows)
    _write_parquet_atomic(curated, paths.curated_listening_events)
    return paths.curated_listening_events


def _require_inputs(paths: DataPaths) -> None:
    required = (
        paths.listening_events_clean,
        paths.curated_artists,
        paths.curated_artist_aliases,
        paths.curated_tracks,
        paths.curated_track_aliases,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Curated listening events require clean events plus artist and track "
            "tables: " + ", ".join(str(path) for path in missing)
        )


def _validate_event_rows(events: pl.DataFrame, *, expected_rows: int) -> None:
    if events.height != expected_rows:
        raise ValueError(
            "Canonical joins changed the listening-event row count: "
            f"expected {expected_rows}, got {events.height}"
        )
    if events.get_column("_event_row_id").n_unique() != expected_rows:
        raise ValueError("Canonical joins duplicated or lost listening events")


def _require_resolved_ids(events: pl.DataFrame, column: str, *, label: str) -> None:
    missing = events.get_column(column).null_count()
    if missing:
        raise ValueError(
            f"{missing} listening events do not map to a canonical {label}"
        )


def _validate_references(events: pl.DataFrame) -> None:
    unknown_artists = events.get_column("_artist_known").null_count()
    if unknown_artists:
        raise ValueError(
            f"{unknown_artists} listening events reference unknown artist IDs"
        )

    unknown_tracks = events.get_column("_track_artist_id").null_count()
    if unknown_tracks:
        raise ValueError(
            f"{unknown_tracks} listening events reference unknown track IDs"
        )

    mismatched_tracks = events.filter(
        pl.col("artist_id") != pl.col("_track_artist_id")
    ).height
    if mismatched_tracks:
        raise ValueError(
            f"{mismatched_tracks} listening events reference a track from a "
            "different artist"
        )


def _validate_curated_events(events: pl.DataFrame, *, expected_rows: int) -> None:
    if events.columns != _CURATED_COLUMNS:
        raise ValueError(
            f"Unexpected curated listening-events columns: {events.columns}"
        )
    if events.height != expected_rows:
        raise ValueError(
            "Curated listening events must preserve the clean event count: "
            f"expected {expected_rows}, got {events.height}"
        )

    missing_values = {
        column: events.get_column(column).null_count()
        for column in _REQUIRED_VALUES
        if events.get_column(column).null_count()
    }
    if missing_values:
        raise ValueError(
            "Curated listening events contain missing required values: "
            f"{missing_values}"
        )

    listened_at = events.schema["listened_at"]
    if "UTC" not in str(listened_at):
        raise ValueError("Curated listened_at timestamps must remain in UTC")


def _write_parquet_atomic(events: pl.DataFrame, output_path: Path) -> None:
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        raise FileExistsError(
            "A partial curated listening-events build already exists; "
            "inspect or remove it before retrying."
        )

    try:
        events.write_parquet(partial_path)
        os.replace(partial_path, output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
