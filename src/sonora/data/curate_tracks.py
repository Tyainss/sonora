import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl

from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths
from sonora.data.track_resolution import TrackResolutionResult, resolve_track_identities

_TRACKS_SCHEMA = {
    "track_id": pl.String,
    "artist_id": pl.String,
    "canonical_name": pl.String,
}
_TRACK_ALIASES_SCHEMA = {
    "artist_id": pl.String,
    "observed_track_name": pl.String,
    "track_id": pl.String,
}


@dataclass(frozen=True, slots=True)
class CuratedTrackBuild:
    tracks: pl.DataFrame
    track_aliases: pl.DataFrame
    resolution: TrackResolutionResult


def build_curated_track_tables(
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
) -> CuratedTrackBuild:
    """Build canonical song-level tracks while preserving existing track IDs."""
    _require_inputs(paths)
    artist_aliases = pl.read_parquet(paths.curated_artist_aliases)
    existing_tracks, existing_aliases = _load_existing_tables(paths)
    resolved_events = _resolved_track_events(paths, artist_aliases=artist_aliases)
    resolution = resolve_track_identities(
        resolved_events,
        existing_aliases=existing_aliases,
    )
    tracks, track_aliases = materialize_track_tables(
        resolved_events,
        resolution,
        existing_tracks=existing_tracks,
    )
    _validate_track_tables(tracks, track_aliases)
    _write_track_tables(paths, tracks=tracks, track_aliases=track_aliases)
    return CuratedTrackBuild(
        tracks=tracks,
        track_aliases=track_aliases,
        resolution=resolution,
    )


def materialize_track_tables(
    events: pl.LazyFrame,
    resolution: TrackResolutionResult,
    *,
    existing_tracks: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign stable IDs to resolved track clusters and choose display names."""
    clusters = resolution.clusters
    existing_ids = (
        set(existing_tracks.get_column("track_id").to_list())
        if existing_tracks is not None
        else set()
    )
    referenced_ids = set(
        clusters.get_column("existing_track_id").drop_nulls().to_list()
    )
    unknown_ids = sorted(referenced_ids - existing_ids)
    if unknown_ids:
        raise ValueError(f"Existing track aliases reference unknown IDs: {unknown_ids}")

    cluster_to_track_id: dict[str, str] = {}
    for row in (
        clusters.select("cluster_key", "existing_track_id")
        .unique()
        .sort("cluster_key")
        .iter_rows(named=True)
    ):
        cluster_to_track_id[row["cluster_key"]] = row["existing_track_id"] or str(
            uuid4()
        )

    cluster_ids = pl.DataFrame(
        {
            "cluster_key": list(cluster_to_track_id),
            "track_id": list(cluster_to_track_id.values()),
        },
        schema={"cluster_key": pl.String, "track_id": pl.String},
    )
    track_aliases = (
        clusters.select("artist_id", "observed_track_name", "cluster_key")
        .join(cluster_ids, on="cluster_key", how="left", validate="m:1")
        .select("artist_id", "observed_track_name", "track_id")
        .sort("artist_id", "observed_track_name")
    )

    play_counts = (
        events.group_by("artist_id", "track_name")
        .agg(pl.len().alias("scrobble_count"))
        .collect()
        .join(
            clusters.select(
                "artist_id",
                pl.col("observed_track_name").alias("track_name"),
                "canonical_candidate_name",
                "cluster_key",
            ),
            on=["artist_id", "track_name"],
            how="left",
            validate="1:1",
        )
        .join(cluster_ids, on="cluster_key", how="left", validate="m:1")
    )
    if play_counts.get_column("track_id").null_count():
        raise ValueError("Every observed track name must resolve to a track ID")

    tracks = (
        play_counts.group_by("track_id", "artist_id", "canonical_candidate_name")
        .agg(pl.col("scrobble_count").sum())
        .sort(
            ["track_id", "scrobble_count", "canonical_candidate_name"],
            descending=[False, True, False],
        )
        .group_by("track_id", maintain_order=True)
        .agg(
            pl.col("artist_id").first(),
            pl.col("canonical_candidate_name").first().alias("canonical_name"),
        )
        .select("track_id", "artist_id", "canonical_name")
        .sort("artist_id", "canonical_name", "track_id")
    )
    return tracks, track_aliases


def _resolved_track_events(
    paths: DataPaths,
    *,
    artist_aliases: pl.DataFrame,
) -> pl.LazyFrame:
    events = pl.scan_parquet(paths.listening_events_clean).join(
        artist_aliases.lazy(),
        left_on="artist_name",
        right_on="observed_artist_name",
        how="left",
        validate="m:1",
    )
    missing_artist_ids = (
        events.select(pl.col("artist_id").null_count()).collect().item()
    )
    if missing_artist_ids:
        raise ValueError(
            f"{missing_artist_ids} listening events do not map to a canonical artist"
        )
    return events.select("artist_id", "track_name", "track_mbid")


def _require_inputs(paths: DataPaths) -> None:
    missing = [
        path
        for path in (
            paths.listening_events_clean,
            paths.curated_artists,
            paths.curated_artist_aliases,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Track curation requires clean events and canonical artist tables: "
            + ", ".join(str(path) for path in missing)
        )


def _load_existing_tables(
    paths: DataPaths,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    tracks_exists = paths.curated_tracks.is_file()
    aliases_exists = paths.curated_track_aliases.is_file()
    if tracks_exists != aliases_exists:
        raise FileNotFoundError(
            "Canonical track tables are incomplete; both tracks.parquet and "
            "track_aliases.parquet must exist together."
        )
    if not tracks_exists:
        return None, None

    tracks = pl.read_parquet(paths.curated_tracks)
    aliases = pl.read_parquet(paths.curated_track_aliases)
    _validate_track_tables(tracks, aliases)
    return tracks, aliases


def _validate_track_tables(
    tracks: pl.DataFrame,
    track_aliases: pl.DataFrame,
) -> None:
    if dict(tracks.schema) != _TRACKS_SCHEMA:
        raise ValueError(f"Unexpected tracks schema: {tracks.schema}")
    if dict(track_aliases.schema) != _TRACK_ALIASES_SCHEMA:
        raise ValueError(f"Unexpected track aliases schema: {track_aliases.schema}")

    if tracks.get_column("track_id").n_unique() != tracks.height:
        raise ValueError("Track IDs must be unique")
    duplicate_aliases = track_aliases.select(
        pl.struct("artist_id", "observed_track_name").is_duplicated().sum()
    ).item()
    if duplicate_aliases:
        raise ValueError("Observed track names must map once within each artist")

    required_values = {
        "tracks.track_id": tracks.get_column("track_id"),
        "tracks.artist_id": tracks.get_column("artist_id"),
        "tracks.canonical_name": tracks.get_column("canonical_name"),
        "track_aliases.artist_id": track_aliases.get_column("artist_id"),
        "track_aliases.observed_track_name": track_aliases.get_column(
            "observed_track_name"
        ),
        "track_aliases.track_id": track_aliases.get_column("track_id"),
    }
    missing_values = {
        name: series.null_count()
        for name, series in required_values.items()
        if series.null_count()
    }
    if missing_values:
        raise ValueError(f"Canonical track tables contain nulls: {missing_values}")

    valid_ids = set(tracks.get_column("track_id").to_list())
    for track_id in valid_ids:
        try:
            UUID(track_id)
        except ValueError as exc:
            raise ValueError(f"Invalid track UUID: {track_id}") from exc

    alias_ids = set(track_aliases.get_column("track_id").to_list())
    unknown_ids = sorted(alias_ids - valid_ids)
    if unknown_ids:
        raise ValueError(f"Track aliases reference unknown IDs: {unknown_ids}")
    unused_ids = sorted(valid_ids - alias_ids)
    if unused_ids:
        raise ValueError(f"Tracks without aliases are not allowed: {unused_ids}")

    track_artist_map = dict(tracks.select("track_id", "artist_id").iter_rows())
    mismatched_artists = [
        row
        for row in track_aliases.select("track_id", "artist_id").iter_rows(named=True)
        if track_artist_map[row["track_id"]] != row["artist_id"]
    ]
    if mismatched_artists:
        raise ValueError("Track aliases must reference tracks from the same artist")


def _write_track_tables(
    paths: DataPaths,
    *,
    tracks: pl.DataFrame,
    track_aliases: pl.DataFrame,
) -> None:
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    track_partial = paths.curated_tracks.with_suffix(
        f"{paths.curated_tracks.suffix}.partial"
    )
    aliases_partial = paths.curated_track_aliases.with_suffix(
        f"{paths.curated_track_aliases.suffix}.partial"
    )
    try:
        _write_partial(tracks, paths.curated_tracks)
        _write_partial(track_aliases, paths.curated_track_aliases)
        os.replace(track_partial, paths.curated_tracks)
        os.replace(aliases_partial, paths.curated_track_aliases)
    except BaseException:
        track_partial.unlink(missing_ok=True)
        aliases_partial.unlink(missing_ok=True)
        raise


def _write_partial(frame: pl.DataFrame, output_path: Path) -> Path:
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    if partial_path.exists():
        raise FileExistsError(f"Partial build already exists: {partial_path}")
    frame.write_parquet(partial_path)
    return partial_path
