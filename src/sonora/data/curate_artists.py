import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl

from sonora.data.artist_candidates import (
    DEFAULT_ARTIST_CANDIDATE_CONFIG,
    ArtistCandidateConfig,
)
from sonora.data.artist_resolution import (
    DEFAULT_ARTIST_RESOLUTION_CONFIG,
    ArtistResolutionConfig,
    ArtistResolutionResult,
    resolve_artist_identities,
)
from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths

_ARTISTS_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
}
_ARTIST_ALIASES_SCHEMA = {
    "observed_artist_name": pl.String,
    "artist_id": pl.String,
}


@dataclass(frozen=True, slots=True)
class CuratedArtistBuild:
    artists: pl.DataFrame
    artist_aliases: pl.DataFrame
    resolution: ArtistResolutionResult


def build_curated_artist_tables(
    *,
    paths: DataPaths = DEFAULT_DATA_PATHS,
    candidate_config: ArtistCandidateConfig = DEFAULT_ARTIST_CANDIDATE_CONFIG,
    resolution_config: ArtistResolutionConfig = DEFAULT_ARTIST_RESOLUTION_CONFIG,
) -> CuratedArtistBuild:
    """Build canonical artist and alias tables while preserving existing artist IDs."""
    if not paths.listening_events_clean.is_file():
        raise FileNotFoundError(
            f"Clean listening events not found: {paths.listening_events_clean}"
        )

    events = pl.scan_parquet(paths.listening_events_clean)
    existing_artists, existing_aliases = _load_existing_tables(paths)
    resolution = resolve_artist_identities(
        events,
        candidate_config=candidate_config,
        resolution_config=resolution_config,
        existing_aliases=existing_aliases,
    )
    artists, artist_aliases = materialize_artist_tables(
        events,
        resolution,
        existing_artists=existing_artists,
    )
    _validate_artist_tables(artists, artist_aliases)
    _write_artist_tables(paths, artists=artists, artist_aliases=artist_aliases)

    return CuratedArtistBuild(
        artists=artists,
        artist_aliases=artist_aliases,
        resolution=resolution,
    )


def materialize_artist_tables(
    events: pl.LazyFrame,
    resolution: ArtistResolutionResult,
    *,
    existing_artists: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign stable IDs to resolved clusters and choose canonical artist names."""
    cluster_rows = resolution.clusters
    existing_ids = (
        set(existing_artists.get_column("artist_id").to_list())
        if existing_artists is not None
        else set()
    )
    _validate_existing_artist_ids(cluster_rows, existing_ids=existing_ids)

    cluster_to_artist_id: dict[str, str] = {}
    for row in (
        cluster_rows.select("cluster_key", "existing_artist_id")
        .unique()
        .sort("cluster_key")
        .iter_rows(named=True)
    ):
        cluster_to_artist_id[row["cluster_key"]] = row["existing_artist_id"] or str(
            uuid4()
        )

    cluster_ids = pl.DataFrame(
        {
            "cluster_key": list(cluster_to_artist_id),
            "artist_id": list(cluster_to_artist_id.values()),
        },
        schema={"cluster_key": pl.String, "artist_id": pl.String},
    )
    artist_aliases = (
        cluster_rows.select("observed_artist_name", "cluster_key")
        .join(cluster_ids, on="cluster_key", how="left", validate="m:1")
        .select("observed_artist_name", "artist_id")
        .sort("observed_artist_name")
    )

    play_counts = (
        events.group_by("artist_name")
        .agg(pl.len().alias("scrobble_count"))
        .collect()
        .join(
            artist_aliases,
            left_on="artist_name",
            right_on="observed_artist_name",
            how="left",
            validate="1:1",
        )
    )
    if play_counts.get_column("artist_id").null_count():
        raise ValueError("Every observed artist name must resolve to an artist ID")

    artists = (
        play_counts.sort(
            ["artist_id", "scrobble_count", "artist_name"],
            descending=[False, True, False],
        )
        .group_by("artist_id", maintain_order=True)
        .agg(pl.col("artist_name").first().alias("canonical_name"))
        .select("artist_id", "canonical_name")
        .sort("canonical_name", "artist_id")
    )
    return artists, artist_aliases


def _load_existing_tables(
    paths: DataPaths,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    artists_exists = paths.curated_artists.is_file()
    aliases_exists = paths.curated_artist_aliases.is_file()

    if artists_exists != aliases_exists:
        raise FileNotFoundError(
            "Canonical artist tables are incomplete; both artists.parquet and "
            "artist_aliases.parquet must exist together."
        )
    if not artists_exists:
        return None, None

    artists = pl.read_parquet(paths.curated_artists)
    aliases = pl.read_parquet(paths.curated_artist_aliases)
    _validate_artist_tables(artists, aliases)
    return artists, aliases


def _validate_existing_artist_ids(
    clusters: pl.DataFrame,
    *,
    existing_ids: set[str],
) -> None:
    referenced_ids = set(
        clusters.get_column("existing_artist_id").drop_nulls().to_list()
    )
    unknown_ids = sorted(referenced_ids - existing_ids)
    if unknown_ids:
        raise ValueError(
            f"Existing artist aliases reference unknown IDs: {unknown_ids}"
        )


def _validate_artist_tables(
    artists: pl.DataFrame,
    artist_aliases: pl.DataFrame,
) -> None:
    if dict(artists.schema) != _ARTISTS_SCHEMA:
        raise ValueError(f"Unexpected artists schema: {artists.schema}")
    if dict(artist_aliases.schema) != _ARTIST_ALIASES_SCHEMA:
        raise ValueError(f"Unexpected artist aliases schema: {artist_aliases.schema}")

    if artists.get_column("artist_id").n_unique() != artists.height:
        raise ValueError("Artist IDs must be unique")
    if (
        artist_aliases.get_column("observed_artist_name").n_unique()
        != artist_aliases.height
    ):
        raise ValueError("Observed artist names must map once")

    required_values = {
        "artists.artist_id": artists.get_column("artist_id"),
        "artists.canonical_name": artists.get_column("canonical_name"),
        "artist_aliases.observed_artist_name": artist_aliases.get_column(
            "observed_artist_name"
        ),
        "artist_aliases.artist_id": artist_aliases.get_column("artist_id"),
    }
    missing_values = {
        name: series.null_count()
        for name, series in required_values.items()
        if series.null_count()
    }
    if missing_values:
        raise ValueError(f"Canonical artist tables contain nulls: {missing_values}")

    valid_ids = set(artists.get_column("artist_id").to_list())
    for artist_id in valid_ids:
        try:
            UUID(artist_id)
        except ValueError as exc:
            raise ValueError(f"Invalid artist UUID: {artist_id}") from exc

    alias_ids = set(artist_aliases.get_column("artist_id").to_list())
    unknown_ids = sorted(alias_ids - valid_ids)
    if unknown_ids:
        raise ValueError(f"Artist aliases reference unknown IDs: {unknown_ids}")

    unused_ids = sorted(valid_ids - alias_ids)
    if unused_ids:
        raise ValueError(f"Artists without aliases are not allowed: {unused_ids}")


def _write_artist_tables(
    paths: DataPaths,
    *,
    artists: pl.DataFrame,
    artist_aliases: pl.DataFrame,
) -> None:
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    artist_partial = paths.curated_artists.with_suffix(
        f"{paths.curated_artists.suffix}.partial"
    )
    aliases_partial = paths.curated_artist_aliases.with_suffix(
        f"{paths.curated_artist_aliases.suffix}.partial"
    )

    try:
        _write_partial(artists, paths.curated_artists)
        _write_partial(artist_aliases, paths.curated_artist_aliases)
        os.replace(artist_partial, paths.curated_artists)
        os.replace(aliases_partial, paths.curated_artist_aliases)
    except BaseException:
        artist_partial.unlink(missing_ok=True)
        aliases_partial.unlink(missing_ok=True)
        raise


def _write_partial(frame: pl.DataFrame, output_path: Path) -> Path:
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    if partial_path.exists():
        raise FileExistsError(f"Partial build already exists: {partial_path}")
    frame.write_parquet(partial_path)
    return partial_path
