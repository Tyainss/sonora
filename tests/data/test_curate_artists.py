from uuid import UUID

import polars as pl

from sonora.data.artist_resolution import ArtistResolutionResult
from sonora.data.curate_artists import (
    build_curated_artist_tables,
    materialize_artist_tables,
)
from sonora.data.paths import DataPaths


def _events(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "artist_name": [artist for artist, _ in rows],
            "track_name": [track for _, track in rows],
            "album_name": [None] * len(rows),
            "artist_mbid": [None] * len(rows),
            "track_mbid": [None] * len(rows),
        }
    )


def test_materialize_artist_tables_uses_most_played_alias_as_canonical_name():
    events = _events(
        [
            ("Samuel Úria", "A"),
            ("Samuel Úria", "B"),
            ("Samuel Uria", "C"),
        ]
    ).lazy()
    resolution = _resolution(
        [
            ("Samuel Uria", "Samuel Uria", None),
            ("Samuel Úria", "Samuel Uria", None),
        ]
    )

    artists, aliases = materialize_artist_tables(events, resolution)

    assert artists.height == 1
    assert artists.row(0, named=True)["canonical_name"] == "Samuel Úria"
    assert aliases.get_column("artist_id").n_unique() == 1
    UUID(artists.row(0, named=True)["artist_id"])


def test_canonical_name_tie_break_is_deterministic():
    events = _events(
        [
            ("Samuel Úria", "A"),
            ("Samuel Uria", "B"),
        ]
    ).lazy()
    resolution = _resolution(
        [
            ("Samuel Uria", "Samuel Uria", None),
            ("Samuel Úria", "Samuel Uria", None),
        ]
    )

    artists, _ = materialize_artist_tables(events, resolution)

    assert artists.row(0, named=True)["canonical_name"] == "Samuel Uria"


def test_build_curated_artist_tables_preserves_ids_across_reruns(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    paths.interim_dir.mkdir(parents=True)
    _events(
        [
            ("Samuel Uria", "A"),
            ("Samuel Úria", "A"),
            ("Björk", "B"),
        ]
    ).write_parquet(paths.listening_events_clean)

    first = build_curated_artist_tables(paths=paths)
    first_ids = dict(
        first.artist_aliases.select("observed_artist_name", "artist_id").iter_rows()
    )
    second = build_curated_artist_tables(paths=paths)
    second_ids = dict(
        second.artist_aliases.select("observed_artist_name", "artist_id").iter_rows()
    )

    assert second_ids == first_ids
    assert second.resolution.evidence.is_empty()
    assert paths.curated_artists.is_file()
    assert paths.curated_artist_aliases.is_file()


def test_new_alias_can_join_existing_artist_without_changing_id(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    paths.interim_dir.mkdir(parents=True)
    _events([("Samuel Úria", "A")]).write_parquet(paths.listening_events_clean)
    first = build_curated_artist_tables(paths=paths)
    artist_id = first.artist_aliases.row(0, named=True)["artist_id"]

    _events(
        [
            ("Samuel Úria", "A"),
            ("Samuel Uria", "A"),
        ]
    ).write_parquet(paths.listening_events_clean)
    second = build_curated_artist_tables(paths=paths)

    assert second.artist_aliases.get_column("artist_id").to_list() == [
        artist_id,
        artist_id,
    ]


def test_build_refuses_partial_existing_artist_state(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    paths.interim_dir.mkdir(parents=True)
    paths.curated_dir.mkdir(parents=True)
    _events([("Artist", "Song")]).write_parquet(paths.listening_events_clean)
    pl.DataFrame({"artist_id": ["id"], "canonical_name": ["Artist"]}).write_parquet(
        paths.curated_artists
    )

    try:
        build_curated_artist_tables(paths=paths)
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("Expected partial curated state to fail")


def _resolution(rows: list[tuple[str, str, str | None]]):
    clusters = pl.DataFrame(
        {
            "observed_artist_name": [row[0] for row in rows],
            "cluster_key": [row[1] for row in rows],
            "existing_artist_id": [row[2] for row in rows],
        },
        schema={
            "observed_artist_name": pl.String,
            "cluster_key": pl.String,
            "existing_artist_id": pl.String,
        },
    )
    return ArtistResolutionResult(
        evidence=pl.DataFrame(),
        decisions=pl.DataFrame(),
        clusters=clusters,
        blocked_merges=pl.DataFrame(),
    )
