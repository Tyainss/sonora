from uuid import UUID

import polars as pl

from sonora.data.curate_tracks import build_curated_track_tables
from sonora.data.paths import DataPaths


def _write_inputs(paths: DataPaths, rows: list[tuple[str, str, str | None]]) -> None:
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    artist_id = "11111111-1111-4111-8111-111111111111"

    pl.DataFrame(
        {
            "artist_id": [artist_id],
            "canonical_name": ["Artist"],
        }
    ).write_parquet(paths.curated_artists)
    pl.DataFrame(
        {
            "observed_artist_name": ["Artist"],
            "artist_id": [artist_id],
        }
    ).write_parquet(paths.curated_artist_aliases)
    pl.DataFrame(
        {
            "user_id": ["user"] * len(rows),
            "listened_at": [None] * len(rows),
            "artist_name": [artist for artist, _, _ in rows],
            "track_name": [track for _, track, _ in rows],
            "album_name": [None] * len(rows),
            "artist_mbid": [None] * len(rows),
            "track_mbid": [mbid for _, _, mbid in rows],
            "album_mbid": [None] * len(rows),
            "source": ["lastfm"] * len(rows),
        },
        schema_overrides={
            "artist_mbid": pl.String,
            "track_mbid": pl.String,
            "album_mbid": pl.String,
        },
    ).write_parquet(paths.listening_events_clean)


def test_build_curated_tracks_collapses_versions_and_uses_song_name(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(
        paths,
        [
            ("Artist", "Song - 2023 Remaster", None),
            ("Artist", "Song - 2023 Remaster", None),
            ("Artist", "Song (Live)", None),
        ],
    )

    build = build_curated_track_tables(paths=paths)

    assert build.tracks.height == 1
    assert build.tracks.row(0, named=True)["canonical_name"] == "Song"
    assert build.track_aliases.height == 2
    assert build.track_aliases.get_column("track_id").n_unique() == 1
    UUID(build.tracks.row(0, named=True)["track_id"])


def test_build_curated_tracks_preserves_ids_across_reruns(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(paths, [("Artist", "Song", None)])
    first = build_curated_track_tables(paths=paths)
    track_id = first.tracks.row(0, named=True)["track_id"]

    _write_inputs(
        paths,
        [
            ("Artist", "Song", None),
            ("Artist", "Song (Live)", None),
        ],
    )
    second = build_curated_track_tables(paths=paths)

    assert second.track_aliases.get_column("track_id").unique().to_list() == [track_id]


def test_build_refuses_partial_existing_track_state(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(paths, [("Artist", "Song", None)])
    pl.DataFrame(
        {
            "track_id": ["11111111-1111-4111-8111-111111111111"],
            "artist_id": ["11111111-1111-4111-8111-111111111111"],
            "canonical_name": ["Song"],
        }
    ).write_parquet(paths.curated_tracks)

    try:
        build_curated_track_tables(paths=paths)
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("Expected partial curated track state to fail")
