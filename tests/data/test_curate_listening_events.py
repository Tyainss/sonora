from datetime import UTC, datetime

import polars as pl
import pytest

from sonora.data.curate_listening_events import build_curated_listening_events
from sonora.data.paths import DataPaths

ARTIST_ID = "11111111-1111-4111-8111-111111111111"
OTHER_ARTIST_ID = "22222222-2222-4222-8222-222222222222"
TRACK_ID = "33333333-3333-4333-8333-333333333333"
OTHER_TRACK_ID = "44444444-4444-4444-8444-444444444444"


def _write_inputs(
    paths: DataPaths,
    *,
    clean_rows: list[dict],
    track_alias_rows: list[dict] | None = None,
    track_rows: list[dict] | None = None,
) -> None:
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    paths.curated_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "artist_id": [ARTIST_ID, OTHER_ARTIST_ID],
            "canonical_name": ["Artist", "Other Artist"],
        }
    ).write_parquet(paths.curated_artists)
    pl.DataFrame(
        {
            "observed_artist_name": ["Artist", "Artist Alias", "Other Artist"],
            "artist_id": [ARTIST_ID, ARTIST_ID, OTHER_ARTIST_ID],
        }
    ).write_parquet(paths.curated_artist_aliases)

    pl.DataFrame(
        track_rows
        or [
            {
                "track_id": TRACK_ID,
                "artist_id": ARTIST_ID,
                "canonical_name": "Song",
            },
            {
                "track_id": OTHER_TRACK_ID,
                "artist_id": ARTIST_ID,
                "canonical_name": "Other Song",
            },
        ]
    ).write_parquet(paths.curated_tracks)
    pl.DataFrame(
        track_alias_rows
        or [
            {
                "artist_id": ARTIST_ID,
                "observed_track_name": "Song",
                "track_id": TRACK_ID,
            },
            {
                "artist_id": ARTIST_ID,
                "observed_track_name": "Song (Live)",
                "track_id": TRACK_ID,
            },
            {
                "artist_id": ARTIST_ID,
                "observed_track_name": "Other Song",
                "track_id": OTHER_TRACK_ID,
            },
        ]
    ).write_parquet(paths.curated_track_aliases)

    pl.DataFrame(
        clean_rows,
        schema_overrides={
            "artist_mbid": pl.String,
            "track_mbid": pl.String,
            "album_mbid": pl.String,
        },
    ).write_parquet(paths.listening_events_clean)


def _clean_row(
    *,
    track_name: str,
    artist_name: str = "Artist",
    minute: int = 0,
    album_name: str | None = "Album",
) -> dict:
    return {
        "user_id": "user",
        "listened_at": datetime(2026, 1, 1, 12, minute, tzinfo=UTC),
        "artist_name": artist_name,
        "track_name": track_name,
        "album_name": album_name,
        "artist_mbid": None,
        "track_mbid": None,
        "album_mbid": None,
        "source": "lastfm",
    }


def test_build_curated_listening_events_maps_canonical_ids(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(
        paths,
        clean_rows=[
            _clean_row(
                artist_name="Artist Alias",
                track_name="Song (Live)",
            ),
            _clean_row(track_name="Other Song", minute=1, album_name=None),
        ],
    )

    output_path = build_curated_listening_events(paths=paths)
    events = pl.read_parquet(output_path)

    assert events.columns == [
        "user_id",
        "listened_at",
        "artist_id",
        "track_id",
        "album_name",
        "source",
    ]
    assert events.height == 2
    assert events.get_column("artist_id").to_list() == [ARTIST_ID, ARTIST_ID]
    assert events.get_column("track_id").to_list() == [TRACK_ID, OTHER_TRACK_ID]
    assert events.get_column("album_name").to_list() == ["Album", None]
    assert "UTC" in str(events.schema["listened_at"])


def test_build_preserves_events_that_become_identical_after_canonicalization(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    first = _clean_row(track_name="Song")
    second = _clean_row(track_name="Song (Live)")
    _write_inputs(paths, clean_rows=[first, second])

    build_curated_listening_events(paths=paths)
    events = pl.read_parquet(paths.curated_listening_events)

    assert events.height == 2
    assert events.unique().height == 1


def test_build_fails_when_a_track_alias_is_missing(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(paths, clean_rows=[_clean_row(track_name="Unknown Song")])

    with pytest.raises(ValueError, match="canonical track"):
        build_curated_listening_events(paths=paths)


def test_build_fails_when_track_belongs_to_another_artist(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_inputs(
        paths,
        clean_rows=[_clean_row(track_name="Song")],
        track_rows=[
            {
                "track_id": TRACK_ID,
                "artist_id": OTHER_ARTIST_ID,
                "canonical_name": "Song",
            }
        ],
        track_alias_rows=[
            {
                "artist_id": ARTIST_ID,
                "observed_track_name": "Song",
                "track_id": TRACK_ID,
            }
        ],
    )

    with pytest.raises(ValueError, match="different artist"):
        build_curated_listening_events(paths=paths)
