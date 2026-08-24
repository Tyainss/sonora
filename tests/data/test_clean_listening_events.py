import json
from datetime import UTC, datetime

import polars as pl
import pytest

from sonora.data.clean_listening_events import (
    build_listening_events_clean,
    clean_listening_events,
)
from sonora.data.paths import DataPaths


def _raw_record(
    *,
    artist_name=" Samuel Uria ",
    track_name=" Song ",
    album_name=" Album ",
    timestamp_unix=1_584_617_400,
    track_mbid=" track-mbid ",
    artist_mbid=" artist-mbid ",
    album_mbid=" album-mbid ",
):
    return {
        "artist_name": artist_name,
        "track_name": track_name,
        "album_name": album_name,
        "timestamp_unix": timestamp_unix,
        "mbid": track_mbid,
        "raw": {
            "name": track_name,
            "artist": {"#text": artist_name, "mbid": artist_mbid},
            "album": {"#text": album_name, "mbid": album_mbid},
        },
    }


def _write_raw_snapshot(paths: DataPaths, records: list[dict]) -> None:
    paths.raw_lastfm_dir.mkdir(parents=True)
    with paths.raw_lastfm_scrobbles.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def test_clean_listening_events_applies_only_safe_structural_cleaning():
    raw = pl.DataFrame(
        [
            _raw_record(
                artist_name=" Samuel Úria ",
                track_name=" Canção! ",
                album_name="   ",
                track_mbid="   ",
                artist_mbid=" artist-id ",
                album_mbid="   ",
            )
        ]
    ).lazy()

    result = clean_listening_events(raw, user_id="user-1").collect()

    assert result.columns == [
        "user_id",
        "listened_at",
        "artist_name",
        "track_name",
        "album_name",
        "artist_mbid",
        "track_mbid",
        "album_mbid",
        "source",
    ]
    assert result.row(0, named=True) == {
        "user_id": "user-1",
        "listened_at": datetime(2020, 3, 19, 11, 30, tzinfo=UTC),
        "artist_name": "Samuel Úria",
        "track_name": "Canção!",
        "album_name": None,
        "artist_mbid": "artist-id",
        "track_mbid": None,
        "album_mbid": None,
        "source": "lastfm",
    }
    assert result.schema["listened_at"] == pl.Datetime("us", "UTC")


def test_clean_listening_events_deduplicates_only_equal_cleaned_events():
    raw = pl.DataFrame(
        [
            _raw_record(),
            _raw_record(
                artist_name="Samuel Uria", track_name="Song", album_name="Album"
            ),
            _raw_record(
                artist_name="Samuel Uria",
                track_name="Song",
                album_name="Album",
                track_mbid="different-track-mbid",
            ),
        ]
    ).lazy()

    result = clean_listening_events(raw, user_id="user-1").collect()

    assert result.height == 2
    assert set(result["track_mbid"].to_list()) == {"track-mbid", "different-track-mbid"}


def test_build_listening_events_clean_writes_expected_parquet(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_raw_snapshot(paths, [_raw_record()])

    output = build_listening_events_clean(user_id="user-1", paths=paths)

    assert output == paths.listening_events_clean
    assert output.is_file()
    result = pl.read_parquet(output)
    assert result.height == 1
    assert result["artist_name"].item() == "Samuel Uria"


def test_build_rejects_missing_required_event_values_before_writing(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_raw_snapshot(paths, [_raw_record(artist_name="   ")])

    with pytest.raises(ValueError, match="artist_name=1"):
        build_listening_events_clean(user_id="user-1", paths=paths)

    assert not paths.listening_events_clean.exists()


def test_clean_listening_events_rejects_missing_raw_columns():
    raw = pl.DataFrame({"artist_name": ["Artist"]}).lazy()

    with pytest.raises(ValueError, match="missing required columns"):
        clean_listening_events(raw, user_id="user-1")


def test_clean_listening_events_rejects_blank_user_id():
    raw = pl.DataFrame([_raw_record()]).lazy()

    with pytest.raises(ValueError, match="user_id"):
        clean_listening_events(raw, user_id="  ")


def test_build_rejects_existing_partial_output(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _write_raw_snapshot(paths, [_raw_record()])
    partial_path = paths.listening_events_clean.with_suffix(".parquet.partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="partial clean listening-events build"):
        build_listening_events_clean(user_id="user-1", paths=paths)

    assert not paths.listening_events_clean.exists()
