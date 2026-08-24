import polars as pl
import pytest

from sonora.data.track_resolution import (
    build_track_song_keys,
    resolve_track_identities,
)


def _events(rows: list[tuple[str, str, str | None]]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "artist_id": [artist_id for artist_id, _, _ in rows],
            "track_name": [track_name for _, track_name, _ in rows],
            "track_mbid": [track_mbid for _, _, track_mbid in rows],
        },
        schema_overrides={"track_mbid": pl.String},
    ).lazy()


def _cluster_map(result):
    return {
        (row["artist_id"], row["observed_track_name"]): row["cluster_key"]
        for row in result.clusters.iter_rows(named=True)
    }


def test_song_keys_collapse_known_metadata_but_keep_title_text():
    catalogue = pl.DataFrame(
        {
            "artist_id": ["artist"] * 6,
            "track_name": [
                "Song",
                "Song - 2023 Remaster",
                "Song (Live)",
                "Song [Explicit]",
                "Song (with Guest)",
                "The World (Is Going Up in Flames)",
            ],
        }
    )

    keys = build_track_song_keys(catalogue)
    by_name = {
        row["track_name"]: row for row in keys.iter_rows(named=True)
    }

    assert by_name["Song - 2023 Remaster"]["song_key"] == "song"
    assert by_name["Song (Live)"]["song_key"] == "song"
    assert by_name["Song [Explicit]"]["song_key"] == "song"
    assert by_name["Song (with Guest)"]["song_key"] == "song"
    assert by_name["The World (Is Going Up in Flames)"]["song_key"] == (
        "the world is going up in flames"
    )


def test_punctuation_only_titles_keep_distinct_safe_keys():
    catalogue = pl.DataFrame(
        {
            "artist_id": ["artist", "artist"],
            "track_name": [".", ".."],
        }
    )

    keys = build_track_song_keys(catalogue)

    assert keys.get_column("song_key").to_list() == ["raw:.", "raw:.."]


def test_same_track_mbid_bridges_different_titles_within_one_artist():
    result = resolve_track_identities(
        _events(
            [
                ("artist", "Kirinaki Shima", "shared-mbid"),
                ("artist", "kirinakijima", "shared-mbid"),
            ]
        )
    )
    clusters = _cluster_map(result)

    assert clusters[("artist", "Kirinaki Shima")] == clusters[
        ("artist", "kirinakijima")
    ]


def test_track_mbid_does_not_bridge_across_artists():
    result = resolve_track_identities(
        _events(
            [
                ("artist-a", "Song A", "shared-mbid"),
                ("artist-b", "Song B", "shared-mbid"),
            ]
        )
    )
    clusters = _cluster_map(result)

    assert clusters[("artist-a", "Song A")] != clusters[("artist-b", "Song B")]


def test_existing_track_ids_are_never_silently_merged():
    events = _events(
        [
            ("artist", "Song", None),
            ("artist", "Song (Live)", None),
        ]
    )
    existing_aliases = pl.DataFrame(
        {
            "artist_id": ["artist", "artist"],
            "observed_track_name": ["Song", "Song (Live)"],
            "track_id": ["track-1", "track-2"],
        }
    )

    with pytest.raises(ValueError, match="multiple existing track IDs"):
        resolve_track_identities(events, existing_aliases=existing_aliases)
