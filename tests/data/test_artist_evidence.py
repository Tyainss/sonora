import polars as pl
import pytest

from sonora.data.artist_candidates import (
    collect_artist_aliases,
    generate_artist_candidate_pairs,
)
from sonora.data.artist_evidence import (
    build_artist_candidate_evidence,
    score_artist_candidate_evidence,
)


def _events(rows: list[dict[str, object]]) -> pl.LazyFrame:
    return pl.DataFrame(rows).lazy()


def test_build_artist_candidate_evidence_combines_lexical_and_catalogue_signals():
    events = _events(
        [
            {
                "artist_name": "Samuel Uria",
                "track_name": "Canção A",
                "album_name": "Álbum",
                "artist_mbid": "artist-1",
                "track_mbid": "track-1",
            },
            {
                "artist_name": "Samuel Uria",
                "track_name": "Canção B",
                "album_name": "Álbum",
                "artist_mbid": "artist-1",
                "track_mbid": "track-2",
            },
            {
                "artist_name": "Samuel Úria",
                "track_name": "Cancao A",
                "album_name": "Album",
                "artist_mbid": "artist-1",
                "track_mbid": "track-1",
            },
        ]
    )

    result = build_artist_candidate_evidence(events)
    row = result.row(0, named=True)

    assert result.height == 1
    assert row["name_levenshtein_similarity"] == pytest.approx(1.0)
    assert row["name_wratio"] == pytest.approx(1.0)
    assert row["shared_track_count"] == 1
    assert row["track_jaccard"] == pytest.approx(0.5)
    assert row["track_containment"] == pytest.approx(1.0)
    assert row["shared_track_scrobble_share_left"] == pytest.approx(0.5)
    assert row["shared_track_scrobble_share_right"] == pytest.approx(1.0)
    assert row["shared_track_mbid_count"] == 1
    assert row["track_mbid_containment"] == pytest.approx(1.0)
    assert row["shared_album_count"] == 1
    assert row["album_containment"] == pytest.approx(1.0)
    assert row["artist_mbid_relation"] == "shared"


def test_semantic_track_suffixes_are_not_stripped_during_artist_evidence():
    events = _events(
        [
            {
                "artist_name": "Artist & One",
                "track_name": "Song",
                "album_name": None,
                "artist_mbid": None,
                "track_mbid": "track-1",
            },
            {
                "artist_name": "Artist And One",
                "track_name": "Song (Live)",
                "album_name": None,
                "artist_mbid": None,
                "track_mbid": "track-1",
            },
        ]
    )

    row = build_artist_candidate_evidence(events).row(0, named=True)

    assert row["shared_track_count"] == 0
    assert row["shared_track_mbid_count"] == 1
    assert row["artist_mbid_relation"] == "missing"


def test_artist_mbid_relation_distinguishes_shared_and_conflict():
    events = _events(
        [
            {
                "artist_name": "Artist A",
                "track_name": "One",
                "album_name": None,
                "artist_mbid": "mbid-1",
                "track_mbid": None,
            },
            {
                "artist_name": "Artist A",
                "track_name": "Two",
                "album_name": None,
                "artist_mbid": "mbid-extra",
                "track_mbid": None,
            },
            {
                "artist_name": "Artist-A",
                "track_name": "One",
                "album_name": None,
                "artist_mbid": "mbid-1",
                "track_mbid": None,
            },
            {
                "artist_name": "Artist B",
                "track_name": "Three",
                "album_name": None,
                "artist_mbid": "mbid-2",
                "track_mbid": None,
            },
        ]
    )
    aliases = collect_artist_aliases(events)
    candidates = generate_artist_candidate_pairs(aliases)
    scored = score_artist_candidate_evidence(events, candidates)

    shared = scored.filter(
        pl.col("observed_artist_name_left").is_in(["Artist A", "Artist-A"])
        & pl.col("observed_artist_name_right").is_in(["Artist A", "Artist-A"])
    ).row(0, named=True)

    assert shared["artist_mbid_relation"] == "shared"
    assert shared["shared_artist_mbid_count"] == 1

    manual_conflict = pl.DataFrame(
        {
            "observed_artist_name_left": ["Artist A"],
            "artist_name_comparison_left": ["artist a"],
            "observed_artist_name_right": ["Artist B"],
            "artist_name_comparison_right": ["artist b"],
        }
    )
    conflict = score_artist_candidate_evidence(events, manual_conflict).row(
        0, named=True
    )

    assert conflict["artist_mbid_relation"] == "conflict"
    assert conflict["shared_artist_mbid_count"] == 0


def test_score_artist_candidate_evidence_rejects_unknown_artist():
    events = _events(
        [
            {
                "artist_name": "Known Artist",
                "track_name": "Song",
                "album_name": None,
                "artist_mbid": None,
                "track_mbid": None,
            }
        ]
    )
    candidates = pl.DataFrame(
        {
            "observed_artist_name_left": ["Known Artist"],
            "artist_name_comparison_left": ["known artist"],
            "observed_artist_name_right": ["Unknown Artist"],
            "artist_name_comparison_right": ["unknown artist"],
        }
    )

    with pytest.raises(ValueError, match="absent from listening events"):
        score_artist_candidate_evidence(events, candidates)


def test_empty_candidate_set_returns_evidence_schema():
    events = _events(
        [
            {
                "artist_name": "Artist",
                "track_name": "Song",
                "album_name": None,
                "artist_mbid": None,
                "track_mbid": None,
            }
        ]
    )
    candidates = pl.DataFrame(
        schema={
            "observed_artist_name_left": pl.String,
            "artist_name_comparison_left": pl.String,
            "observed_artist_name_right": pl.String,
            "artist_name_comparison_right": pl.String,
        }
    )

    result = score_artist_candidate_evidence(events, candidates)

    assert result.is_empty()
    assert "track_containment" in result.columns
    assert "artist_mbid_relation" in result.columns
