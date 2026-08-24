import polars as pl
import pytest

from sonora.data.artist_resolution import (
    ArtistResolutionConfig,
    cluster_artist_aliases,
    decide_artist_identity,
)


def _evidence_row(**overrides: object) -> dict[str, object]:
    row = {
        "observed_artist_name_left": "Artist One",
        "artist_name_comparison_left": "artist one",
        "observed_artist_name_right": "Artist Two",
        "artist_name_comparison_right": "artist two",
        "candidate_from_exact_name": False,
        "candidate_from_shared_token": False,
        "candidate_from_shared_ngrams": True,
        "candidate_from_shared_tracks": False,
        "blocking_shared_token_count": 0,
        "blocking_shared_ngram_count": 2,
        "blocking_shared_track_count": 0,
        "name_levenshtein_similarity": 0.80,
        "name_wratio": 0.80,
        "left_scrobble_count": 10,
        "right_scrobble_count": 10,
        "left_track_count": 5,
        "right_track_count": 5,
        "shared_track_count": 0,
        "track_jaccard": 0.0,
        "track_containment": 0.0,
        "shared_track_scrobble_share_left": 0.0,
        "shared_track_scrobble_share_right": 0.0,
        "left_track_mbid_count": 0,
        "right_track_mbid_count": 0,
        "shared_track_mbid_count": 0,
        "track_mbid_jaccard": None,
        "track_mbid_containment": None,
        "left_album_count": 1,
        "right_album_count": 1,
        "shared_album_count": 0,
        "album_jaccard": 0.0,
        "album_containment": 0.0,
        "left_artist_mbid_count": 0,
        "right_artist_mbid_count": 0,
        "shared_artist_mbid_count": 0,
        "artist_mbid_relation": "missing",
    }
    row.update(overrides)
    return row


def _cluster_events(*names: str) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "artist_name": list(names),
            "track_name": [f"Track {index}" for index, _ in enumerate(names)],
            "artist_mbid": [None] * len(names),
        }
    ).lazy()


def _decision(**overrides: object) -> dict[str, object]:
    row = _evidence_row(**overrides)
    return decide_artist_identity(pl.DataFrame([row])).row(0, named=True)


def test_exact_comparison_name_merges_without_identifier_conflict():
    row = _decision(
        candidate_from_exact_name=True,
        name_levenshtein_similarity=1.0,
        name_wratio=1.0,
    )

    assert row["identity_decision"] == "merge"
    assert row["decision_rule"] == "same_comparison_name"


def test_near_exact_name_can_merge_without_catalogue_support():
    row = _decision(
        name_levenshtein_similarity=0.97,
        name_wratio=0.98,
    )

    assert row["identity_decision"] == "merge"
    assert row["decision_rule"] == "near_exact_name"


def test_name_and_catalogue_support_can_merge():
    row = _decision(
        name_levenshtein_similarity=0.40,
        name_wratio=0.91,
        shared_track_count=4,
        track_containment=0.75,
    )

    assert row["identity_decision"] == "merge"
    assert row["decision_rule"] == "name_and_catalogue"


def test_bidirectional_catalogue_support_can_merge_without_name_support():
    row = _decision(
        name_levenshtein_similarity=0.40,
        name_wratio=0.53,
        shared_track_count=7,
        track_containment=0.54,
        shared_track_scrobble_share_left=0.74,
        shared_track_scrobble_share_right=0.62,
    )

    assert row["identity_decision"] == "merge"
    assert row["decision_rule"] == "bidirectional_catalogue"


def test_one_sided_catalogue_support_stays_possible():
    row = _decision(
        name_levenshtein_similarity=0.30,
        name_wratio=0.45,
        shared_track_count=7,
        track_containment=0.90,
        shared_track_scrobble_share_left=1.0,
        shared_track_scrobble_share_right=0.21,
    )

    assert row["identity_decision"] == "possible_match"
    assert row["decision_rule"] == "strong_catalogue"


def test_mbid_conflict_needs_exceptionally_strong_support_to_merge():
    merged = _decision(
        artist_mbid_relation="conflict",
        name_levenshtein_similarity=0.40,
        name_wratio=0.90,
        shared_track_count=33,
        track_containment=0.825,
    )
    blocked = _decision(
        artist_mbid_relation="conflict",
        name_levenshtein_similarity=0.89,
        name_wratio=0.94,
        shared_track_count=0,
        track_containment=0.0,
    )

    assert merged["identity_decision"] == "merge"
    assert merged["decision_rule"] == "mbid_conflict_overridden_by_catalogue"
    assert blocked["identity_decision"] == "possible_match"
    assert blocked["decision_rule"] == "mbid_conflict_with_support"


def test_shared_artist_mbid_merges():
    row = _decision(
        artist_mbid_relation="shared",
        shared_artist_mbid_count=1,
    )

    assert row["identity_decision"] == "merge"
    assert row["decision_rule"] == "shared_artist_mbid"


def test_weak_candidate_is_rejected():
    row = _decision()

    assert row["identity_decision"] == "reject"
    assert row["decision_rule"] == "insufficient_evidence"


def test_cluster_compatibility_blocks_naive_transitivity():
    names = pl.DataFrame(
        {"observed_artist_name": ["A", "B", "C"]},
        schema={"observed_artist_name": pl.String},
    )
    decisions = pl.DataFrame(
        [
            {
                **_evidence_row(
                    observed_artist_name_left="A",
                    observed_artist_name_right="B",
                    name_wratio=1.0,
                    name_levenshtein_similarity=1.0,
                ),
                "identity_decision": "merge",
                "decision_rule": "same_comparison_name",
            },
            {
                **_evidence_row(
                    observed_artist_name_left="B",
                    observed_artist_name_right="C",
                    name_wratio=0.98,
                    name_levenshtein_similarity=0.97,
                ),
                "identity_decision": "merge",
                "decision_rule": "near_exact_name",
            },
            {
                **_evidence_row(
                    observed_artist_name_left="A",
                    observed_artist_name_right="C",
                ),
                "identity_decision": "reject",
                "decision_rule": "insufficient_evidence",
            },
        ]
    )

    clusters, blocked = cluster_artist_aliases(
        _cluster_events("A", "B", "C"),
        names,
        decisions,
    )

    cluster_by_name = dict(
        clusters.select("observed_artist_name", "cluster_key").iter_rows()
    )
    assert cluster_by_name["A"] == cluster_by_name["B"]
    assert cluster_by_name["C"] != cluster_by_name["A"]
    assert blocked.get_column("reason").to_list() == ["cluster_incompatible"]


def test_cluster_can_use_combined_catalogue_support():
    names = pl.DataFrame(
        {"observed_artist_name": ["A", "B", "C"]},
        schema={"observed_artist_name": pl.String},
    )
    events = pl.DataFrame(
        {
            "artist_name": ["A", "B", "B", "C", "C"],
            "track_name": ["A only", "Shared 1", "Shared 2", "Shared 1", "Shared 2"],
            "artist_mbid": [None] * 5,
        }
    ).lazy()
    decisions = pl.DataFrame(
        [
            {
                **_evidence_row(
                    observed_artist_name_left="A",
                    observed_artist_name_right="B",
                    name_wratio=1.0,
                    name_levenshtein_similarity=1.0,
                ),
                "identity_decision": "merge",
                "decision_rule": "same_comparison_name",
            },
            {
                **_evidence_row(
                    observed_artist_name_left="B",
                    observed_artist_name_right="C",
                    name_wratio=0.90,
                    name_levenshtein_similarity=0.50,
                    shared_track_count=2,
                    track_containment=1.0,
                ),
                "identity_decision": "merge",
                "decision_rule": "name_and_catalogue",
            },
        ]
    )

    clusters, blocked = cluster_artist_aliases(events, names, decisions)

    assert clusters.get_column("cluster_key").n_unique() == 1
    assert blocked.is_empty()


def test_existing_canonical_artists_are_not_merged_implicitly():
    names = pl.DataFrame(
        {"observed_artist_name": ["A", "B"]},
        schema={"observed_artist_name": pl.String},
    )
    existing_aliases = pl.DataFrame(
        {
            "observed_artist_name": ["A", "B"],
            "artist_id": ["artist-a", "artist-b"],
        }
    )
    decisions = pl.DataFrame(
        [
            {
                **_evidence_row(
                    observed_artist_name_left="A",
                    observed_artist_name_right="B",
                    name_wratio=1.0,
                    name_levenshtein_similarity=1.0,
                ),
                "identity_decision": "merge",
                "decision_rule": "same_comparison_name",
            }
        ]
    )

    clusters, blocked = cluster_artist_aliases(
        _cluster_events("A", "B"),
        names,
        decisions,
        existing_aliases=existing_aliases,
    )

    assert clusters.get_column("cluster_key").n_unique() == 2
    assert blocked.get_column("reason").to_list() == ["canonical_merge_required"]


def test_resolution_config_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="near_exact_wratio"):
        ArtistResolutionConfig(near_exact_wratio=1.1)
