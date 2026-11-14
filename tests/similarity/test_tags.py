import polars as pl
import pytest

from sonora.similarity.tags import (
    build_artist_tag_vectors,
    clean_artist_tags,
    find_similar_artists,
)


def _artist_tags(rows):
    return pl.DataFrame(
        [
            {
                "artist_id": artist_id,
                "canonical_name": name,
                "status": status,
                "tags": [{"name": tag, "weight": weight} for tag, weight in tags],
            }
            for artist_id, name, status, tags in rows
        ],
        schema={
            "artist_id": pl.String,
            "canonical_name": pl.String,
            "status": pl.String,
            "tags": pl.List(pl.Struct({"name": pl.String, "weight": pl.Int64})),
        },
    )


def test_binary_and_weighted_tags_rank_differently():
    dataset = _artist_tags(
        [
            ("source", "Source", "success", [("rock", 100), ("jazz", 10)]),
            ("rock", "Rocker", "success", [("rock", 100)]),
            ("jazz", "Jazzer", "success", [("jazz", 100)]),
            ("other", "Other", "success", [("electronic", 100)]),
        ]
    )
    vectors = build_artist_tag_vectors(dataset)

    binary = find_similar_artists(vectors, "source", representation="binary")
    weighted = find_similar_artists(vectors, "source", representation="weighted")

    assert binary["artist_id"].to_list() == ["jazz", "rock"]
    assert binary["similarity"].to_list() == pytest.approx([2**-0.5] * 2)
    assert weighted["artist_id"].to_list() == ["rock", "jazz"]
    assert weighted["similarity"][0] > weighted["similarity"][1]


def test_tfidf_prioritizes_distinctive_shared_tags():
    dataset = _artist_tags(
        [
            ("source", "Source", "success", [("rock", 100), ("jazz", 100)]),
            ("rock1", "A Rocker", "success", [("rock", 100)]),
            ("rock2", "B Rocker", "success", [("rock", 100)]),
            ("rock3", "C Rocker", "success", [("rock", 100)]),
            ("jazz", "Z Jazzer", "success", [("jazz", 100)]),
        ]
    )
    vectors = build_artist_tag_vectors(dataset, clean_tags=True)

    binary = find_similar_artists(vectors, "source", representation="binary")
    tfidf = find_similar_artists(vectors, "source", representation="tfidf")

    assert binary["artist_id"][0] == "rock1"
    assert tfidf["artist_id"][0] == "jazz"
    assert tfidf["similarity"][0] > tfidf["similarity"][1]


def test_tfidf_excludes_artists_without_shared_tags():
    vectors = build_artist_tag_vectors(
        _artist_tags(
            [
                ("a", "Alpha", "success", [("rock", 100)]),
                ("b", "Beta", "success", [("jazz", 100)]),
            ]
        )
    )
    assert find_similar_artists(vectors, "a", representation="tfidf").is_empty()


def test_skips_missing_or_unverified_tags_and_does_not_return_zero_matches():
    vectors = build_artist_tag_vectors(
        _artist_tags(
            [
                ("a", "Alpha", "success", [("folk", 100)]),
                ("b", "Beta", "success", [("folk", 50)]),
                ("c", "Charlie", "empty", []),
                ("d", "Delta", "identity_mismatch", [("folk", 100)]),
                ("e", "Echo", "success", [("ambient", 100)]),
            ]
        )
    )

    assert vectors.artists["artist_id"].to_list() == ["a", "b", "e"]
    result = find_similar_artists(vectors, "a", representation="weighted")
    assert result["artist_id"].to_list() == ["b"]
    with pytest.raises(ValueError, match="no usable tags"):
        find_similar_artists(vectors, "c", representation="binary")


def test_raw_tag_spelling_is_preserved():
    vectors = build_artist_tag_vectors(
        _artist_tags(
            [
                ("a", "Alpha", "success", [("hip-hop", 100)]),
                ("b", "Beta", "success", [("hip hop", 100)]),
            ]
        )
    )
    assert find_similar_artists(vectors, "a", representation="binary").is_empty()


def test_zero_weights_do_not_create_weighted_similarity():
    vectors = build_artist_tag_vectors(
        _artist_tags(
            [
                ("a", "Alpha", "success", [("rock", 0)]),
                ("b", "Beta", "success", [("rock", 100)]),
            ]
        )
    )
    assert find_similar_artists(vectors, "a", representation="weighted").is_empty()
    assert find_similar_artists(vectors, "a", representation="binary").height == 1


def test_rejects_duplicate_artist_ids():
    dataset = _artist_tags(
        [
            ("a", "Alpha", "success", [("rock", 100)]),
            ("a", "Beta", "success", [("jazz", 100)]),
        ]
    )
    with pytest.raises(ValueError, match="duplicate artist IDs"):
        build_artist_tag_vectors(dataset)


def test_rejects_no_usable_tags_and_invalid_parameters():
    with pytest.raises(ValueError, match="No artists"):
        build_artist_tag_vectors(_artist_tags([("a", "Alpha", "empty", [])]))
    vectors = build_artist_tag_vectors(
        _artist_tags([("a", "Alpha", "success", [("rock", 100)])])
    )
    with pytest.raises(ValueError, match="Representation"):
        find_similar_artists(vectors, "a", representation="unknown")
    with pytest.raises(ValueError, match="top_k"):
        find_similar_artists(vectors, "a", representation="binary", top_k=0)


def test_cleaning_merges_equivalent_tags_using_the_highest_weight():
    original = [
        {"name": " Hip-Hop ", "weight": 100},
        {"name": "hip hop", "weight": 24},
        {"name": "LOFI", "weight": 40},
        {"name": "Lo-Fi", "weight": 60},
        {"name": "alternative rnb", "weight": 25},
        {"name": "alternative r&b", "weight": 10},
        {"name": "post-rock", "weight": 35},
        {"name": "post rock", "weight": 50},
    ]
    assert clean_artist_tags(original) == [
        {"name": "hip hop", "weight": 100},
        {"name": "lo fi", "weight": 60},
        {"name": "alternative r&b", "weight": 25},
        {"name": "post rock", "weight": 50},
    ]
    assert original[0]["name"] == " Hip-Hop "


def test_cleaning_removes_known_artifacts_without_discarding_context():
    raw = [
        ("All", 80),
        ("my top songs", 70),
        ("funk_add_to_lidarr_batch_18", 60),
        ("seen live twice", 50),
        ("FDC 2023 Composers", 40),
        ("upcoming album 2024", 30),
        ("under 2000 listeners", 20),
        ("Spotify", 10),
        ("american", 100),
        ("female vocalists", 90),
        ("portuguese", 80),
        ("post-brexit new wave", 70),
        ("live jazz", 60),
        ("experimental hip-hop", 50),
        ("rare niche style", 40),
    ]
    assert clean_artist_tags([{"name": name, "weight": weight} for name, weight in raw]) == [
        {"name": "american", "weight": 100},
        {"name": "female vocalists", "weight": 90},
        {"name": "portuguese", "weight": 80},
        {"name": "post brexit new wave", "weight": 70},
        {"name": "live jazz", "weight": 60},
        {"name": "experimental hip hop", "weight": 50},
        {"name": "rare niche style", "weight": 40},
    ]


def test_cleaned_similarity_joins_aliases_and_excludes_artifact_only_artists():
    dataset = _artist_tags(
        [
            ("a", "Alpha", "success", [("Hip-Hop", 100), ("hip hop", 10)]),
            ("b", "Beta", "success", [("hiphop", 100)]),
            ("c", "Charlie", "success", [("my top songs", 100)]),
            ("d", "Delta", "success", [("rock", 100)]),
        ]
    )
    raw = build_artist_tag_vectors(dataset)
    cleaned = build_artist_tag_vectors(dataset, clean_tags=True)

    assert raw.artists.height == 4
    assert cleaned.artists["artist_id"].to_list() == ["a", "b", "d"]
    assert find_similar_artists(raw, "a", representation="binary").is_empty()
    assert find_similar_artists(cleaned, "a", representation="binary")[
        "artist_id"
    ].to_list() == ["b"]
    assert find_similar_artists(cleaned, "a", representation="weighted")[
        "similarity"
    ][0] == pytest.approx(1.0)


def test_negative_weights_are_rejected_even_if_cleaning_would_discard_them():
    dataset = _artist_tags(
        [("a", "Alpha", "success", [("my top songs", -1), ("rock", 100)])]
    )
    with pytest.raises(ValueError, match="non-negative"):
        build_artist_tag_vectors(dataset, clean_tags=True)
