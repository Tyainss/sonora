import polars as pl
import pytest

from sonora.similarity.tags import build_artist_tag_vectors, find_similar_artists


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
        find_similar_artists(vectors, "a", representation="tfidf")
    with pytest.raises(ValueError, match="top_k"):
        find_similar_artists(vectors, "a", representation="binary", top_k=0)
