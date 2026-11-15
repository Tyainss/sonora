import json

import polars as pl
import pytest

from sonora.data.paths import DataPaths
from sonora.similarity.evaluation import (
    build_artist_similarity_evaluation,
    load_artist_similarity_evaluation,
    score_artist_similarity,
)


def _inputs(paths, *, changed=False):
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    paths.artist_similarity_references.parent.mkdir(parents=True, exist_ok=True)
    ids = [f"id-{i:03d}" for i in range(501 if changed else 500)]
    pl.DataFrame(
        {
            "artist_id": ids,
            "canonical_name": [f"Artist {i}" for i in range(len(ids))],
        }
    ).write_parquet(paths.curated_artists)
    events = [("user", artist_id) for artist_id in ids]
    events += [("user", "id-000")] * 2
    if changed:
        events += [("user", "id-500")] * 10
    pl.DataFrame(
        events, schema={"user_id": pl.String, "artist_id": pl.String}, orient="row"
    ).write_parquet(paths.curated_listening_events)
    paths.artist_similarity_references.write_text(
        json.dumps({"Artist 0": [f"Artist {i}" for i in range(1, 6)]}),
        encoding="utf-8",
    )


def test_freezes_top_500_and_resolves_ranked_references(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths)
    output = build_artist_similarity_evaluation(paths=paths)
    snapshot = load_artist_similarity_evaluation(paths=paths)
    assert output == paths.artist_similarity_evaluation
    assert len(snapshot["candidate_artist_ids"]) == 500
    assert snapshot["candidate_artist_ids"][0] == "id-000"
    assert snapshot["references"][0]["artist_id"] == "id-000"
    assert [row["artist_id"] for row in snapshot["references"][0]["matches"]] == [
        f"id-{i:03d}" for i in range(1, 6)
    ]

    original = output.read_bytes()
    _inputs(paths, changed=True)
    assert build_artist_similarity_evaluation(paths=paths) == output
    assert output.read_bytes() == original
    assert "id-500" not in load_artist_similarity_evaluation(paths=paths)[
        "candidate_artist_ids"
    ]


def test_updates_references_without_changing_pool(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths)
    build_artist_similarity_evaluation(paths=paths)
    first = load_artist_similarity_evaluation(paths=paths)
    paths.artist_similarity_references.write_text(
        json.dumps({"Artist 0": [f"Artist {i}" for i in range(5, 0, -1)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="References changed"):
        load_artist_similarity_evaluation(paths=paths)
    with pytest.raises(ValueError, match="References changed"):
        build_artist_similarity_evaluation(paths=paths)
    build_artist_similarity_evaluation(paths=paths, update_references=True)
    second = load_artist_similarity_evaluation(paths=paths)
    assert second["candidate_artist_ids"] == first["candidate_artist_ids"]
    assert [row["artist_id"] for row in second["references"][0]["matches"]] == [
        f"id-{i:03d}" for i in range(5, 0, -1)
    ]


def test_rejects_reference_not_in_candidate_pool(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths, changed=True)
    paths.artist_similarity_references.write_text(
        json.dumps({
            "Artist 0": ["Artist 99", "Artist 1", "Artist 2", "Artist 3", "Artist 4"]
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found in frozen top 500"):
        build_artist_similarity_evaluation(paths=paths)
    assert not paths.artist_similarity_evaluation.exists()


def test_scores_known_matches_and_missing_queries():
    snapshot = {
        "candidate_artist_ids": ["a", "b", "c", "d", "e", "f", "g"],
        "references": [
            {
                "artist_id": "a",
                "canonical_name": "Alpha",
                "matches": [{"artist_id": name} for name in ("b", "c", "d", "e", "f")],
            }
        ],
    }
    scores = score_artist_similarity(
        snapshot,
        {
            "TF-IDF": {"a": ["g", "b", "c", "d", "e", "f"]},
            "BGE-M3 tags": {},
        },
    )
    assert scores.filter(pl.col("Approach") == "TF-IDF").row(0, named=True) == {
        "Approach": "TF-IDF", "Artist": "Alpha", "Available": True,
        "Top 5": 4, "Top 10": 5,
    }
    assert scores.filter(pl.col("Approach") == "BGE-M3 tags").row(0, named=True)[
        "Top 5"
    ] == 0
    with pytest.raises(ValueError, match="Invalid"):
        score_artist_similarity(snapshot, {"Invalid": {"a": ["g", "g"]}})
    with pytest.raises(ValueError, match="Invalid"):
        score_artist_similarity(snapshot, {"Invalid": {"a": ["x"]}})
