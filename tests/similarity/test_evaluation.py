import csv
import json

import polars as pl
import pytest

from sonora.data.paths import DataPaths
from sonora.similarity.evaluation import (
    build_artist_similarity_evaluation,
    load_artist_similarity_evaluation,
    load_artist_similarity_reference_sets,
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
    tfidf = scores.filter(pl.col("Approach") == "TF-IDF").row(0, named=True)
    assert {key: value for key, value in tfidf.items() if key != "NDCG@10"} == {
        "Approach": "TF-IDF", "Artist": "Alpha", "Available": True,
        "Top 5": 4, "Top 10": 5,
    }
    assert 0 < tfidf["NDCG@10"] < 1
    assert scores.filter(pl.col("Approach") == "BGE-M3 tags").row(0, named=True)[
        "NDCG@10"
    ] is None
    assert scores.filter(pl.col("Approach") == "BGE-M3 tags").row(0, named=True)[
        "Top 5"
    ] == 0
    with pytest.raises(ValueError, match="Invalid"):
        score_artist_similarity(snapshot, {"Invalid": {"a": ["g", "g"]}})
    with pytest.raises(ValueError, match="Invalid"):
        score_artist_similarity(snapshot, {"Invalid": {"a": ["x"]}})


def test_ndcg_rewards_order_and_important_references():
    snapshot = {
        "candidate_artist_ids": ["a", "b", "c", "d", "e", "f", "g"],
        "references": [
            {
                "artist_id": "a",
                "canonical_name": "Alpha",
                "matches": [
                    {"artist_id": name} for name in ("b", "c", "d", "e", "f")
                ],
            }
        ],
    }
    rankings = {
        "Perfect": {"a": ["b", "c", "d", "e", "f", "g"]},
        "Reversed": {"a": ["f", "e", "d", "c", "b", "g"]},
        "Strong first": {"a": ["b", "g"]},
        "Weak first": {"a": ["f", "g"]},
        "No matches": {"a": ["g"]},
        "Unavailable": {},
    }
    scores = {
        row["Approach"]: row
        for row in score_artist_similarity(snapshot, rankings).iter_rows(named=True)
    }

    assert scores["Perfect"]["NDCG@10"] == pytest.approx(1.0)
    assert 0 < scores["Reversed"]["NDCG@10"] < 1
    assert scores["Strong first"]["NDCG@10"] > scores["Weak first"]["NDCG@10"]
    assert scores["No matches"]["NDCG@10"] == 0.0
    assert scores["Unavailable"]["NDCG@10"] is None
    assert scores["Perfect"]["Top 5"] == scores["Reversed"]["Top 5"] == 5


def _independent_references(paths, *, matches=None):
    metadata = {
        "schema_version": 1,
        "reference_sets": {
            "listener": {
                "label": "Listener",
                "file": paths.artist_similarity_references.name,
            },
            "llm": {
                "label": "Independent LLM",
                "file": paths.artist_similarity_llm_references.name,
            },
        },
    }
    paths.artist_similarity_reference_metadata.write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    columns = [
        "reference_artist",
        "reference_artist_id",
        "rank",
        "similar_artist",
        "similar_artist_id",
        "why_similar",
    ]
    with paths.artist_similarity_llm_references.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for rank, i in enumerate(matches or (5, 4, 3, 2, 1), start=1):
            writer.writerow(
                {
                    "reference_artist": "Artist 0",
                    "reference_artist_id": "id-000",
                    "rank": rank,
                    "similar_artist": f"Artist {i}",
                    "similar_artist_id": f"id-{i:03d}",
                    "why_similar": f"Reason {i}",
                }
            )


def test_loads_separate_references_over_the_same_pool(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths)
    build_artist_similarity_evaluation(paths=paths)
    _independent_references(paths)
    sets = load_artist_similarity_reference_sets(paths=paths)
    assert list(sets) == ["Listener", "Independent LLM"]
    human, llm = sets.values()
    assert llm["candidate_artist_ids"] == human["candidate_artist_ids"]
    assert [r["artist_id"] for r in llm["references"][0]["matches"]] == [
        "id-005", "id-004", "id-003", "id-002", "id-001"
    ]
    assert llm["references"][0]["matches"][0]["why_similar"] == "Reason 5"
    assert human["references"][0]["matches"][0]["artist_id"] == "id-001"

    rankings = {"TF-IDF": {"id-000": [f"id-{i:03d}" for i in range(1, 6)]}}
    listener_score = score_artist_similarity(human, rankings)["NDCG@10"][0]
    llm_score = score_artist_similarity(llm, rankings)["NDCG@10"][0]
    assert listener_score == pytest.approx(1.0)
    assert llm_score < listener_score


@pytest.mark.parametrize(
    "bad_matches",
    [
        (1, 2, 3, 4, 500),  # Outside the frozen pool.
        (1, 2, 3, 4, 4),  # Duplicate artist.
        (0, 1, 2, 3, 4),  # Self-match.
    ],
)
def test_rejects_invalid_llm_references(tmp_path, bad_matches):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths)
    build_artist_similarity_evaluation(paths=paths)
    _independent_references(paths, matches=bad_matches)
    with pytest.raises(ValueError, match="Invalid"):
        load_artist_similarity_reference_sets(paths=paths)


def test_rejects_missing_llm_ranks_and_wrong_names(tmp_path):
    paths = DataPaths(data_dir=tmp_path / "data")
    _inputs(paths)
    build_artist_similarity_evaluation(paths=paths)
    _independent_references(paths)
    with paths.artist_similarity_llm_references.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[4]["rank"] = "4"
    with paths.artist_similarity_llm_references.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="rank"):
        load_artist_similarity_reference_sets(paths=paths)

    rows[4]["rank"] = "5"
    rows[0]["similar_artist"] = "Wrong name"
    with paths.artist_similarity_llm_references.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Invalid"):
        load_artist_similarity_reference_sets(paths=paths)
