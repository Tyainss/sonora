import json

import numpy as np
import polars as pl
import pytest

from sonora.data.paths import DataPaths
from sonora.similarity import embeddings


def _write_descriptions(paths, rows):
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "artist_id": pl.String,
            "canonical_name": pl.String,
            "status": pl.String,
            "summary": pl.String,
        },
        orient="row",
    ).write_parquet(paths.curated_artist_descriptions)


class FakeModel:
    max_seq_length = 8192

    def __init__(self):
        self.calls = []
        self.fail_after = None

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False}
        return {"input_ids": [[1] * (len(text.split()) + 2) for text in texts]}

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        assert normalize_embeddings is True and not show_progress_bar
        assert batch_size == embeddings.BATCH_SIZE
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Encoding stopped")
        self.calls.append(list(texts))
        vectors = np.zeros(
            (len(texts), embeddings.EMBEDDING_DIMENSIONS), dtype=np.float32
        )
        for index, text in enumerate(texts):
            vectors[index, 0] = 1.0
            vectors[index, 1] = sum(ord(char) for char in text) % 100 / 100
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = DataPaths(data_dir=tmp_path / "data")
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_model", lambda: model)
    monkeypatch.setattr(embeddings, "version", lambda package: "test-version")
    monkeypatch.setattr(embeddings, "BATCH_SIZE", 2)
    return paths, model


def test_build_and_reuse_embeddings(setup):
    paths, model = setup
    _write_descriptions(
        paths,
        [
            ("b", "Beta", "success", "Experimental <b>jazz</b>"),
            ("a", "Alpha", "success", "Indie folk"),
            ("c", "Gamma", "empty", None),
            ("d", "Delta", "success", "Read more on Last.fm"),
        ],
    )
    output = embeddings.build_artist_summary_embeddings(paths=paths)
    saved = pl.read_parquet(output)
    assert saved.schema == embeddings._SCHEMA
    assert saved["artist_id"].to_list() == ["a", "b"]
    assert len(model.calls) == 1
    assert model.calls[0] == ["Indie folk", "Experimental jazz"]
    assert not output.with_suffix(".parquet.partial").exists()
    assert np.allclose(
        np.linalg.norm(np.array(saved["embedding"].to_list()), axis=1), 1
    )

    metadata_path = paths.artist_summary_embeddings_metadata
    metadata = json.loads(metadata_path.read_text())
    assert metadata["model_revision"] == embeddings.MODEL_REVISION
    assert metadata["source_sha256"] == embeddings._sha256(
        paths.curated_artist_descriptions
    )
    assert metadata["embeddings_sha256"] == embeddings._sha256(output)
    assert metadata["artists"] == 2
    embeddings.build_artist_summary_embeddings(paths=paths)
    assert len(model.calls) == 1
    assert json.loads(metadata_path.read_text()) == metadata


def test_only_changed_and_new_summaries_are_encoded(setup):
    paths, model = setup
    _write_descriptions(
        paths,
        [("a", "Alpha", "success", "Indie folk"),
         ("b", "Beta", "success", "Electronic")],
    )
    embeddings.build_artist_summary_embeddings(paths=paths)
    model.calls.clear()

    _write_descriptions(
        paths,
        [("a", "Alpha", "success", "Indie folk"),
         ("b", "Beta", "success", "Electronic jazz"),
         ("c", "Gamma", "success", "Ambient")],
    )
    embeddings.build_artist_summary_embeddings(paths=paths)
    assert model.calls == [["Electronic jazz", "Ambient"]]
    model.calls.clear()

    _write_descriptions(paths, [("a", "Alpha", "success", "Indie folk")])
    embeddings.build_artist_summary_embeddings(paths=paths)
    assert not model.calls
    assert (
        pl.read_parquet(paths.artist_summary_embeddings)["artist_id"].to_list()
        == ["a"]
    )


def test_interrupted_run_resumes_from_checkpoint(setup):
    paths, model = setup
    _write_descriptions(
        paths,
        [(str(i), f"Artist {i}", "success", f"Music {i}") for i in range(5)],
    )
    model.fail_after = 1
    with pytest.raises(RuntimeError, match="Encoding stopped"):
        embeddings.build_artist_summary_embeddings(paths=paths)
    checkpoint = paths.artist_summary_embeddings.with_suffix(".parquet.partial")
    assert checkpoint.exists()
    assert pl.read_parquet(checkpoint).height == 2
    model.calls.clear()
    model.fail_after = None
    embeddings.build_artist_summary_embeddings(paths=paths)
    assert model.calls == [["Music 2", "Music 3"], ["Music 4"]]
    assert not checkpoint.exists()
    assert pl.read_parquet(paths.artist_summary_embeddings).height == 5


def test_changed_encoder_version_recomputes_all(setup, monkeypatch):
    paths, model = setup
    _write_descriptions(paths, [("a", "Alpha", "success", "Indie rock")])
    embeddings.build_artist_summary_embeddings(paths=paths)
    model.calls.clear()
    monkeypatch.setattr(embeddings, "version", lambda package: "new-version")
    embeddings.build_artist_summary_embeddings(paths=paths)
    assert model.calls == [["Indie rock"]]
    assert (
        pl.read_parquet(paths.artist_summary_embeddings).item(0, "encoder_version")
        == "new-version"
    )


def test_failed_encoding_preserves_previous_output(setup, monkeypatch):
    paths, model = setup
    _write_descriptions(paths, [("a", "Alpha", "success", "Jazz")])
    embeddings.build_artist_summary_embeddings(paths=paths)
    original = paths.artist_summary_embeddings.read_bytes()
    _write_descriptions(
        paths,
        [("a", "Alpha", "success", "Jazz"),
         ("b", "Beta", "success", "New artist")],
    )
    model.fail_after = 0
    with pytest.raises(RuntimeError, match="Encoding stopped"):
        embeddings.build_artist_summary_embeddings(paths=paths)
    assert paths.artist_summary_embeddings.read_bytes() == original


def test_invalid_and_duplicate_descriptions(setup):
    paths, _ = setup
    _write_descriptions(paths, [("a", "Alpha", "success", "music"),
                                ("a", "Alpha", "success", "more music")])
    with pytest.raises(ValueError, match="duplicate"):
        embeddings.build_artist_summary_embeddings(paths=paths)
    _write_descriptions(paths, [("a", "Alpha", "empty", None)])
    with pytest.raises(ValueError, match="No artists"):
        embeddings.build_artist_summary_embeddings(paths=paths)


def test_similarity_search_uses_shared_candidates(setup):
    paths, _ = setup
    _write_descriptions(
        paths,
        [("a", "Alpha", "success", "jazz"),
         ("b", "Beta", "success", "folk"),
         ("c", "Gamma", "success", "rock")],
    )
    embeddings.build_artist_summary_embeddings(paths=paths)
    frame = pl.read_parquet(paths.artist_summary_embeddings)
    artists = pl.DataFrame(
        {"artist_id": ["a", "b", "c"], "canonical_name": ["Alpha", "Beta", "Gamma"]}
    )
    matches = embeddings.find_similar_summary_artists(
        frame, artists, "a", candidate_ids={"a", "c"}
    )
    assert matches["artist_id"].to_list() == ["c"]
    assert 0 < matches.item(0, "similarity") <= 1
    with pytest.raises(ValueError, match="no summary embedding"):
        embeddings.find_similar_summary_artists(frame, artists, "missing")
    with pytest.raises(ValueError, match="positive"):
        embeddings.find_similar_summary_artists(frame, artists, "a", top_k=0)


def _write_tags(paths, rows):
    paths.curated_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "artist_id": pl.String,
            "canonical_name": pl.String,
            "status": pl.String,
            "tags": pl.List(
                pl.Struct({"name": pl.String, "weight": pl.Int64})
            ),
        },
        orient="row",
    ).write_parquet(paths.curated_artist_tags)


def test_tag_embeddings_use_clean_names_without_weights(setup):
    paths, model = setup
    _write_tags(
        paths,
        [
            ("b", "Beta", "success", [
                {"name": "hip-hop", "weight": 100},
                {"name": "Hip Hop", "weight": 20},
                {"name": "Jazz", "weight": 10},
            ]),
            ("a", "Alpha", "success", [
                {"name": "Jazz", "weight": 100},
                {"name": "HIP HOP", "weight": 15},
                {"name": "spotify", "weight": 50},
            ]),
            ("c", "Gamma", "empty", []),
            ("d", "Delta", "success", [{"name": "spotify", "weight": 30}]),
        ],
    )
    output = embeddings.build_artist_tag_embeddings(paths=paths)
    saved = pl.read_parquet(output)
    assert saved.schema == embeddings._TAG_SCHEMA
    assert saved["artist_id"].to_list() == ["a", "b"]
    assert model.calls == [["hip hop, jazz", "hip hop, jazz"]]
    assert not output.with_suffix(".parquet.partial").exists()

    metadata_path = paths.artist_tag_embeddings_metadata
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_sha256"] == embeddings._sha256(paths.curated_artist_tags)
    assert metadata["model_revision"] == embeddings.MODEL_REVISION
    assert metadata["embeddings_sha256"] == embeddings._sha256(output)
    embeddings.build_artist_tag_embeddings(paths=paths)
    assert len(model.calls) == 1


def test_tag_embeddings_incremental_updates_and_recovery(setup):
    paths, model = setup
    _write_tags(paths, [
        ("a", "Alpha", "success", [{"name": "Jazz", "weight": 100}]),
        ("b", "Beta", "success", [{"name": "Folk", "weight": 20}]),
    ])
    embeddings.build_artist_tag_embeddings(paths=paths)
    model.calls.clear()
    _write_tags(paths, [
        ("a", "Alpha Renamed", "success", [{"name": "JAZZ", "weight": 10}]),
        ("b", "Beta", "success", [{"name": "Folk rock", "weight": 20}]),
        ("c", "Gamma", "success", [{"name": "Ambient", "weight": 10}]),
        ("d", "Delta", "success", [{"name": "Indie pop", "weight": 10}]),
    ])
    model.fail_after = 1
    with pytest.raises(RuntimeError, match="Encoding stopped"):
        embeddings.build_artist_tag_embeddings(paths=paths)
    assert model.calls == [["folk rock", "ambient"]]
    assert paths.artist_tag_embeddings.with_suffix(".parquet.partial").exists()
    model.calls.clear()
    model.fail_after = None
    embeddings.build_artist_tag_embeddings(paths=paths)
    assert model.calls == [["indie pop"]]
    assert pl.read_parquet(paths.artist_tag_embeddings).height == 4


def test_tag_embedding_similarity_and_invalid_inputs(setup):
    paths, _ = setup
    _write_tags(paths, [
        ("a", "Alpha", "success", [{"name": "Jazz", "weight": 100}]),
        ("b", "Beta", "success", [{"name": "Folk", "weight": 50}]),
        ("c", "Gamma", "success", [{"name": "Rock", "weight": 50}]),
    ])
    embeddings.build_artist_tag_embeddings(paths=paths)
    frame = pl.read_parquet(paths.artist_tag_embeddings)
    names = pl.DataFrame({
        "artist_id": ["a", "b", "c"],
        "canonical_name": ["Alpha", "Beta", "Gamma"],
    })
    result = embeddings.find_similar_tag_artists(
        frame, names, "a", candidate_ids={"a", "c"}
    )
    assert result["artist_id"].to_list() == ["c"]
    assert 0 < result.item(0, "similarity") <= 1
    with pytest.raises(ValueError, match="no tag embedding"):
        embeddings.find_similar_tag_artists(frame, names, "missing")
    with pytest.raises(ValueError, match="Invalid tag embeddings"):
        embeddings.find_similar_tag_artists(
            frame.rename({"tags_sha256": "summary_sha256"}), names, "a"
        )
    with pytest.raises(ValueError, match="positive"):
        embeddings.find_similar_tag_artists(frame, names, "a", top_k=0)


def test_tag_embeddings_reject_bad_input(setup):
    paths, _ = setup
    _write_tags(paths, [("a", "Alpha", "success", [
        {"name": "Rock", "weight": -1}
    ])])
    with pytest.raises(ValueError, match="non-negative"):
        embeddings.build_artist_tag_embeddings(paths=paths)
    _write_tags(paths, [("a", "Alpha", "success", [
        {"name": "spotify", "weight": 100}
    ])])
    with pytest.raises(ValueError, match="No artists"):
        embeddings.build_artist_tag_embeddings(paths=paths)
