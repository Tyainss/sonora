import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl

from sonora.data.paths import DEFAULT_DATA_PATHS, DataPaths
from sonora.similarity.descriptions import clean_artist_description
from sonora.similarity.tags import clean_artist_tags

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 16

_SCHEMA = {
    "artist_id": pl.String,
    "summary_sha256": pl.String,
    "model_revision": pl.String,
    "encoder_version": pl.String,
    "embedding": pl.Array(pl.Float32, EMBEDDING_DIMENSIONS),
}
_TAG_SCHEMA = {
    "artist_id": pl.String,
    "tags_sha256": pl.String,
    "model_revision": pl.String,
    "encoder_version": pl.String,
    "embedding": pl.Array(pl.Float32, EMBEDDING_DIMENSIONS),
}
_RESULT_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "similarity": pl.Float64,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _summaries(paths: DataPaths) -> dict[str, tuple[str, str]]:
    descriptions = pl.read_parquet(paths.curated_artist_descriptions)
    required = {"artist_id", "canonical_name", "status", "summary"}
    if not required.issubset(descriptions.columns):
        raise ValueError("Artist descriptions are missing required columns")
    if descriptions["artist_id"].n_unique() != descriptions.height:
        raise ValueError("Artist descriptions contain duplicate artist IDs")

    rows = {}
    for artist_id, name, status, summary in descriptions.select(
        "artist_id", "canonical_name", "status", "summary"
    ).iter_rows():
        if status == "success":
            text = clean_artist_description(summary)
            if text:
                rows[artist_id] = (name, text)
    return dict(sorted(rows.items()))


def _tags(paths: DataPaths) -> dict[str, tuple[str, str]]:
    artist_tags = pl.read_parquet(paths.curated_artist_tags)
    required = {"artist_id", "canonical_name", "status", "tags"}
    if not required.issubset(artist_tags.columns):
        raise ValueError("Artist tags are missing required columns")
    if artist_tags["artist_id"].n_unique() != artist_tags.height:
        raise ValueError("Artist tags contain duplicate artist IDs")

    rows = {}
    for artist_id, name, status, tags in artist_tags.select(
        "artist_id", "canonical_name", "status", "tags"
    ).iter_rows():
        if status != "success" or not tags:
            continue
        if any(tag["weight"] < 0 for tag in tags):
            raise ValueError("Tag weights must be non-negative")
        names = sorted(tag["name"] for tag in clean_artist_tags(tags))
        if names:
            rows[artist_id] = (name, ", ".join(names))
    return dict(sorted(rows.items()))


def _read_cached(
    path: Path,
    texts: dict[str, tuple[str, str]],
    encoder_version: str,
    schema: dict[str, pl.DataType],
) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    frame = pl.read_parquet(path)
    if frame.schema != schema:
        raise ValueError(f"Unexpected embedding schema: {path}")
    if frame["artist_id"].n_unique() != frame.height:
        raise ValueError(f"Duplicate artist IDs in {path}")
    cached = {}
    for artist_id, text_hash, revision, saved_version, vector in frame.iter_rows():
        if (
            revision != MODEL_REVISION
            or saved_version != encoder_version
            or artist_id not in texts
        ):
            continue
        text = texts[artist_id][1]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_hash:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(array).all() or not np.isclose(
            np.linalg.norm(array), 1.0, atol=1e-3
        ):
            raise ValueError(f"Invalid embedding for artist {artist_id}")
        cached[artist_id] = vector
    return cached


def _write_checkpoint(
    checkpoint: Path,
    texts: dict[str, tuple[str, str]],
    vectors: dict[str, list[float]],
    encoder_version: str,
    schema: dict[str, pl.DataType],
) -> None:
    rows = [
        (
            artist_id,
            hashlib.sha256(texts[artist_id][1].encode("utf-8")).hexdigest(),
            MODEL_REVISION,
            encoder_version,
            vectors[artist_id],
        )
        for artist_id in sorted(vectors)
    ]
    frame = pl.DataFrame(rows, schema=schema, orient="row")
    temporary = checkpoint.with_suffix(".new.partial")
    frame.write_parquet(temporary)
    os.replace(temporary, checkpoint)


def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)


def _build_embeddings(
    *,
    source: Path,
    texts: dict[str, tuple[str, str]],
    output: Path,
    metadata_path: Path,
    schema: dict[str, pl.DataType],
    preprocessing: str,
    label: str,
) -> Path:
    source_sha256 = _sha256(source)
    if not texts:
        description = "summary descriptions" if label == "summary" else label
        raise ValueError(f"No artists have usable {description}")

    checkpoint = output.with_suffix(".parquet.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder_version = version("sentence-transformers")
    vectors = _read_cached(output, texts, encoder_version, schema)
    vectors.update(_read_cached(checkpoint, texts, encoder_version, schema))
    remaining = [artist_id for artist_id in texts if artist_id not in vectors]
    print(
        f"{label.capitalize()} embeddings: "
        f"{len(vectors)} reused, {len(remaining)} to encode"
    )

    truncated = 0
    if remaining:
        model = _load_model()
        if model.max_seq_length != 8192:
            raise ValueError("Unexpected BGE-M3 token limit")
        for offset in range(0, len(remaining), BATCH_SIZE):
            batch_ids = remaining[offset : offset + BATCH_SIZE]
            batch_texts = [texts[artist_id][1] for artist_id in batch_ids]
            token_ids = model.tokenizer(
                batch_texts, add_special_tokens=True, truncation=False
            )["input_ids"]
            lengths = [len(ids) for ids in token_ids]
            truncated += sum(length > model.max_seq_length for length in lengths)
            encoded = np.asarray(
                model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if encoded.shape != (len(batch_ids), EMBEDDING_DIMENSIONS):
                raise ValueError("Unexpected BGE-M3 embedding dimensions")
            if not np.isfinite(encoded).all() or not np.allclose(
                np.linalg.norm(encoded, axis=1), 1.0, atol=1e-3
            ):
                raise ValueError("Invalid BGE-M3 embeddings")
            vectors.update(zip(batch_ids, encoded.tolist(), strict=True))
            _write_checkpoint(checkpoint, texts, vectors, encoder_version, schema)
            print(f"{label.capitalize()} embeddings: {len(vectors)}/{len(texts)}")

    if truncated:
        print(f"Truncated {truncated} {label} texts at {model.max_seq_length:,} tokens")
    if _sha256(source) != source_sha256:
        raise RuntimeError(f"Artist {label} changed during embedding generation")

    _write_checkpoint(checkpoint, texts, vectors, encoder_version, schema)
    os.replace(checkpoint, output)
    metadata = {
        "source": str(source),
        "source_sha256": source_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sentence_transformers_version": encoder_version,
        "preprocessing": preprocessing,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "normalized": True,
        "batch_size": BATCH_SIZE,
        "artists": len(texts),
        "embeddings_sha256": _sha256(output),
    }
    temporary = metadata_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    return output


def build_artist_summary_embeddings(*, paths: DataPaths = DEFAULT_DATA_PATHS) -> Path:
    """Persist BGE-M3 summary embeddings, reusing unchanged artist texts."""
    return _build_embeddings(
        source=paths.curated_artist_descriptions,
        texts=_summaries(paths),
        output=paths.artist_summary_embeddings,
        metadata_path=paths.artist_summary_embeddings_metadata,
        schema=_SCHEMA,
        preprocessing="sonora.similarity.descriptions.clean_artist_description",
        label="summary",
    )


def build_artist_tag_embeddings(*, paths: DataPaths = DEFAULT_DATA_PATHS) -> Path:
    """Persist BGE-M3 vectors built from cleaned tag names, without tag weights."""
    return _build_embeddings(
        source=paths.curated_artist_tags,
        texts=_tags(paths),
        output=paths.artist_tag_embeddings,
        metadata_path=paths.artist_tag_embeddings_metadata,
        schema=_TAG_SCHEMA,
        preprocessing="sonora.similarity.tags.clean_artist_tags; sorted tag names",
        label="tags",
    )


def _find_similar_embeddings(
    embeddings: pl.DataFrame,
    artist_names: pl.DataFrame,
    artist_id: str,
    *,
    schema: dict[str, pl.DataType],
    label: str,
    top_k: int,
    candidate_ids: set[str] | None,
) -> pl.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if (
        embeddings.schema != schema
        or embeddings["artist_id"].n_unique() != embeddings.height
    ):
        raise ValueError(f"Invalid {label} embeddings")
    names = dict(artist_names.select("artist_id", "canonical_name").iter_rows())
    ids = embeddings["artist_id"].to_list()
    if artist_id not in ids:
        raise ValueError(f"Artist has no {label} embedding: {artist_id}")
    if not set(ids).issubset(names):
        raise ValueError("Missing canonical artist names")
    matrix = np.asarray(embeddings["embedding"].to_list(), dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != EMBEDDING_DIMENSIONS
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"Invalid {label} embeddings")
    query_index = ids.index(artist_id)
    similarities = matrix @ matrix[query_index]
    order = np.argsort(-similarities, kind="stable")
    indices = [
        index
        for index in order
        if ids[index] != artist_id
        and (candidate_ids is None or ids[index] in candidate_ids)
    ][:top_k]
    return pl.DataFrame(
        {
            "artist_id": [ids[index] for index in indices],
            "canonical_name": [names[ids[index]] for index in indices],
            "similarity": [float(similarities[index]) for index in indices],
        },
        schema=_RESULT_SCHEMA,
    )


def find_similar_summary_artists(
    embeddings: pl.DataFrame,
    artist_names: pl.DataFrame,
    artist_id: str,
    *,
    top_k: int = 10,
    candidate_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Rank artists using normalized summary embeddings."""
    return _find_similar_embeddings(
        embeddings,
        artist_names,
        artist_id,
        schema=_SCHEMA,
        label="summary",
        top_k=top_k,
        candidate_ids=candidate_ids,
    )


def find_similar_tag_artists(
    embeddings: pl.DataFrame,
    artist_names: pl.DataFrame,
    artist_id: str,
    *,
    top_k: int = 10,
    candidate_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Rank artists using normalized embeddings of cleaned tag names."""
    return _find_similar_embeddings(
        embeddings,
        artist_names,
        artist_id,
        schema=_TAG_SCHEMA,
        label="tag",
        top_k=top_k,
        candidate_ids=candidate_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BGE-M3 artist embeddings"
    )
    parser.add_argument("--input", choices=("summary", "tags"), default="summary")
    args = parser.parse_args()
    path = (
        build_artist_tag_embeddings()
        if args.input == "tags"
        else build_artist_summary_embeddings()
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
