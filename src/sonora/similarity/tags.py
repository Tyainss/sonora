import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

_RESULT_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "similarity": pl.Float64,
}

_NOISE_TAGS = {
    "all",
    "check out",
    "lastfmsc",
    "listen to later",
    "must listen",
    "my cup of tea",
    "my top songs",
    "recommended",
    "spotify",
    "top",
    "to::listen::to",
    ":to listen to in the future:",
}
_NOISE_PATTERNS = (
    re.compile(r"funk add to lidarr batch \d+"),
    re.compile(r"seen live(?: .+)?"),
    re.compile(r"upcoming album \d{4}"),
    re.compile(r"under \d+ listeners"),
    re.compile(r"fdc \d{4} (?:composers|participants)"),
)
_TAG_ALIASES = {
    "chillhop": "chill hop",
    "down tempo": "downtempo",
    "electropop": "electro pop",
    "neosoul": "neo soul",
    "synthpop": "synth pop",
}


def _clean_tag_name(name: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    normalized = re.sub(r"[\s_-]+", " ", normalized).strip()
    if not normalized:
        return None
    if normalized in _NOISE_TAGS or any(
        pattern.fullmatch(normalized) for pattern in _NOISE_PATTERNS
    ):
        return None
    normalized = re.sub(r"\bhiphop\b", "hip hop", normalized)
    normalized = re.sub(r"\blofi\b", "lo fi", normalized)
    normalized = re.sub(r"\brnb\b", "r&b", normalized)
    return _TAG_ALIASES.get(normalized, normalized)


def clean_artist_tags(tags: list[dict]) -> list[dict]:
    """Normalize tags, discard known noise, and keep the highest duplicate weight."""
    weights: dict[str, int] = {}
    for tag in tags:
        name = _clean_tag_name(tag["name"])
        if name is not None:
            weight = tag["weight"]
            weights[name] = max(weights.get(name, weight), weight)
    return [{"name": name, "weight": weight} for name, weight in weights.items()]


@dataclass(frozen=True)
class ArtistTagVectors:
    artists: pl.DataFrame
    binary: csr_matrix
    weighted: csr_matrix
    tfidf: csr_matrix


def build_artist_tag_vectors(
    artist_tags: pl.DataFrame, *, clean_tags: bool = False
) -> ArtistTagVectors:
    """Build binary, Last.fm-weighted, and TF-IDF vectors from artist tags."""
    required = {"artist_id", "canonical_name", "status", "tags"}
    if not required.issubset(artist_tags.columns):
        raise ValueError("Artist tags are missing required columns")
    if artist_tags["artist_id"].n_unique() != artist_tags.height:
        raise ValueError("Artist tags contain duplicate artist IDs")

    candidates = (
        artist_tags.filter(
            (pl.col("status") == "success") & (pl.col("tags").list.len() > 0)
        )
        .select("artist_id", "canonical_name", "tags")
        .sort("canonical_name", "artist_id")
    )
    features = []
    artist_rows = []
    for artist_id, canonical_name, tags in candidates.iter_rows():
        if any(tag["weight"] < 0 for tag in tags):
            raise ValueError("Tag weights must be non-negative")
        if clean_tags:
            tags = clean_artist_tags(tags)
        values = {tag["name"]: float(tag["weight"]) for tag in tags}
        if values:
            features.append(values)
            artist_rows.append((artist_id, canonical_name))

    if not features:
        raise ValueError("No artists with usable tags")
    artists = pl.DataFrame(
        artist_rows,
        schema={"artist_id": pl.String, "canonical_name": pl.String},
        orient="row",
    )
    weights = DictVectorizer(sparse=True).fit_transform(features).tocsr()
    binary = weights.copy()
    binary.data[:] = 1.0
    tfidf = TfidfTransformer().fit_transform(binary).tocsr()

    return ArtistTagVectors(
        artists=artists,
        binary=normalize(binary),
        weighted=normalize(weights),
        tfidf=tfidf,
    )


def find_similar_artists(
    vectors: ArtistTagVectors,
    artist_id: str,
    *,
    representation: str,
    top_k: int = 10,
) -> pl.DataFrame:
    """Rank other artists with shared tags by cosine similarity."""
    if representation not in {"binary", "weighted", "tfidf"}:
        raise ValueError("Representation must be 'binary', 'weighted', or 'tfidf'")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    ids = vectors.artists["artist_id"].to_list()
    if artist_id not in ids:
        raise ValueError(f"Artist has no usable tags: {artist_id}")

    query_index = ids.index(artist_id)
    matrix = getattr(vectors, representation)
    similarities = (matrix @ matrix[query_index].T).toarray().ravel()
    similarities[query_index] = 0.0
    names = vectors.artists["canonical_name"].to_list()
    order = np.argsort(-similarities, kind="stable")
    indices = [index for index in order if similarities[index] > 0][:top_k]

    return pl.DataFrame(
        {
            "artist_id": [ids[index] for index in indices],
            "canonical_name": [names[index] for index in indices],
            "similarity": [float(similarities[index]) for index in indices],
        },
        schema=_RESULT_SCHEMA,
    )
