from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

_RESULT_SCHEMA = {
    "artist_id": pl.String,
    "canonical_name": pl.String,
    "similarity": pl.Float64,
}


@dataclass(frozen=True)
class ArtistTagVectors:
    artists: pl.DataFrame
    binary: csr_matrix
    weighted: csr_matrix


def build_artist_tag_vectors(artist_tags: pl.DataFrame) -> ArtistTagVectors:
    """Build binary and Last.fm-weighted vectors using the original tag names."""
    required = {"artist_id", "canonical_name", "status", "tags"}
    if not required.issubset(artist_tags.columns):
        raise ValueError("Artist tags are missing required columns")
    if artist_tags["artist_id"].n_unique() != artist_tags.height:
        raise ValueError("Artist tags contain duplicate artist IDs")

    artists = (
        artist_tags.filter(
            (pl.col("status") == "success") & (pl.col("tags").list.len() > 0)
        )
        .select("artist_id", "canonical_name", "tags")
        .sort("canonical_name", "artist_id")
    )
    if artists.is_empty():
        raise ValueError("No artists with usable tags")

    features = []
    for tags in artists["tags"].to_list():
        values = {tag["name"]: float(tag["weight"]) for tag in tags}
        if any(weight < 0 for weight in values.values()):
            raise ValueError("Tag weights must be non-negative")
        features.append(values)

    weights = DictVectorizer(sparse=True).fit_transform(features).tocsr()
    binary = weights.copy()
    binary.data[:] = 1.0

    return ArtistTagVectors(
        artists=artists.drop("tags"),
        binary=normalize(binary),
        weighted=normalize(weights),
    )


def find_similar_artists(
    vectors: ArtistTagVectors,
    artist_id: str,
    *,
    representation: str,
    top_k: int = 10,
) -> pl.DataFrame:
    """Rank other artists with shared tags by cosine similarity."""
    if representation not in {"binary", "weighted"}:
        raise ValueError("Representation must be 'binary' or 'weighted'")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    ids = vectors.artists["artist_id"].to_list()
    if artist_id not in ids:
        raise ValueError(f"Artist has no usable tags: {artist_id}")

    query_index = ids.index(artist_id)
    matrix = vectors.binary if representation == "binary" else vectors.weighted
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
