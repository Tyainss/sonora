"""Combine artist similarity rankings without comparing their raw scores."""


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60, top_k: int = 10
) -> list[str]:
    """Rank artists by the sum of reciprocal ranks across input lists."""
    if k < 1 or top_k < 1:
        raise ValueError("k and top_k must be positive")

    scores: dict[str, float] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("Each ranking must contain unique artist IDs")
        for rank, artist_id in enumerate(ranking, start=1):
            scores[artist_id] = scores.get(artist_id, 0.0) + 1 / (k + rank)

    return sorted(scores, key=lambda artist_id: -scores[artist_id])[:top_k]
