import pytest

from sonora.similarity.fusion import reciprocal_rank_fusion


def test_rewards_artists_ranked_highly_by_both_approaches():
    result = reciprocal_rank_fusion(
        [["a", "b", "c"], ["c", "a", "d"]], top_k=4
    )
    assert result == ["a", "c", "b", "d"]


def test_uses_candidates_beyond_each_approachs_top_ten():
    tfidf = [f"tag-{i}" for i in range(10)] + ["shared"]
    bge = [f"dense-{i}" for i in range(10)] + ["shared"]
    result = reciprocal_rank_fusion([tfidf, bge])
    assert "shared" in result


def test_missing_sparse_matches_leave_dense_ranking_unchanged():
    assert reciprocal_rank_fusion([[], ["c", "a", "b"]]) == ["c", "a", "b"]


def test_ties_follow_first_seen_order():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]]) == ["a", "b"]


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="unique"):
        reciprocal_rank_fusion([["a", "a"]])
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], top_k=0)
