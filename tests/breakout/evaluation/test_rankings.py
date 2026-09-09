import datetime as dt

import polars as pl
import pytest

from sonora.breakout.evaluation.rankings import (
    combine_rankings,
    rank_scores,
)


def candidates(day, artists=("a", "b", "c")):
    return pl.DataFrame(
        {
            "user_id": ["u"] * len(artists),
            "artist_id": list(artists),
            "scoring_date": [day] * len(artists),
        }
    )


def test_daily_ranks_follow_score_order():
    expected = candidates(dt.date(2024, 1, 1))
    scores = expected.reverse().with_columns(pl.Series("score", [0.5, 0.8, 0.9]))

    ranked = rank_scores(scores, expected)

    assert ranked["artist_id"].to_list() == ["a", "b", "c"]
    assert ranked["rank"].to_list() == [1, 2, 3]


def test_tied_scores_require_the_ranking_method_to_define_an_order():
    expected = candidates(dt.date(2024, 1, 1))
    scores = expected.with_columns(pl.Series("score", [0.9, 0.9, 0.5]))

    with pytest.raises(ValueError, match="unique order"):
        rank_scores(scores, expected)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unexpected"])
def test_score_coverage_is_exact(kind):
    expected = candidates(dt.date(2024, 1, 1))
    scores = expected.with_columns(pl.Series("score", [3.0, 2.0, 1.0]))
    if kind == "duplicate":
        scores = pl.concat([scores, scores.head(1)])
    elif kind == "missing":
        scores = scores.head(2)
    else:
        scores = pl.concat(
            [
                scores,
                candidates(dt.date(2024, 1, 2), ("z",)).with_columns(
                    pl.lit(4.0).alias("score")
                ),
            ]
        )

    with pytest.raises(ValueError):
        rank_scores(scores, expected)


def test_ranked_folds_form_one_continuous_chronological_stream():
    june = candidates(dt.date(2024, 6, 30), ("a",)).with_columns(
        pl.lit(1.0).alias("score")
    )
    july = candidates(dt.date(2024, 7, 1), ("a",)).with_columns(
        pl.lit(2.0).alias("score")
    )

    combined = combine_rankings([rank_scores(july, july), rank_scores(june, june)])

    assert combined["scoring_date"].to_list() == [
        dt.date(2024, 6, 30),
        dt.date(2024, 7, 1),
    ]


def test_invalid_dates_scores_and_ranks_are_rejected():
    expected = candidates(dt.date(2024, 1, 1))
    scores = expected.with_columns(pl.Series("score", [3.0, 2.0, 1.0]))
    invalid_date = scores.with_columns(pl.col("scoring_date").cast(pl.Datetime))
    invalid_score = scores.with_columns(pl.lit(float("nan")).alias("score"))
    invalid_ranks = rank_scores(scores, expected).with_columns(
        pl.lit(1, dtype=pl.UInt32).alias("rank")
    )

    with pytest.raises(ValueError):
        rank_scores(invalid_date, invalid_date)
    with pytest.raises(ValueError):
        rank_scores(invalid_score, expected)
    with pytest.raises(ValueError, match="daily score order"):
        combine_rankings([invalid_ranks])
