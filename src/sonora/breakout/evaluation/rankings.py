"""Model-independent score validation and daily ranking."""

import polars as pl

RANKING_KEYS = ("user_id", "artist_id", "scoring_date")
SCORE_COLUMNS = (*RANKING_KEYS, "score")


def rank_scores(
    scores: pl.DataFrame, expected_candidates: pl.DataFrame
) -> pl.DataFrame:
    """Validate a complete score stream and assign ordinal ranks."""
    _validate_candidates(expected_candidates)
    _validate_scores(scores)
    expected = expected_candidates.select(RANKING_KEYS)
    actual = scores.select(RANKING_KEYS)
    missing = expected.join(actual, on=RANKING_KEYS, how="anti")
    unexpected = actual.join(expected, on=RANKING_KEYS, how="anti")
    if missing.height:
        raise ValueError(f"Missing scores for {missing.height} expected candidates")
    if unexpected.height:
        raise ValueError(f"Found {unexpected.height} unexpected score rows")
    ties = (
        scores.group_by("user_id", "scoring_date", "score")
        .len()
        .filter(pl.col("len") > 1)
    )
    if ties.height:
        raise ValueError(
            "Scores must define a unique order within each user and scoring date"
        )
    return (
        scores.select(SCORE_COLUMNS)
        .with_columns(pl.col("score").cast(pl.Float64))
        .sort(
            ["user_id", "scoring_date", "score"],
            descending=[False, False, True],
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.UInt32)
            .over("user_id", "scoring_date")
            .alias("rank")
        )
        .select(*RANKING_KEYS, "score", "rank")
    )


def combine_score_streams(streams: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine disjoint fold score outputs into one chronological OOS stream."""
    if not streams:
        raise ValueError("At least one score stream is required")
    for stream in streams:
        _validate_scores(stream)
    combined = pl.concat([stream.select(SCORE_COLUMNS) for stream in streams])
    _validate_scores(combined)
    return combined.sort("user_id", "scoring_date", "artist_id")


def combine_rankings(rankings: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine ranked folds while verifying their shared ranking contract."""
    if not rankings:
        raise ValueError("At least one ranking stream is required")
    for ranking in rankings:
        _validate_ranking(ranking)
    combined = pl.concat(
        [ranking.select(*RANKING_KEYS, "score") for ranking in rankings]
    )
    return rank_scores(combined, combined.select(RANKING_KEYS))


def _validate_candidates(candidates: pl.DataFrame) -> None:
    missing = set(RANKING_KEYS) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if candidates.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if any(candidates[column].null_count() for column in RANKING_KEYS):
        raise ValueError("Candidate keys cannot contain nulls")
    if candidates.select(RANKING_KEYS).n_unique() != candidates.height:
        raise ValueError("Expected candidates contain duplicate rows")


def _validate_scores(scores: pl.DataFrame) -> None:
    missing = set(SCORE_COLUMNS) - set(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if scores.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if not scores.schema["score"].is_numeric():
        raise ValueError("score must be numeric")
    if any(scores[column].null_count() for column in SCORE_COLUMNS):
        raise ValueError("Score rows cannot contain nulls")
    if scores.select(RANKING_KEYS).n_unique() != scores.height:
        raise ValueError("Duplicate score rows are not allowed")
    if not scores.select(pl.col("score").is_finite().all()).item():
        raise ValueError("Scores must be finite")


def _validate_ranking(ranking: pl.DataFrame) -> None:
    _validate_scores(ranking)
    if "rank" not in ranking.columns or ranking["rank"].null_count():
        raise ValueError("Ranked streams require non-null ranks")
    if not ranking.schema["rank"].is_integer():
        raise ValueError("rank must be an integer")
    expected = rank_scores(ranking.select(SCORE_COLUMNS), ranking.select(RANKING_KEYS))
    differences = expected.join(
        ranking.select(*RANKING_KEYS, "rank"),
        on=RANKING_KEYS,
        suffix="_provided",
        validate="1:1",
    ).filter(pl.col("rank") != pl.col("rank_provided"))
    if differences.height:
        raise ValueError("Ranked stream does not match its daily score order")
