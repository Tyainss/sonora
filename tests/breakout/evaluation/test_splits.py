import datetime as dt

import polars as pl
import pytest

from sonora.breakout.evaluation.periods import EvaluationPeriod
from sonora.breakout.evaluation.splits import build_walk_forward_folds


def test_training_waits_for_the_entire_horizon_and_expands():
    targets = pl.DataFrame(
        {
            "user_id": ["u"] * 5,
            "artist_id": ["early", "edge", "fast", "h2", "h2-edge"],
            "scoring_date": [
                dt.date(2021, 10, 1),
                dt.date(2021, 11, 2),
                dt.date(2021, 12, 15),
                dt.date(2022, 7, 2),
                dt.date(2022, 5, 2),
            ],
            "horizon_end_date": [
                dt.date(2021, 11, 30),
                dt.date(2022, 1, 1),
                dt.date(2022, 2, 13),
                dt.date(2022, 8, 31),
                dt.date(2022, 7, 1),
            ],
            "target_60d": [False, False, True, False, True],
            "has_full_horizon": [True] * 5,
        }
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False),
        EvaluationPeriod("2022 H2", dt.date(2022, 7, 1), dt.date(2022, 12, 31), False),
    ]

    first, second = build_walk_forward_folds(targets, periods)

    assert first.training["artist_id"].to_list() == ["early"]
    assert "fast" not in first.training["artist_id"]
    assert set(second.training["artist_id"]) == {"early", "edge", "fast"}
    assert "h2-edge" not in second.training["artist_id"]
    assert first.scoring["artist_id"].to_list() == ["h2-edge"]
    assert second.scoring["artist_id"].to_list() == ["h2"]
    assert first.training_rows < second.training_rows
    assert first.training["scoring_date"].max() < first.scoring_start


def test_fast_positive_does_not_make_an_unresolved_scoring_date_evaluable():
    targets = pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["fast-positive"],
            "scoring_date": [dt.date(2022, 6, 30)],
            "horizon_end_date": [dt.date(2022, 8, 29)],
            "target_60d": [True],
            "has_full_horizon": [False],
        }
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False)
    ]

    with pytest.raises(ValueError, match="unresolved"):
        build_walk_forward_folds(targets, periods)


def test_training_excludes_positive_rows_without_complete_future_observation():
    targets = pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["fast-positive"],
            "scoring_date": [dt.date(2021, 1, 1)],
            "horizon_end_date": [dt.date(2021, 3, 2)],
            "target_60d": [True],
            "has_full_horizon": [False],
        }
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 6, 30), False)
    ]

    fold = build_walk_forward_folds(targets, periods)[0]

    assert fold.training.is_empty()
