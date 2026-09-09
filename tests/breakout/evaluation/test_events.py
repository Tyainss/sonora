import datetime as dt

import polars as pl
import pytest

from sonora.breakout.evaluation.events import evaluate_events


def event(
    *,
    start=dt.date(2024, 7, 1),
    first_end=dt.date(2024, 7, 7),
    confirmation=dt.date(2024, 7, 10),
):
    return pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["a"],
            "breakout_period_start": [start],
            "first_important_period_end": [first_end],
            "confirmation_date": [confirmation],
        }
    )


def rankings(rows):
    return pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
            "rank": pl.UInt32,
        },
        orient="row",
    )


def evaluate(rank_rows, **event_kwargs):
    return evaluate_events(
        event(**event_kwargs),
        rankings([("u", "a", date, rank) for date, rank in rank_rows]),
        oos_start=dt.date(2024, 5, 1),
        oos_end=dt.date(2024, 12, 31),
    ).row(0, named=True)


def test_event_window_opportunities_and_evidence_cross_fold_boundary():
    result = evaluate(
        [
            (dt.date(2024, 5, 11), 1),
            (dt.date(2024, 6, 30), 3),
            (dt.date(2024, 7, 2), 2),
            (dt.date(2024, 7, 10), 1),
        ]
    )

    assert result["evaluation_window_start"] == dt.date(2024, 5, 11)
    assert result["evaluation_window_end"] == dt.date(2024, 7, 9)
    assert result["valid_opportunities"] == 3
    assert result["pre_start_opportunities"] == 2
    assert result["top3_appearances"] == 3
    assert result["detection_date"] == dt.date(2024, 6, 30)
    assert result["timing_stage"] == "Anticipatory"
    assert result["days_to_breakout_start"] == 1
    assert result["days_to_confirmation"] == 10
    assert result["is_comparable"] is True


def test_zero_opportunities_are_unscoreable_not_missed():
    result = evaluate([])

    assert result["required_top3_appearances"] == 0
    assert result["is_scoreable"] is False
    assert result["detected"] is False
    assert result["timing_stage"] == "Unscoreable"


def test_one_opportunity_requires_one_appearance_and_two_require_two():
    one = evaluate([(dt.date(2024, 7, 1), 3)])
    two = evaluate([(dt.date(2024, 7, 1), 3), (dt.date(2024, 7, 2), 4)])

    assert one["required_top3_appearances"] == 1
    assert one["detection_date"] == dt.date(2024, 7, 1)
    assert two["required_top3_appearances"] == 2
    assert two["detected"] is False
    assert two["timing_stage"] == "Missed"


@pytest.mark.parametrize(
    ("dates", "stage", "days_to_start"),
    [
        ([dt.date(2024, 6, 29), dt.date(2024, 6, 30)], "Anticipatory", 1),
        ([dt.date(2024, 6, 30), dt.date(2024, 7, 1)], "Early", 0),
        ([dt.date(2024, 7, 6), dt.date(2024, 7, 7)], "Early", -6),
        ([dt.date(2024, 7, 7), dt.date(2024, 7, 8)], "Late", -7),
    ],
)
def test_detection_date_controls_exact_timing_boundaries(dates, stage, days_to_start):
    result = evaluate([(date, 1) for date in dates])

    assert result["detection_date"] == dates[1]
    assert result["timing_stage"] == stage
    assert result["days_to_breakout_start"] == days_to_start


def test_full_window_comparability_requires_both_oos_boundaries():
    result = evaluate_events(
        event(
            confirmation=dt.date(2024, 5, 20),
            start=dt.date(2024, 5, 5),
            first_end=dt.date(2024, 5, 11),
        ),
        rankings([]),
        oos_start=dt.date(2024, 4, 1),
        oos_end=dt.date(2024, 12, 31),
    ).row(0, named=True)

    assert result["evaluation_window_start"] == dt.date(2024, 3, 21)
    assert result["is_comparable"] is False
