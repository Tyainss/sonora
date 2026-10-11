import datetime as dt

import polars as pl
import pytest

from sonora.breakout.evaluation.comparison import compare_event_detections


def event_results(detection_dates):
    count = len(detection_dates)
    return pl.DataFrame(
        {
            "user_id": ["u"] * count,
            "artist_id": [f"a{i}" for i in range(count)],
            "breakout_period_start": [dt.date(2022, 3, 1)] * count,
            "first_important_period_end": [dt.date(2022, 3, 7)] * count,
            "confirmation_date": [dt.date(2022, 3, 14)] * count,
            "detection_date": pl.Series(detection_dates, dtype=pl.Date),
            "timing_stage": [
                "Missed" if day is None else "Anticipatory" for day in detection_dates
            ],
            "is_comparable": [True] * count,
            "is_scoreable": [True] * count,
        }
    )


def test_pairing_keeps_gains_losses_and_timing_changes_separate():
    day = dt.date(2022, 2, 20)
    reference = event_results([day, day, day, None, day, None])
    candidate = event_results(
        [day - dt.timedelta(days=3), day + dt.timedelta(days=2), day, day, None, None]
    )
    paired = compare_event_detections(reference, candidate.reverse())
    assert paired["change"].to_list() == [
        "Earlier",
        "Later",
        "Same day",
        "Newly detected",
        "Lost detection",
        "Missed by both",
    ]
    assert paired["days_earlier"].to_list() == [3, -2, 0, None, None, None]


def test_events_without_a_full_comparison_window_or_opportunity_are_excluded():
    results = event_results([None] * 3).with_columns(
        pl.Series("is_comparable", [True, False, True]),
        pl.Series("is_scoreable", [True, True, False]),
    )
    paired = compare_event_detections(results, results)
    assert paired["artist_id"].to_list() == ["a0"]


def test_event_identity_and_dates_must_match_between_approaches():
    reference = event_results([None, None])
    with pytest.raises(ValueError, match="same breakout events"):
        compare_event_detections(reference, reference.head(1))
    changed = reference.with_columns(pl.col("confirmation_date") + pl.duration(days=1))
    with pytest.raises(ValueError, match="same breakout events"):
        compare_event_detections(reference, changed)


def test_duplicate_events_are_rejected():
    reference = event_results([None])
    with pytest.raises(ValueError, match="unique"):
        compare_event_detections(reference, pl.concat([reference, reference]))
