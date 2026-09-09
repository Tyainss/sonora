import datetime as dt

import polars as pl
import pytest

from sonora.breakout.evaluation.metrics import (
    daily_ranking_metrics,
    summarize_daily_metrics,
    summarize_event_opportunities,
    summarize_events,
    summarize_events_by_period,
)


def event_results(rows):
    return pl.DataFrame(rows)


def test_event_denominators_pool_raw_events_and_respect_opportunity():
    rows = []
    for period, count, detected_count in [("2024 H1", 4, 3), ("2024 H2", 20, 10)]:
        for index in range(count):
            detected = index < detected_count
            rows.append(
                {
                    "confirmation_period": period,
                    "is_comparable": True,
                    "is_scoreable": True,
                    "is_anticipatable": True,
                    "detected": detected,
                    "timing_stage": "Anticipatory" if detected else "Missed",
                    "days_to_breakout_start": 10 if detected else None,
                    "days_to_confirmation": 30 if detected else None,
                }
            )
    rows.append(
        {
            "confirmation_period": "2024 H2",
            "is_comparable": True,
            "is_scoreable": False,
            "is_anticipatable": False,
            "detected": False,
            "timing_stage": "Unscoreable",
            "days_to_breakout_start": None,
            "days_to_confirmation": None,
        }
    )

    overall = summarize_events(event_results(rows)).row(0, named=True)
    periods = summarize_events_by_period(event_results(rows))

    assert overall["event_coverage_at_3"] == pytest.approx(13 / 24)
    assert overall["zero_opportunity_events"] == 1
    assert overall["anticipatory_detection_rate"] == pytest.approx(13 / 24)
    assert overall["miss_rate"] == pytest.approx(11 / 24)
    assert overall["days_to_confirmation_median"] == 30
    assert periods["event_coverage_at_3"].to_list() == [0.75, 0.5]


def test_period_summary_keeps_periods_with_no_events():
    empty = pl.DataFrame(
        schema={
            "confirmation_period": pl.String,
            "is_comparable": pl.Boolean,
            "is_scoreable": pl.Boolean,
            "is_anticipatable": pl.Boolean,
            "detected": pl.Boolean,
            "timing_stage": pl.String,
            "days_to_breakout_start": pl.Int64,
            "days_to_confirmation": pl.Int64,
        }
    )

    summary = summarize_events_by_period(empty, periods=["2024 H1"])

    assert summary["confirmation_period"].to_list() == ["2024 H1"]
    assert summary["events"].to_list() == [0]


def test_stage_coverage_and_timing_quantiles_are_pooled():
    stages = ["Anticipatory", "Early", "Late", "Missed"]
    events = event_results(
        [
            {
                "confirmation_period": "2024 H1",
                "is_comparable": True,
                "is_scoreable": True,
                "is_anticipatable": True,
                "detected": stage != "Missed",
                "timing_stage": stage,
                "days_to_breakout_start": value if stage != "Missed" else None,
                "days_to_confirmation": value if stage != "Missed" else None,
            }
            for stage, value in zip(stages, [30, 20, 10, 0])
        ]
    )

    summary = summarize_events(events).row(0, named=True)

    assert summary["anticipatory_coverage"] == 0.25
    assert summary["coverage_through_first_period"] == 0.5
    assert summary["coverage_through_confirmation"] == 0.75
    assert summary["miss_rate"] == 0.25
    assert summary["days_to_confirmation_p25"] == 15
    assert summary["days_to_confirmation_median"] == 20
    assert summary["days_to_confirmation_p75"] == 25


def daily_inputs():
    rank_rows = []
    target_rows = []
    positives_by_day = [set(), {"a"}, {"a", "b"}, {"a", "b", "c"}, {"a", "b", "c", "d"}]
    orders = [
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "d", "e"],
        ["a", "c", "d", "b", "e"],
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "d", "e"],
    ]
    for offset, (positives, order) in enumerate(zip(positives_by_day, orders)):
        day = dt.date(2024, 1, 1) + dt.timedelta(days=offset)
        for rank, artist in enumerate(order, start=1):
            rank_rows.append(("u", artist, day, rank))
            target_rows.append(("u", artist, day, artist in positives))
    rankings = pl.DataFrame(
        rank_rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
            "rank": pl.UInt32,
        },
        orient="row",
    )
    targets = pl.DataFrame(
        target_rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
            "target_60d": pl.Boolean,
        },
        orient="row",
    )
    return rankings, targets


def test_top3_capture_and_ndcg_use_only_opportunity_days():
    rankings, targets = daily_inputs()
    daily = daily_ranking_metrics(rankings, targets)
    summary = summarize_daily_metrics(daily).row(0, named=True)

    assert daily["top3_capture"].to_list() == [None, 1.0, 0.5, 1.0, 1.0]
    assert daily["ndcg_at_3"][1] == 1.0
    assert daily["ndcg_at_3"][2] < 1.0
    assert daily["ndcg_at_3"][3] == pytest.approx(1.0)
    assert daily["ndcg_at_3"][4] == pytest.approx(1.0)
    assert summary["opportunity_days"] == 4
    assert summary["zero_positive_days"] == 1
    assert summary["mean_top3_capture"] == 0.875


def test_anticipatable_share_uses_all_historical_events():
    events = event_results(
        [
            {
                "confirmation_period": "2024 H1",
                "is_comparable": True,
                "is_scoreable": True,
                "is_anticipatable": True,
            },
            {
                "confirmation_period": "2024 H1",
                "is_comparable": False,
                "is_scoreable": True,
                "is_anticipatable": True,
            },
            {
                "confirmation_period": "2024 H1",
                "is_comparable": False,
                "is_scoreable": False,
                "is_anticipatable": False,
            },
        ]
    )

    summary = summarize_event_opportunities(events).row(0, named=True)

    assert summary["historical_events"] == 3
    assert summary["anticipatable_events"] == 2
    assert summary["anticipatable_share"] == pytest.approx(2 / 3)
