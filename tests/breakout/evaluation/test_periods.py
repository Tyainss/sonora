import datetime as dt

from sonora.breakout.evaluation.periods import (
    build_evaluation_periods,
    build_holdout_boundaries,
    confirmation_period,
    last_evaluable_scoring_date,
)


def test_explicit_evaluation_start_and_calendar_half_boundaries():
    periods = build_evaluation_periods(
        dt.date(2018, 9, 12), dt.date(2020, 10, 1)
    )

    assert [
        (p.period, p.scoring_start, p.scoring_end, p.is_partial)
        for p in periods[:2]
    ] == [
        ("2018 H2", dt.date(2018, 9, 12), dt.date(2018, 12, 31), True),
        ("2019 H1", dt.date(2019, 1, 1), dt.date(2019, 6, 30), False),
    ]


def test_complete_and_partial_final_periods_follow_observation_coverage():
    complete = build_evaluation_periods(
        dt.date(2022, 1, 1), dt.date(2022, 8, 29)
    )
    partial = build_evaluation_periods(
        dt.date(2022, 1, 1), dt.date(2022, 5, 1)
    )

    assert complete[-1].scoring_end == dt.date(2022, 6, 30)
    assert complete[-1].is_partial is False
    assert partial[-1].scoring_end == dt.date(2022, 3, 2)
    assert partial[-1].is_partial is True
    assert last_evaluable_scoring_date(dt.date(2026, 8, 10)) == dt.date(2026, 6, 11)


def test_evaluation_start_can_differ_between_users():
    coverage_end = dt.date(2025, 12, 31)

    user_a = build_evaluation_periods(dt.date(2022, 1, 1), coverage_end)
    user_b = build_evaluation_periods(dt.date(2024, 8, 15), coverage_end)

    assert user_a[0].scoring_start == dt.date(2022, 1, 1)
    assert user_b[0].scoring_start == dt.date(2024, 8, 15)
    assert user_b[0].is_partial is True


def test_confirmation_cohorts_and_holdout_lead_in_boundaries():
    assert confirmation_period(dt.date(2024, 6, 30)) == "2024 H1"
    assert confirmation_period(dt.date(2024, 7, 1)) == "2024 H2"

    holdout = build_holdout_boundaries(dt.date(2025, 1, 1), generation="v2")
    assert holdout.development_end == dt.date(2024, 11, 1)
    assert holdout.lead_in_start == dt.date(2024, 11, 2)
    assert holdout.lead_in_end == dt.date(2024, 12, 31)
    assert holdout.event_end == dt.date(2025, 6, 30)
    assert holdout.generation == "v2"
