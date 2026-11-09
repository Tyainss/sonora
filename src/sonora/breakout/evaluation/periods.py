"""Calendar boundaries for historical breakout evaluation."""

import calendar
import datetime as dt
from dataclasses import dataclass

from sonora.breakout.targets import DEFAULT_HORIZON_DAYS


@dataclass(frozen=True)
class EvaluationPeriod:
    """One calendar-aligned scoring period."""

    period: str
    scoring_start: dt.date
    scoring_end: dt.date
    is_partial: bool


@dataclass(frozen=True)
class HoldoutBoundaries:
    """Versioned unseen lead-in and event-period boundaries."""

    generation: str
    development_end: dt.date
    lead_in_start: dt.date
    lead_in_end: dt.date
    event_start: dt.date
    event_end: dt.date


def last_evaluable_scoring_date(
    observation_coverage_end: dt.date,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dt.date:
    """Return the last date with a completely observed future horizon."""
    _require_date(observation_coverage_end, "observation_coverage_end")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    return observation_coverage_end - dt.timedelta(days=horizon_days)


def confirmation_period(value: dt.date) -> str:
    """Return the calendar half containing a confirmation date."""
    _require_date(value, "confirmation_date")
    return f"{value.year} H{1 if value.month <= 6 else 2}"


def build_evaluation_periods(
    evaluation_start: dt.date,
    observation_coverage_end: dt.date,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[EvaluationPeriod]:
    """Build calendar half-years from a declared evaluation start."""
    _require_date(evaluation_start, "evaluation_start")
    final_date = last_evaluable_scoring_date(
        observation_coverage_end, horizon_days=horizon_days
    )
    periods = []
    start = evaluation_start
    while start <= final_date:
        natural_start = (
            dt.date(start.year, 1, 1)
            if start.month <= 6
            else dt.date(start.year, 7, 1)
        )
        natural_end = (
            dt.date(start.year, 6, 30)
            if start.month <= 6
            else dt.date(start.year, 12, 31)
        )
        scoring_end = min(natural_end, final_date)
        periods.append(
            EvaluationPeriod(
                period=confirmation_period(start),
                scoring_start=start,
                scoring_end=scoring_end,
                is_partial=start > natural_start or scoring_end < natural_end,
            )
        )
        start = natural_end + dt.timedelta(days=1)
    return periods


def build_holdout_boundaries(
    event_start: dt.date,
    *,
    generation: str,
    event_months: int = 6,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> HoldoutBoundaries:
    """Reserve an unseen scoring lead-in before a versioned event period."""
    _require_date(event_start, "event_start")
    if not generation.strip():
        raise ValueError("generation must not be empty")
    if event_months < 1 or horizon_days < 1:
        raise ValueError("event_months and horizon_days must be positive")
    lead_in_start = event_start - dt.timedelta(days=horizon_days)
    return HoldoutBoundaries(
        generation=generation,
        development_end=lead_in_start - dt.timedelta(days=1),
        lead_in_start=lead_in_start,
        lead_in_end=event_start - dt.timedelta(days=1),
        event_start=event_start,
        event_end=_add_months(event_start, event_months) - dt.timedelta(days=1),
    )


def _add_months(value: dt.date, months: int) -> dt.date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def _require_date(value: object, name: str) -> None:
    if not isinstance(value, dt.date) or isinstance(value, dt.datetime):
        raise TypeError(f"{name} must be a date")
