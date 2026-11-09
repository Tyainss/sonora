"""High-level composition of the common breakout evaluator."""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from sonora.breakout.evaluation.events import evaluate_events
from sonora.breakout.evaluation.metrics import (
    daily_ranking_metrics,
    summarize_daily_metrics,
    summarize_event_opportunities,
    summarize_events,
    summarize_events_by_period,
)
from sonora.breakout.evaluation.periods import confirmation_period
from sonora.breakout.evaluation.rankings import rank_scores


@dataclass(frozen=True)
class EvaluationResult:
    """Standard outputs shared by every future ranking approach."""

    rankings: pl.DataFrame
    event_results: pl.DataFrame
    opportunity_summary: pl.DataFrame
    event_summary: pl.DataFrame
    period_summary: pl.DataFrame
    daily_metrics: pl.DataFrame
    daily_summary: pl.DataFrame


def evaluate_scores(
    scores: pl.DataFrame,
    expected_candidates: pl.DataFrame,
    events: pl.DataFrame,
    *,
    oos_start: dt.date,
    oos_end: dt.date,
) -> EvaluationResult:
    """Rank a continuous OOS score stream and produce standard evaluation views."""
    rankings = rank_scores(scores, expected_candidates)
    scoring_dates = rankings["scoring_date"]
    if not scoring_dates.is_empty() and (
        scoring_dates.min() < oos_start or scoring_dates.max() > oos_end
    ):
        raise ValueError("Expected candidates fall outside the declared OOS period")
    event_results = evaluate_events(
        events, rankings, oos_start=oos_start, oos_end=oos_end
    )
    daily_metrics = daily_ranking_metrics(rankings, expected_candidates)
    return EvaluationResult(
        rankings=rankings,
        event_results=event_results,
        opportunity_summary=summarize_event_opportunities(event_results),
        event_summary=summarize_events(event_results),
        period_summary=summarize_events_by_period(
            event_results, periods=_calendar_period_names(oos_start, oos_end)
        ),
        daily_metrics=daily_metrics,
        daily_summary=summarize_daily_metrics(daily_metrics),
    )


def _calendar_period_names(start: dt.date, end: dt.date) -> list[str]:
    current = dt.date(start.year, 1 if start.month <= 6 else 7, 1)
    names = []
    while current <= end:
        names.append(confirmation_period(current))
        current = (
            dt.date(current.year, 7, 1)
            if current.month == 1
            else dt.date(current.year + 1, 1, 1)
        )
    return names
