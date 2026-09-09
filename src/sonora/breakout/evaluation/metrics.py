"""Pooled event summaries and opportunity-day ranking metrics."""

import math

import polars as pl

from sonora.breakout.evaluation.rankings import RANKING_KEYS

TOP_K = 3


def summarize_events(
    event_results: pl.DataFrame, *, comparable_only: bool = True
) -> pl.DataFrame:
    """Summarize raw event counts, stage coverage, and pooled timing."""
    population = (
        event_results.filter(pl.col("is_comparable"))
        if comparable_only
        else event_results
    )
    return pl.DataFrame([_event_summary_row(population)])


def summarize_event_opportunities(event_results: pl.DataFrame) -> pl.DataFrame:
    """Summarize historical opportunity and anticipatability across all events."""
    historical_events = event_results.height
    comparable = event_results.filter(pl.col("is_comparable"))
    anticipatable_events = event_results.filter(pl.col("is_anticipatable")).height
    return pl.DataFrame(
        [
            {
                "historical_events": historical_events,
                "full_window_comparable_events": comparable.height,
                "comparable_scoreable_events": comparable.filter(
                    pl.col("is_scoreable")
                ).height,
                "zero_opportunity_events": event_results.filter(
                    ~pl.col("is_scoreable")
                ).height,
                "anticipatable_events": anticipatable_events,
                "anticipatable_share": _ratio(
                    anticipatable_events, historical_events
                ),
            }
        ]
    )


def summarize_events_by_period(
    event_results: pl.DataFrame,
    *,
    comparable_only: bool = True,
    periods: list[str] | None = None,
) -> pl.DataFrame:
    """Report independent confirmation-date cohorts without averaging them."""
    population = (
        event_results.filter(pl.col("is_comparable"))
        if comparable_only
        else event_results
    )
    period_names = set(periods or []) | set(population["confirmation_period"])
    rows = []
    for period in sorted(period_names):
        row = _event_summary_row(
            population.filter(pl.col("confirmation_period") == period)
        )
        rows.append({"confirmation_period": period, **row})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def daily_ranking_metrics(
    rankings: pl.DataFrame,
    targets: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate Top-3 Capture and binary NDCG@3 on each user scoring day."""
    required = {*RANKING_KEYS, "rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    target_columns = {*RANKING_KEYS, "target_60d"}
    missing = target_columns - set(targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    expected = targets.select(RANKING_KEYS)
    actual = rankings.select(RANKING_KEYS)
    if expected.n_unique() != expected.height:
        raise ValueError("Target candidates contain duplicate rows")
    if actual.n_unique() != actual.height:
        raise ValueError("Ranking candidates contain duplicate rows")
    if expected.join(actual, on=RANKING_KEYS, how="anti").height:
        raise ValueError("Rankings are missing target candidates")
    if actual.join(expected, on=RANKING_KEYS, how="anti").height:
        raise ValueError("Rankings contain unexpected candidates")
    if targets["target_60d"].null_count():
        raise ValueError("Daily ranking metrics require resolved target labels")

    joined = rankings.join(
        targets.select(*RANKING_KEYS, "target_60d"),
        on=RANKING_KEYS,
        validate="1:1",
    ).with_columns(
        pl.when((pl.col("rank") <= TOP_K) & pl.col("target_60d"))
        .then(1 / (pl.col("rank") + 1).log(2))
        .otherwise(0.0)
        .alias("_dcg")
    )
    daily = joined.group_by("user_id", "scoring_date").agg(
        pl.col("target_60d").sum().cast(pl.UInt32).alias("positive_artists"),
        (pl.col("target_60d") & (pl.col("rank") <= TOP_K))
        .sum()
        .cast(pl.UInt32)
        .alias("top3_hits"),
        pl.col("_dcg").sum().alias("_dcg"),
    )
    ideal_by_count = [
        sum(1 / math.log2(rank + 1) for rank in range(1, count + 1))
        for count in range(TOP_K + 1)
    ]
    ideal = pl.lit(ideal_by_count[-1])
    for count in range(TOP_K - 1, -1, -1):
        ideal = (
            pl.when(pl.col("positive_artists") == count)
            .then(pl.lit(ideal_by_count[count]))
            .otherwise(ideal)
        )
    return (
        daily.with_columns(
            (pl.col("positive_artists") > 0).alias("is_opportunity_day"),
            pl.when(pl.col("positive_artists") > 0)
            .then(
                pl.col("top3_hits")
                / pl.min_horizontal(pl.lit(TOP_K), "positive_artists")
            )
            .otherwise(None)
            .alias("top3_capture"),
            ideal.alias("_idcg"),
        )
        .with_columns(
            pl.when(pl.col("is_opportunity_day"))
            .then(pl.col("_dcg") / pl.col("_idcg"))
            .otherwise(None)
            .alias("ndcg_at_3")
        )
        .drop("_dcg", "_idcg")
        .sort("user_id", "scoring_date")
    )


def summarize_daily_metrics(daily_metrics: pl.DataFrame) -> pl.DataFrame:
    """Summarize opportunity days while retaining zero-positive-day context."""
    scoring_days = daily_metrics.height
    opportunity_days = daily_metrics.filter(pl.col("is_opportunity_day")).height
    zero_positive_days = scoring_days - opportunity_days
    return pl.DataFrame(
        [
            {
                "scoring_days": scoring_days,
                "opportunity_days": opportunity_days,
                "zero_positive_days": zero_positive_days,
                "zero_positive_day_frequency": _ratio(zero_positive_days, scoring_days),
                "mean_top3_capture": daily_metrics["top3_capture"].mean(),
                "mean_ndcg_at_3": daily_metrics["ndcg_at_3"].mean(),
            }
        ]
    )


def _event_summary_row(events: pl.DataFrame) -> dict:
    scoreable = events.filter(pl.col("is_scoreable"))
    detected = scoreable.filter(pl.col("detected"))
    anticipatable = events.filter(pl.col("is_anticipatable"))
    anticipatory = events.filter(pl.col("timing_stage") == "Anticipatory")
    early_or_better = events.filter(
        pl.col("timing_stage").is_in(["Anticipatory", "Early"])
    )
    missed = scoreable.filter(pl.col("timing_stage") == "Missed")
    row = {
        "events": events.height,
        "scoreable_events": scoreable.height,
        "zero_opportunity_events": events.height - scoreable.height,
        "detected_events": detected.height,
        "anticipatable_events": anticipatable.height,
        "anticipatory_detections": anticipatory.height,
        "event_coverage_at_3": _ratio(detected.height, scoreable.height),
        "anticipatory_detection_rate": _ratio(
            anticipatory.height, anticipatable.height
        ),
        "anticipatory_coverage": _ratio(anticipatory.height, scoreable.height),
        "coverage_through_first_period": _ratio(
            early_or_better.height, scoreable.height
        ),
        "coverage_through_confirmation": _ratio(detected.height, scoreable.height),
        "miss_rate": _ratio(missed.height, scoreable.height),
    }
    for column in ["days_to_breakout_start", "days_to_confirmation"]:
        values = detected[column]
        row[f"{column}_p25"] = _quantile(values, 0.25)
        row[f"{column}_median"] = _quantile(values, 0.5)
        row[f"{column}_p75"] = _quantile(values, 0.75)
    return row


def _quantile(values: pl.Series, quantile: float) -> float | None:
    if values.is_empty():
        return None
    return values.quantile(quantile, interpolation="linear")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
