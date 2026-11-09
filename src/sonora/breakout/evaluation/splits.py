"""Point-in-time-correct walk-forward target splits."""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from sonora.breakout.evaluation.periods import EvaluationPeriod

TARGET_KEYS = ("user_id", "artist_id", "scoring_date")
REQUIRED_COLUMNS = (
    *TARGET_KEYS,
    "horizon_end_date",
    "target_60d",
    "has_full_horizon",
)


@dataclass(frozen=True)
class WalkForwardFold:
    """Inspectable training and scoring rows for one scoring period."""

    period: str
    scoring_start: dt.date
    scoring_end: dt.date
    is_partial: bool
    training_cutoff: dt.date
    training: pl.DataFrame
    scoring: pl.DataFrame

    @property
    def training_rows(self) -> int:
        return self.training.height

    @property
    def scoring_rows(self) -> int:
        return self.scoring.height


def build_walk_forward_folds(
    targets: pl.DataFrame, periods: list[EvaluationPeriod]
) -> list[WalkForwardFold]:
    """Select expanding resolved history and disjoint fold scoring rows."""
    _validate_targets(targets)
    _validate_periods(periods)
    folds = []
    for period in periods:
        training = targets.filter(
            (pl.col("horizon_end_date") < period.scoring_start)
            & pl.col("has_full_horizon")
            & pl.col("target_60d").is_not_null()
        )
        scoring = targets.filter(
            pl.col("scoring_date").is_between(period.scoring_start, period.scoring_end)
        )
        if (
            scoring["target_60d"].null_count()
            or not scoring.select(pl.col("has_full_horizon").all()).item()
        ):
            raise ValueError(
                f"Scoring period {period.period} contains unresolved target rows"
            )
        folds.append(
            WalkForwardFold(
                period=period.period,
                scoring_start=period.scoring_start,
                scoring_end=period.scoring_end,
                is_partial=period.is_partial,
                training_cutoff=period.scoring_start,
                training=training,
                scoring=scoring,
            )
        )
    return folds


def _validate_targets(targets: pl.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    if targets.schema["scoring_date"] != pl.Date:
        raise ValueError("scoring_date must use Date values")
    if targets.schema["horizon_end_date"] != pl.Date:
        raise ValueError("horizon_end_date must use Date values")
    if targets.select(TARGET_KEYS).n_unique() != targets.height:
        raise ValueError("Target candidates must be unique per user, artist, and day")
    if any(
        targets[column].null_count()
        for column in REQUIRED_COLUMNS
        if column != "target_60d"
    ):
        raise ValueError("Target identity and horizon columns cannot contain nulls")


def _validate_periods(periods: list[EvaluationPeriod]) -> None:
    previous_end = None
    for period in periods:
        if period.scoring_end < period.scoring_start:
            raise ValueError("Scoring period end cannot precede its start")
        if previous_end is not None and period.scoring_start <= previous_end:
            raise ValueError("Scoring periods must be chronological and disjoint")
        previous_end = period.scoring_end
