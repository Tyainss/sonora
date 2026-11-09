"""Persist a modelling cycle's holdout boundaries across data refreshes."""

import calendar
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from sonora.breakout.evaluation.periods import (
    HoldoutBoundaries,
    build_holdout_boundaries,
    last_evaluable_scoring_date,
)


def load_or_create_holdout(
    path: Path,
    *,
    user_id: str,
    generation: str,
    observation_coverage_end: dt.date,
) -> HoldoutBoundaries:
    """Reuse a pinned cycle, or reserve the newest fully observed six-month period.

    The manifest belongs to one user and generation. Additional history keeps
    its boundaries fixed; a different cycle requires a different manifest.
    """
    latest_end = last_evaluable_scoring_date(observation_coverage_end)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            saved["schema_version"] != 1
            or saved["user_id"] != user_id
            or saved["boundaries"]["generation"] != generation
        ):
            raise ValueError(
                "Holdout manifest belongs to a different user or generation"
            )
        values = saved["boundaries"]
        holdout = build_holdout_boundaries(
            dt.date.fromisoformat(values["event_start"]), generation=generation
        )
        if values != {key: str(value) for key, value in asdict(holdout).items()}:
            raise ValueError("Holdout manifest contains inconsistent boundaries")
        if holdout.event_end > latest_end:
            raise ValueError("Observation coverage does not cover the pinned holdout")
        return holdout

    exclusive_end = latest_end + dt.timedelta(days=1)
    month_index = exclusive_end.year * 12 + exclusive_end.month - 1 - 6
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    start = dt.date(
        year, month, min(exclusive_end.day, calendar.monthrange(year, month)[1])
    )
    holdout = build_holdout_boundaries(start, generation=generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(
            {
                "schema_version": 1,
                "user_id": user_id,
                "observation_coverage_end": str(observation_coverage_end),
                "boundaries": {
                    key: str(value) for key, value in asdict(holdout).items()
                },
            },
            output,
            indent=2,
        )
        output.write("\n")
    return holdout
