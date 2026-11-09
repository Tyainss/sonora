"""Compare detection dates for the same scoreable breakout events."""

import polars as pl

from sonora.breakout.evaluation.events import EVENT_COLUMNS, EVENT_KEYS


def compare_event_detections(
    reference: pl.DataFrame, candidate: pl.DataFrame
) -> pl.DataFrame:
    """Pair comparable scoreable events and separate timing changes from misses.

    days_earlier is reference date minus candidate date, available only when
    both detect the event. Positive values mean the candidate detects earlier.
    """
    columns = (*EVENT_COLUMNS, "detection_date", "timing_stage")
    populations = []
    for results in [reference, candidate]:
        rows = results.filter(pl.col("is_comparable") & pl.col("is_scoreable"))
        if rows.select(EVENT_KEYS).n_unique() != rows.height:
            raise ValueError("Event comparisons require unique user and artist keys")
        populations.append(rows.select(columns))
    left, right = populations
    if (
        left.join(right, on=EVENT_COLUMNS, how="anti").height
        or right.join(left, on=EVENT_COLUMNS, how="anti").height
    ):
        raise ValueError("Both approaches must evaluate the same breakout events")
    paired = left.rename(
        {
            "detection_date": "reference_detection_date",
            "timing_stage": "reference_stage",
        }
    ).join(
        right.select(
            *EVENT_KEYS,
            pl.col("detection_date").alias("model_detection_date"),
            pl.col("timing_stage").alias("model_stage"),
        ),
        on=EVENT_KEYS,
        validate="1:1",
    )
    return (
        paired.with_columns(
            (pl.col("reference_detection_date") - pl.col("model_detection_date"))
            .dt.total_days()
            .alias("days_earlier")
        )
        .with_columns(
            pl.when(
                pl.col("reference_detection_date").is_null()
                & pl.col("model_detection_date").is_null()
            )
            .then(pl.lit("Missed by both"))
            .when(pl.col("reference_detection_date").is_null())
            .then(pl.lit("Newly detected"))
            .when(pl.col("model_detection_date").is_null())
            .then(pl.lit("Lost detection"))
            .when(pl.col("days_earlier") > 0)
            .then(pl.lit("Earlier"))
            .when(pl.col("days_earlier") < 0)
            .then(pl.lit("Later"))
            .otherwise(pl.lit("Same day"))
            .alias("change")
        )
        .sort("confirmation_date", *EVENT_KEYS)
    )
