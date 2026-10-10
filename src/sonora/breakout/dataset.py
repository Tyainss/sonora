"""Join daily breakout targets to point-in-time model features."""

import polars as pl

from sonora.breakout.features import CANDIDATE_KEYS, build_breakout_features


def build_modeling_data(
    events: pl.DataFrame | pl.LazyFrame,
    targets: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Append raw features while preserving every supplied target row and column.

    Pass the full listening history and caller-selected daily targets. Unknown
    labels and incomplete horizons remain intact; walk-forward splits select
    training rows. Features use only listening before each scoring date.
    """
    target_frame = targets.collect() if isinstance(targets, pl.LazyFrame) else targets
    required = {*CANDIDATE_KEYS, "target_60d", "horizon_end_date", "has_full_horizon"}
    missing = required - set(target_frame.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")

    features = build_breakout_features(events, target_frame.select(CANDIDATE_KEYS))
    feature_columns = [
        column for column in features.columns if column not in CANDIDATE_KEYS
    ]
    overlapping = set(feature_columns) & set(target_frame.columns)
    if overlapping:
        raise ValueError(
            f"Targets already contain feature columns: {sorted(overlapping)}"
        )

    result = target_frame.join(
        features,
        on=CANDIDATE_KEYS,
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    if result.height != target_frame.height or any(
        result[column].null_count() for column in feature_columns
    ):
        raise ValueError("Every target row must receive model features")
    return result
