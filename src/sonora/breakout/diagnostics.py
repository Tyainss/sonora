"""Daily ranking and feature contributions for lost breakout detections."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from sonora.breakout.evaluation.comparison import compare_event_detections
from sonora.breakout.evaluation.evaluate import EvaluationResult
from sonora.breakout.evaluation.events import EVENT_KEYS, TOP_K
from sonora.breakout.features import CANDIDATE_KEYS
from sonora.breakout.modeling import (
    MODEL_FEATURES,
    RECENCY_FEATURES,
    ModelFoldResult,
    model_features,
)


@dataclass(frozen=True)
class LostDetectionDiagnostics:
    """Event counts, eligible-day ranks, and artists promoted on displaced days."""

    summary: pl.DataFrame
    daily_ranks: pl.DataFrame
    displacements: pl.DataFrame


def review_detection_changes(
    baseline: EvaluationResult,
    model: EvaluationResult,
    fold_results: list[ModelFoldResult],
) -> pl.DataFrame:
    """Review every comparable event with history at entry and both detection dates.

    Entry is the first eligible scoring day in the event's evaluation window.
    The history group describes listens before that day's recent seven-day window.
    It is a point-in-time description, not an artist age or discovery label.
    """
    paired = compare_event_detections(baseline.event_results, model.event_results)
    days = (
        paired.select(EVENT_KEYS)
        .join(
            model.event_results.select(
                *EVENT_KEYS, "evaluation_window_start", "evaluation_window_end"
            ),
            on=EVENT_KEYS,
            validate="1:1",
        )
        .join(model.rankings.select(CANDIDATE_KEYS), on=EVENT_KEYS, validate="1:m")
        .filter(
            pl.col("scoring_date").is_between(
                pl.col("evaluation_window_start"), pl.col("evaluation_window_end")
            )
        )
    )
    reviewed = paired.join(
        days.group_by(EVENT_KEYS).agg(
            pl.col("scoring_date").min().alias("first_eligible_date")
        ),
        on=EVENT_KEYS,
        validate="1:1",
    )
    for checkpoint, date_column in [
        ("entry", "first_eligible_date"),
        ("baseline_detection", "reference_detection_date"),
        ("model_detection", "model_detection_date"),
    ]:
        keys = reviewed.filter(pl.col(date_column).is_not_null()).select(
            *EVENT_KEYS, pl.col(date_column).alias("scoring_date")
        )
        features = _score_context(fold_results, keys).select(
            *EVENT_KEYS,
            pl.col("listening_share_7d").alias(f"{checkpoint}_share"),
            pl.col("previous_listening_count").alias(f"{checkpoint}_previous_count"),
            pl.col("prior_listening_gap_days").alias(f"{checkpoint}_prior_gap_days"),
        )
        reviewed = reviewed.join(features, on=EVENT_KEYS, how="left", validate="1:1")
    return reviewed.with_columns(
        pl.when(pl.col("entry_previous_count") == 0)
        .then(pl.lit("No earlier listens"))
        .otherwise(pl.lit("Earlier listens"))
        .alias("history_at_entry")
    ).sort("confirmation_date", *EVENT_KEYS)


def diagnose_lost_detections(
    baseline: EvaluationResult,
    model: EvaluationResult,
    fold_results: list[ModelFoldResult],
) -> LostDetectionDiagnostics:
    """Explain model misses that the baseline detected, using fitted fold models.

    Daily ranks cover only the event's eligible days inside its evaluation window.
    Displacements list artists entering the model Top 3 from outside the baseline
    Top 3 on days the lost artist leaves it. This is a set comparison, not a
    one-to-one assignment when several artists change places.
    Linear contributions are null for tree models; their rows retain raw
    features, ranks and decision scores without additive feature attribution.
    """
    lost = (
        compare_event_detections(baseline.event_results, model.event_results)
        .filter(pl.col("change") == "Lost detection")
        .select(*EVENT_KEYS, "reference_detection_date")
        .join(
            model.event_results.select(
                *EVENT_KEYS,
                "confirmation_date",
                "evaluation_window_start",
                "evaluation_window_end",
                "valid_opportunities",
                "required_top3_appearances",
            ),
            on=EVENT_KEYS,
            validate="1:1",
        )
    )
    ranked = baseline.rankings.select(
        *CANDIDATE_KEYS, pl.col("rank").alias("baseline_rank")
    ).join(
        model.rankings.select(*CANDIDATE_KEYS, pl.col("rank").alias("model_rank")),
        on=CANDIDATE_KEYS,
        how="left",
        validate="1:1",
    )
    if ranked.height != model.rankings.height or ranked["model_rank"].null_count():
        raise ValueError("Diagnostics require the same candidates in both rankings")
    daily = (
        lost.join(ranked, on=EVENT_KEYS, validate="1:m")
        .filter(
            pl.col("scoring_date").is_between(
                pl.col("evaluation_window_start"), pl.col("evaluation_window_end")
            )
        )
        .select(*CANDIDATE_KEYS, "baseline_rank", "model_rank")
    )
    displaced = daily.filter(
        (pl.col("baseline_rank") <= TOP_K) & (pl.col("model_rank") > TOP_K)
    )
    promoted = ranked.filter(
        (pl.col("baseline_rank") > TOP_K) & (pl.col("model_rank") <= TOP_K)
    ).rename(
        {
            "artist_id": "promoted_artist_id",
            "baseline_rank": "promoted_baseline_rank",
            "model_rank": "promoted_model_rank",
        }
    )
    displacements = displaced.join(promoted, on=["user_id", "scoring_date"])
    keys = pl.concat(
        [
            daily.select(CANDIDATE_KEYS),
            displacements.select(
                "user_id",
                pl.col("promoted_artist_id").alias("artist_id"),
                "scoring_date",
            ),
        ]
    ).unique()
    explanations = _score_context(fold_results, keys)
    daily = daily.join(explanations, on=CANDIDATE_KEYS, validate="1:1")
    displacements = (
        displacements.join(explanations, on=CANDIDATE_KEYS, validate="m:1")
        .join(
            explanations.rename(
                {
                    column: f"promoted_{column}"
                    for column in explanations.columns
                    if column not in ["user_id", "scoring_date"]
                }
            ),
            on=["user_id", "promoted_artist_id", "scoring_date"],
            validate="m:1",
        )
        .with_columns(
            (pl.col("promoted_model_score") - pl.col("model_score")).alias(
                "model_advantage"
            ),
            (
                pl.col("promoted_share_contribution") - pl.col("share_contribution")
            ).alias("share_advantage"),
            (
                pl.col("promoted_previous_count_contribution")
                - pl.col("previous_count_contribution")
            ).alias("previous_count_advantage"),
            (
                pl.col("promoted_interaction_contribution")
                - pl.col("interaction_contribution")
            ).alias("interaction_advantage"),
            (
                pl.col("promoted_prior_gap_contribution")
                - pl.col("prior_gap_contribution")
            ).alias("prior_gap_advantage"),
            (
                pl.col("promoted_no_history_contribution")
                - pl.col("no_history_contribution")
            ).alias("no_history_advantage"),
        )
    )
    summary = (
        lost.select(
            *EVENT_KEYS,
            "confirmation_date",
            "reference_detection_date",
            "valid_opportunities",
            "required_top3_appearances",
        )
        .join(
            daily.group_by(EVENT_KEYS).agg(
                (pl.col("baseline_rank") <= TOP_K).sum().alias("baseline_top3_days"),
                (pl.col("model_rank") <= TOP_K).sum().alias("model_top3_days"),
                pl.col("model_rank").min().alias("best_model_rank"),
            ),
            on=EVENT_KEYS,
            validate="1:1",
        )
        .sort("confirmation_date", *EVENT_KEYS)
    )
    return LostDetectionDiagnostics(
        summary=summary,
        daily_ranks=daily.sort(*EVENT_KEYS, "scoring_date"),
        displacements=displacements.sort(
            *EVENT_KEYS, "scoring_date", "promoted_model_rank"
        ),
    )


def _score_context(
    fold_results: list[ModelFoldResult], keys: pl.DataFrame
) -> pl.DataFrame:
    """Verify original model scores and decompose only logistic predictions."""
    columns = {
        **keys.schema,
        **{feature: pl.Float64 for feature in MODEL_FEATURES},
        "prior_listening_gap_days": pl.Float64,
        "period": pl.String,
        "share_contribution": pl.Float64,
        "previous_count_contribution": pl.Float64,
        "interaction_contribution": pl.Float64,
        "prior_gap_contribution": pl.Float64,
        "no_history_contribution": pl.Float64,
        "intercept": pl.Float64,
        "model_score": pl.Float64,
    }
    pieces = []
    for result in fold_results:
        rows = result.fold.scoring.join(keys, on=CANDIDATE_KEYS, how="semi")
        if rows.is_empty():
            continue
        features = model_features(result.model)
        has_recency = features == RECENCY_FEATURES
        rows = rows.select(
            *CANDIDATE_KEYS,
            pl.col(MODEL_FEATURES).cast(pl.Float64),
            pl.col("prior_listening_gap_days").cast(pl.Float64)
            if "prior_listening_gap_days" in rows.columns
            else pl.lit(None, dtype=pl.Float64).alias("prior_listening_gap_days"),
        )
        inputs = rows.select(features).to_numpy()
        classifier = result.model.named_steps["classifier"]
        if not isinstance(classifier, LogisticRegression):
            scored = rows.with_columns(
                pl.lit(result.fold.period).alias("period"),
                *[
                    pl.lit(None, dtype=pl.Float64).alias(column)
                    for column in columns
                    if column.endswith("contribution") or column == "intercept"
                ],
            ).join(
                result.scores.select(*CANDIDATE_KEYS, "model_score"),
                on=CANDIDATE_KEYS,
                how="left",
                validate="1:1",
            )
            with threadpool_limits(limits=1, user_api="openmp"):
                decisions = result.model.decision_function(inputs)
            if scored["model_score"].null_count() or not np.allclose(
                decisions,
                scored["model_score"].to_numpy(),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("Fitted model scores do not match the scored fold")
            pieces.append(scored.select(list(columns)))
            continue
        contributions = result.model[:-1].transform(inputs) * classifier.coef_[0]
        explained = rows.with_columns(
            pl.lit(result.fold.period).alias("period"),
            pl.Series("share_contribution", contributions[:, 0]),
            pl.Series("previous_count_contribution", contributions[:, 1]),
            pl.Series(
                "interaction_contribution",
                contributions[:, 2]
                if "interaction" in result.model.named_steps
                else np.zeros(rows.height),
            ),
            pl.Series(
                "prior_gap_contribution",
                contributions[:, 2] if has_recency else np.zeros(rows.height),
            ),
            pl.Series(
                "no_history_contribution",
                contributions[:, 3] if has_recency else np.zeros(rows.height),
            ),
            pl.lit(float(classifier.intercept_[0])).alias("intercept"),
        ).join(
            result.scores.select(*CANDIDATE_KEYS, "model_score"),
            on=CANDIDATE_KEYS,
            how="left",
            validate="1:1",
        )
        reconstructed = (
            explained.select(
                pl.col("share_contribution")
                + pl.col("previous_count_contribution")
                + pl.col("interaction_contribution")
                + pl.col("prior_gap_contribution")
                + pl.col("no_history_contribution")
                + pl.col("intercept")
            )
            .to_series()
            .to_numpy()
        )
        if explained["model_score"].null_count() or not np.allclose(
            reconstructed, explained["model_score"].to_numpy(), rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Fitted model contributions do not match the scored fold")
        pieces.append(explained.select(list(columns)))
    combined = pl.concat(pieces) if pieces else pl.DataFrame(schema=columns)
    if (
        combined.height != keys.height
        or combined.select(CANDIDATE_KEYS).n_unique() != keys.height
    ):
        raise ValueError("Every diagnostic candidate requires exactly one fitted fold")
    return combined
