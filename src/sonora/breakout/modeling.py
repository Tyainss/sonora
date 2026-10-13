"""Fold-specific logistic regression and deterministic daily scoring."""

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)

from sonora.breakout.evaluation.periods import EvaluationPeriod
from sonora.breakout.evaluation.rankings import rank_scores
from sonora.breakout.evaluation.splits import WalkForwardFold, build_walk_forward_folds
from sonora.breakout.features import CANDIDATE_KEYS

MODEL_FEATURES = ("listening_share_7d", "previous_listening_count")


@dataclass(frozen=True)
class LogisticFoldResult:
    """Training boundaries, fitted preprocessing/model, and one fold's scores."""

    fold: WalkForwardFold
    model: Pipeline
    scores: pl.DataFrame


def fit_logistic_regression(
    training: pl.DataFrame, *, interaction: bool = False
) -> Pipeline:
    """Fit L2 regression, optionally adding share × log-count before scaling."""
    inputs = _model_inputs(training)
    required = {"target_60d", "has_full_horizon"}
    if required - set(training.columns):
        raise ValueError("Training requires target_60d and has_full_horizon")
    if training.is_empty():
        raise ValueError("Logistic regression requires non-empty training data")
    if (
        training.schema["target_60d"] != pl.Boolean
        or training.schema["has_full_horizon"] != pl.Boolean
        or training["target_60d"].null_count()
        or training["has_full_horizon"].null_count()
        or not training["has_full_horizon"].all()
    ):
        raise ValueError("Training requires Boolean labels and fully observed horizons")
    if training["target_60d"].n_unique() != 2:
        raise ValueError("Logistic regression requires both target classes")

    model = Pipeline(
        [
            (
                "transform",
                ColumnTransformer(
                    [
                        ("recent_share", "passthrough", [0]),
                        ("previous_count", FunctionTransformer(np.log1p), [1]),
                    ]
                ),
            ),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=1000,
                ),
            ),
        ]
    )
    if interaction:
        model.steps.insert(
            1,
            (
                "interaction",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(inputs, training["target_60d"].to_numpy())
    return model


def score_logistic_regression(
    model: Pipeline,
    candidates: pl.DataFrame,
    baseline_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Order by raw decision score, resolving exact ties with baseline ordering.

    model_score retains the decision function. score is a unique negative daily
    rank for the existing evaluator; neither column is a breakout probability.
    baseline_scores must cover exactly the supplied candidates.
    """
    inputs = _model_inputs(candidates)
    baseline = rank_scores(baseline_scores, candidates).select(
        *CANDIDATE_KEYS, pl.col("score").alias("baseline_score")
    )
    if candidates.is_empty():
        return candidates.select(CANDIDATE_KEYS).with_columns(
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in ["model_score", "baseline_score", "score"]
        )
    decisions = model.decision_function(inputs)
    if not np.isfinite(decisions).all():
        raise ValueError("Model decision scores must be finite")
    return (
        candidates.select(CANDIDATE_KEYS)
        .with_columns(pl.Series("model_score", decisions))
        .join(baseline, on=CANDIDATE_KEYS, validate="1:1")
        .sort(
            ["user_id", "scoring_date", "model_score", "baseline_score"],
            descending=[False, False, True, True],
        )
        .with_columns(
            (
                -pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over(
                    "user_id", "scoring_date"
                )
            )
            .cast(pl.Float64)
            .alias("score")
        )
    )


def run_logistic_walk_forward(
    data: pl.DataFrame,
    periods: list[EvaluationPeriod],
    baseline_scores: pl.DataFrame,
    *,
    interaction: bool = False,
) -> list[LogisticFoldResult]:
    """Fit a fresh pipeline for each non-empty scoring fold using existing splits.

    Period selection belongs to the caller. Baseline scores cover exactly
    the folds' scoring rows.
    """
    folds = [
        fold for fold in build_walk_forward_folds(data, periods) if fold.scoring_rows
    ]
    if not folds:
        raise ValueError("At least one scoring row is required")
    expected = pl.concat([fold.scoring.select(CANDIDATE_KEYS) for fold in folds])
    rank_scores(baseline_scores, expected)
    results = []
    for fold in folds:
        model = fit_logistic_regression(fold.training, interaction=interaction)
        baseline = baseline_scores.join(
            fold.scoring.select(CANDIDATE_KEYS), on=CANDIDATE_KEYS, how="semi"
        )
        results.append(
            LogisticFoldResult(
                fold=fold,
                model=model,
                scores=score_logistic_regression(model, fold.scoring, baseline),
            )
        )
    return results


def _model_inputs(rows: pl.DataFrame) -> np.ndarray:
    missing = set(MODEL_FEATURES) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if any(
        not rows.schema[column].is_numeric() or rows[column].null_count()
        for column in MODEL_FEATURES
    ):
        raise ValueError("Model features must be numeric and non-null")
    inputs = rows.select(pl.col(MODEL_FEATURES).cast(pl.Float64)).to_numpy()
    if not np.isfinite(inputs).all():
        raise ValueError("Model features must be finite")
    if rows.filter(
        ~pl.col("listening_share_7d").is_between(0.0, 1.0)
        | (pl.col("previous_listening_count") < 0)
    ).height:
        raise ValueError(
            "Listening share must be in [0, 1] and previous count non-negative"
        )
    return inputs
