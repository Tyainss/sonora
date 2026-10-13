"""Fold-specific classifiers and deterministic daily scoring."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)
from threadpoolctl import threadpool_limits

from sonora.breakout.evaluation.periods import EvaluationPeriod
from sonora.breakout.evaluation.rankings import rank_scores
from sonora.breakout.evaluation.splits import WalkForwardFold, build_walk_forward_folds
from sonora.breakout.features import CANDIDATE_KEYS, RECENT_SHARE_DAYS

MODEL_FEATURES = ("listening_share_7d", "previous_listening_count")
RECENCY_FEATURES = (*MODEL_FEATURES, "prior_listening_gap_days")


@dataclass(frozen=True)
class ModelFoldResult:
    """Training boundaries, fitted preprocessing/model, and one fold's scores."""

    fold: WalkForwardFold
    model: Pipeline
    scores: pl.DataFrame


def fit_logistic_regression(
    training: pl.DataFrame, *, interaction: bool = False, recency: bool = False
) -> Pipeline:
    """Fit additive, interaction, or recency regression with training-only scaling.

    Recency uses the gap in days, a training-median fill, and an always-present
    missing-history indicator. At least one observed training gap is required.
    """
    if interaction and recency:
        raise ValueError("Interaction and recency are separate model comparisons")
    inputs = _model_inputs(training, recency=recency)
    _validate_training(training, require_gap=recency)

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
    if recency:
        model.named_steps["transform"].transformers.extend(
            [
                ("prior_gap", SimpleImputer(strategy="median"), [2]),
                ("no_earlier_history", MissingIndicator(features="all"), [2]),
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


def fit_boosted_trees(training: pl.DataFrame) -> Pipeline:
    """Fit fixed, small histogram trees on three raw features with native NaNs.

    Automatic random validation and early stopping are disabled. Every fold
    fits all its resolved training rows for the same number of rounds.
    At least one observed training gap is required.
    """
    inputs = _model_inputs(training, recency=True)
    _validate_training(training, require_gap=True)
    model = Pipeline(
        [
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=100,
                    max_leaf_nodes=7,
                    max_depth=3,
                    min_samples_leaf=100,
                    l2_regularization=1.0,
                    categorical_features=None,
                    early_stopping=False,
                    random_state=0,
                    class_weight=None,
                ),
            ),
        ]
    )
    with threadpool_limits(limits=1, user_api="openmp"):
        model.fit(inputs, training["target_60d"].to_numpy())
    return model


def score_model(
    model: Pipeline,
    candidates: pl.DataFrame,
    baseline_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Order by raw decision score, resolving exact ties with baseline ordering.

    model_score retains the decision function. score is a unique negative daily
    rank for the existing evaluator; neither column is a breakout probability.
    baseline_scores must cover exactly the supplied candidates.
    """
    inputs = _model_inputs(
        candidates, recency=model_features(model) == RECENCY_FEATURES
    )
    baseline = rank_scores(baseline_scores, candidates).select(
        *CANDIDATE_KEYS, pl.col("score").alias("baseline_score")
    )
    if candidates.is_empty():
        return candidates.select(CANDIDATE_KEYS).with_columns(
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in ["model_score", "baseline_score", "score"]
        )
    with threadpool_limits(limits=1, user_api="openmp"):
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
    recency: bool = False,
) -> list[ModelFoldResult]:
    """Fit a fresh pipeline for each non-empty scoring fold using existing splits.

    Period selection belongs to the caller. Baseline scores cover exactly
    the folds' scoring rows.
    """
    return _run_walk_forward(
        data,
        periods,
        baseline_scores,
        partial(fit_logistic_regression, interaction=interaction, recency=recency),
    )


def run_boosted_tree_walk_forward(
    data: pl.DataFrame,
    periods: list[EvaluationPeriod],
    baseline_scores: pl.DataFrame,
) -> list[ModelFoldResult]:
    """Fit the same fixed tree configuration on each expanding training fold."""
    return _run_walk_forward(data, periods, baseline_scores, fit_boosted_trees)


def _run_walk_forward(
    data: pl.DataFrame,
    periods: list[EvaluationPeriod],
    baseline_scores: pl.DataFrame,
    fit_model: Callable[[pl.DataFrame], Pipeline],
) -> list[ModelFoldResult]:
    folds = [
        fold for fold in build_walk_forward_folds(data, periods) if fold.scoring_rows
    ]
    if not folds:
        raise ValueError("At least one scoring row is required")
    expected = pl.concat([fold.scoring.select(CANDIDATE_KEYS) for fold in folds])
    rank_scores(baseline_scores, expected)
    results = []
    for fold in folds:
        model = fit_model(fold.training)
        baseline = baseline_scores.join(
            fold.scoring.select(CANDIDATE_KEYS), on=CANDIDATE_KEYS, how="semi"
        )
        results.append(
            ModelFoldResult(
                fold=fold,
                model=model,
                scores=score_model(model, fold.scoring, baseline),
            )
        )
    return results


def model_features(model: Pipeline) -> tuple[str, ...]:
    """Return the fitted pipeline's raw input columns in training order."""
    if isinstance(model.named_steps["classifier"], HistGradientBoostingClassifier):
        return RECENCY_FEATURES
    return (
        RECENCY_FEATURES
        if "prior_gap" in model.named_steps["transform"].named_transformers_
        else MODEL_FEATURES
    )


def _validate_training(training: pl.DataFrame, *, require_gap: bool = False) -> None:
    required = {"target_60d", "has_full_horizon"}
    if required - set(training.columns):
        raise ValueError("Training requires target_60d and has_full_horizon")
    if training.is_empty():
        raise ValueError("Training requires non-empty data")
    if (
        training.schema["target_60d"] != pl.Boolean
        or training.schema["has_full_horizon"] != pl.Boolean
        or training["target_60d"].null_count()
        or training["has_full_horizon"].null_count()
        or not training["has_full_horizon"].all()
    ):
        raise ValueError("Training requires Boolean labels and fully observed horizons")
    if training["target_60d"].n_unique() != 2:
        raise ValueError("Training requires both target classes")
    if (
        require_gap
        and training["prior_listening_gap_days"].null_count() == training.height
    ):
        raise ValueError(
            "Recency-based models require at least one observed training gap"
        )


def _model_inputs(rows: pl.DataFrame, *, recency: bool = False) -> np.ndarray:
    features = RECENCY_FEATURES if recency else MODEL_FEATURES
    missing = set(features) - set(rows.columns)
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
    if recency:
        gap = "prior_listening_gap_days"
        if rows.schema[gap] != pl.Null and not rows.schema[gap].is_numeric():
            raise ValueError("Prior-listening gaps must be numeric or null")
        if rows.filter(
            (pl.col(gap).is_null() != (pl.col("previous_listening_count") == 0))
            | (
                pl.col(gap).is_not_null()
                & (
                    ~pl.col(gap).cast(pl.Float64).is_finite()
                    | (pl.col(gap) < RECENT_SHARE_DAYS + 1)
                )
            )
        ).height:
            raise ValueError(
                "Gaps must be finite and at least 8 days; only zero earlier counts have missing gaps"
            )
        return rows.select(pl.col(features).cast(pl.Float64)).to_numpy()
    return inputs
