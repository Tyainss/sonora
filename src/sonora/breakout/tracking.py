"""Explicit local recording of a development model comparison."""

import datetime as dt
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow.sklearn
import polars as pl
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.models import infer_signature

from sonora.breakout.diagnostics import (
    diagnose_lost_detections,
    review_detection_changes,
)
from sonora.breakout.evaluation.comparison import compare_event_detections
from sonora.breakout.evaluation.evaluate import EvaluationResult
from sonora.breakout.evaluation.events import TOP_K
from sonora.breakout.features import CANDIDATE_KEYS, RECENT_SHARE_DAYS
from sonora.breakout.modeling import MODEL_FEATURES, LogisticFoldResult
from sonora.breakout.targets import DEFAULT_HORIZON_DAYS
from sonora.data.paths import DataPaths

EXPERIMENT_NAME = "breakout-development-comparison"
TRACKING_DIRECTORY = Path("artifacts/mlflow")
MODEL_PACKAGES = ("mlflow", "scikit-learn", "numpy", "scipy", "cloudpickle")


@dataclass(frozen=True)
class ComparisonContext:
    """Data and code snapshots captured when the comparison is computed."""

    documents: dict[str, str]
    git_commit: str
    git_dirty: bool
    holdout: dict
    model_requirements: list[str]


def capture_comparison_context(
    *,
    project_root: Path,
    paths: DataPaths,
    build_metadata: dict,
    holdout_path: Path,
) -> ComparisonContext:
    """Verify the target build and capture source files before model fitting."""
    root = project_root.resolve()
    data_paths = DataPaths(root / paths.data_dir)
    metadata_text = data_paths.breakout_build_metadata.read_text(encoding="utf-8")
    if json.loads(metadata_text) != build_metadata:
        raise ValueError("Build metadata changed; reload the comparison inputs")
    inputs = [
        (data_paths.curated_listening_events, build_metadata["source"]["sha256"]),
        (
            data_paths.daily_breakout_targets,
            build_metadata["outputs"][data_paths.daily_breakout_targets.name],
        ),
        (
            data_paths.breakout_events,
            build_metadata["outputs"][data_paths.breakout_events.name],
        ),
    ]
    hashes = {}
    for path, expected in inputs:
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{path.name} does not match the recorded target build")
        hashes[str(path.relative_to(root))] = actual

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout

    status = git("status", "--porcelain", "--untracked-files=normal")
    holdout_text = (root / holdout_path).read_text(encoding="utf-8")
    documents = {
        "provenance/build_metadata.json": metadata_text,
        "provenance/holdout.json": holdout_text,
        "provenance/input_hashes.json": json.dumps(hashes, indent=2),
        "code/git_status.txt": status,
        "code/working_tree.patch": git(
            "diff", "HEAD", "--", "src", "notebooks", "pyproject.toml", "uv.lock"
        ),
    }
    code_paths = [
        *sorted((root / "src").rglob("*.py")),
        root / "notebooks/marimo/11_breakout_model_comparison.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    for path in code_paths:
        documents[f"code/{path.relative_to(root).as_posix()}"] = path.read_text(
            encoding="utf-8"
        )
    packages = (*MODEL_PACKAGES, "polars", "pyarrow", "marimo")
    versions = {package: version(package) for package in packages}
    documents["provenance/package_versions.json"] = json.dumps(versions, indent=2)
    return ComparisonContext(
        documents=documents,
        git_commit=git("rev-parse", "HEAD").strip(),
        git_dirty=bool(status),
        holdout=json.loads(holdout_text),
        model_requirements=[
            f"{package}=={versions[package]}" for package in MODEL_PACKAGES
        ],
    )


def record_comparison(
    *,
    baseline: EvaluationResult,
    model: EvaluationResult,
    fold_results: list[LogisticFoldResult],
    context: ComparisonContext,
    project_root: Path,
    interaction_model: EvaluationResult | None = None,
    interaction_fold_results: list[LogisticFoldResult] | None = None,
) -> str:
    """Record the baseline, additive model, and optional interaction in one run.

    Recording disables MLflow usage telemetry and preserves the active run/URI.
    Undefined metrics stay null in result artifacts and are omitted from metrics.
    """
    if (interaction_model is None) != (interaction_fold_results is None):
        raise ValueError(
            "Interaction evaluation and fitted folds must be supplied together"
        )
    approaches = [("model", model, fold_results)]
    if interaction_model is not None:
        approaches.append(("interaction", interaction_model, interaction_fold_results))
    baseline_keys = baseline.rankings.select(CANDIDATE_KEYS).sort(CANDIDATE_KEYS)
    development_end = dt.date.fromisoformat(
        context.holdout["boundaries"]["development_end"]
    )
    paired_results = []
    for name, evaluation, results in approaches:
        if not results:
            raise ValueError("At least one fitted fold is required")
        keys = evaluation.rankings.select(CANDIDATE_KEYS).sort(CANDIDATE_KEYS)
        fold_keys = pl.concat(
            [result.scores.select(CANDIDATE_KEYS) for result in results]
        ).sort(CANDIDATE_KEYS)
        if not baseline_keys.equals(keys) or not keys.equals(fold_keys):
            raise ValueError(
                "Both evaluations and fitted folds must use the same candidates"
            )
        if keys["scoring_date"].max() > development_end:
            raise ValueError(
                "A development comparison cannot include reserved scoring dates"
            )
        paired_results.append(
            (
                name,
                compare_event_detections(
                    baseline.event_results, evaluation.event_results
                ),
            )
        )
    if interaction_model is not None:
        paired_results.append(
            (
                "interaction_vs_model",
                compare_event_detections(
                    model.event_results, interaction_model.event_results
                ),
            )
        )
    MLFLOW_DISABLE_TELEMETRY.set(True)
    storage = (project_root / TRACKING_DIRECTORY).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(
        tracking_uri=f"sqlite:///{(storage / 'mlflow.db').as_posix()}"
    )
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            EXPERIMENT_NAME, artifact_location=(storage / "artifacts").as_uri()
        )
    )
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": "listening-share-vs-logistic-regression"
            + ("-and-interaction" if interaction_model is not None else ""),
            "mlflow.source.git.commit": context.git_commit,
            "git.dirty": str(context.git_dirty).lower(),
            "evaluation.scope": "development",
            "holdout.generation": context.holdout["boundaries"]["generation"],
        },
    )
    run_id = run.info.run_id
    try:
        params = {
            "baseline": "recent_listening_share",
            "baseline.window_days": RECENT_SHARE_DAYS,
            "ranking": "decision_function; exact ties use baseline ordering",
            "horizon_days": DEFAULT_HORIZON_DAYS,
            "top_k": TOP_K,
            "required_appearances": "min(2, valid_opportunities)",
            "training": "expanding; horizon_end_date < scoring_start",
            "fold_count": len(fold_results),
            "scoring_start": fold_results[0].fold.scoring_start,
            "scoring_end": fold_results[-1].fold.scoring_end,
            **{
                f"holdout.{key}": value
                for key, value in context.holdout["boundaries"].items()
            },
        }
        for name, _, results in approaches:
            has_interaction = "interaction" in results[0].model.named_steps
            params.update(
                {
                    name: "logistic_regression",
                    f"{name}.features": ",".join(MODEL_FEATURES),
                    f"{name}.previous_count_transform": "log1p",
                    f"{name}.scaling": "StandardScaler fitted per training fold",
                    f"{name}.interaction": has_interaction,
                    f"{name}.terms": "share,log1p_count"
                    + (",share_x_log1p_count" if has_interaction else ""),
                    **{
                        f"{name}.{key}": value
                        for key, value in results[0]
                        .model.named_steps["classifier"]
                        .get_params()
                        .items()
                    },
                }
            )
            if any(
                ("interaction" in result.model.named_steps) != has_interaction
                for result in results
            ):
                raise ValueError(
                    "A model variant must use the same terms in every fold"
                )
        for key, value in params.items():
            client.log_param(run_id, key, str(value))
        with TemporaryDirectory(prefix="sonora-comparison-") as temporary:
            destination = Path(temporary)
            for name, contents in context.documents.items():
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            for name, evaluation in [
                ("baseline", baseline),
                *[(name, evaluation) for name, evaluation, _ in approaches],
            ]:
                directory = destination / name
                directory.mkdir()
                for table in [
                    "rankings",
                    "event_results",
                    "opportunity_summary",
                    "event_summary",
                    "period_summary",
                    "daily_metrics",
                    "daily_summary",
                ]:
                    getattr(evaluation, table).write_parquet(
                        directory / f"{table}.parquet"
                    )
                for summary in [evaluation.event_summary, evaluation.daily_summary]:
                    for key, value in summary.row(0, named=True).items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            client.log_metric(run_id, f"{name}.{key}", float(value))
            for name, paired in paired_results:
                prefix = "paired" if name == "model" else f"{name}.paired"
                file = (
                    "paired_events.parquet"
                    if name == "model"
                    else f"{name}_paired_events.parquet"
                )
                paired.write_parquet(destination / file)
                changes = dict(paired.group_by("change").len().iter_rows())
                for change in [
                    "Earlier",
                    "Later",
                    "Same day",
                    "Newly detected",
                    "Lost detection",
                    "Missed by both",
                ]:
                    client.log_metric(
                        run_id,
                        f"{prefix}.{change.lower().replace(' ', '_')}",
                        changes.get(change, 0),
                    )
                median = paired["days_earlier"].median()
                if median is not None:
                    client.log_metric(
                        run_id, f"{prefix}.median_days_earlier", float(median)
                    )
            for name, evaluation, results in approaches:
                base = destination if name == "model" else destination / name
                diagnostics = diagnose_lost_detections(baseline, evaluation, results)
                directory = base / "diagnostics/lost_detections"
                directory.mkdir(parents=True)
                for table in ["summary", "daily_ranks", "displacements"]:
                    getattr(diagnostics, table).write_parquet(
                        directory / f"{table}.parquet"
                    )
                review_detection_changes(baseline, evaluation, results).write_parquet(
                    base / "diagnostics/event_review.parquet"
                )
                folds = []
                for result in results:
                    fold = result.fold
                    inputs = (
                        fold.scoring.select(pl.col(MODEL_FEATURES).cast(pl.Float64))
                        .head(5)
                        .to_numpy()
                    )
                    folder = base / "folds" / fold.period.replace(" ", "_")
                    folder.mkdir(parents=True)
                    result.scores.write_parquet(folder / "scores.parquet")
                    mlflow.sklearn.save_model(
                        result.model,
                        folder / "model",
                        signature=infer_signature(inputs, result.model.predict(inputs)),
                        input_example=inputs,
                        pip_requirements=context.model_requirements,
                        metadata={
                            "feature_order": list(MODEL_FEATURES),
                            "ranking_method": "decision_function",
                            "interaction": "interaction" in result.model.named_steps,
                        },
                    )
                    folds.append(
                        {
                            "period": fold.period,
                            "scoring_start": fold.scoring_start,
                            "scoring_end": fold.scoring_end,
                            "training_cutoff": fold.training_cutoff,
                            "is_partial": fold.is_partial,
                            "training_rows": fold.training_rows,
                            "scoring_rows": fold.scoring_rows,
                            "positive_training_rows": fold.training["target_60d"].sum(),
                            "training_breakout_artists": fold.training.filter(
                                pl.col("target_60d")
                            )
                            .select("user_id", "artist_id")
                            .n_unique(),
                        }
                    )
                pl.DataFrame(folds).write_parquet(base / "folds.parquet")
            client.log_artifacts(run_id, str(destination))
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id, status="FINISHED")
    return run_id
