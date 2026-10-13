import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient

from sonora.breakout.evaluation import EvaluationPeriod, evaluate_scores
from sonora.breakout.evaluation.holdout import load_or_create_holdout
from sonora.breakout.modeling import MODEL_FEATURES, run_logistic_walk_forward
from sonora.breakout.timing import TimingAudit
from sonora.breakout.tracking import capture_comparison_context, record_comparison
from sonora.data.paths import DataPaths


@pytest.fixture(autouse=True)
def local_tracking_only(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")


@pytest.fixture
def comparison(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    paths = DataPaths(root / "data")
    paths.curated_dir.mkdir(parents=True)
    paths.breakout_dir.mkdir(parents=True)
    source = root / "src/sonora/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    notebook = root / "notebooks/marimo/11_breakout_model_comparison.py"
    notebook.parent.mkdir(parents=True)
    notebook.write_text("# comparison\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'sonora'\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".gitignore").write_text("data/\nartifacts/mlflow/\n")
    for args in [
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "Initial",
        ],
    ]:
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    source.write_text("VALUE = 2\n")
    (root / "src/sonora/untracked.py").write_text("NEW = True\n")
    rows = []
    for index in range(8):
        rows.append(
            (
                "u",
                f"a{index}",
                dt.date(2021, 10, 1),
                0.05 * index,
                index * 3,
                index >= 4,
            )
        )
    for day in [
        dt.date(2022, 1, 1),
        dt.date(2022, 1, 2),
        dt.date(2022, 7, 1),
        dt.date(2022, 7, 2),
    ]:
        for index in range(4):
            rows.append(("u", f"a{index}", day, 0.1 * index, index * 7, index == 0))
    data = pl.DataFrame(
        rows,
        schema={
            "user_id": pl.String,
            "artist_id": pl.String,
            "scoring_date": pl.Date,
            "listening_share_7d": pl.Float64,
            "previous_listening_count": pl.Int64,
            "target_60d": pl.Boolean,
        },
        orient="row",
    ).with_columns(
        (pl.col("scoring_date") + dt.timedelta(days=60)).alias("horizon_end_date"),
        pl.lit(True).alias("has_full_horizon"),
    )
    data.write_parquet(paths.daily_breakout_targets)
    events = pl.DataFrame(
        {
            "user_id": ["u"],
            "artist_id": ["a0"],
            "breakout_period_start": [dt.date(2022, 2, 1)],
            "first_important_period_end": [dt.date(2022, 2, 7)],
            "confirmation_date": [dt.date(2022, 3, 2)],
        }
    )
    events.write_parquet(paths.breakout_events)
    pl.DataFrame(
        {"listened_at": [dt.datetime(2021, 1, 1, tzinfo=dt.UTC)]}
    ).write_parquet(paths.curated_listening_events)
    metadata = {
        "source": {
            "sha256": hashlib.sha256(
                paths.curated_listening_events.read_bytes()
            ).hexdigest()
        },
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [paths.daily_breakout_targets, paths.breakout_events]
        },
    }
    paths.breakout_build_metadata.write_text(json.dumps(metadata))
    holdout_path = root / "artifacts/breakout/first_ml_holdout.json"
    load_or_create_holdout(
        holdout_path,
        user_id="u",
        generation="first_ml",
        observation_coverage_end=dt.date(2023, 8, 10),
    )
    periods = [
        EvaluationPeriod("2022 H1", dt.date(2022, 1, 1), dt.date(2022, 1, 2), True),
        EvaluationPeriod("2022 H2", dt.date(2022, 7, 1), dt.date(2022, 7, 2), True),
    ]
    candidates = data.filter(pl.col("scoring_date") >= dt.date(2022, 1, 1))
    scores = candidates.select("user_id", "artist_id", "scoring_date").with_columns(
        pl.col("artist_id").str.slice(1).cast(pl.Float64).alias("score")
    )
    folds = run_logistic_walk_forward(data, periods, scores)
    evaluations = [
        evaluate_scores(
            stream,
            candidates,
            events,
            oos_start=periods[0].scoring_start,
            oos_end=periods[-1].scoring_end,
        )
        for stream in [scores, pl.concat([result.scores for result in folds])]
    ]
    context = capture_comparison_context(
        project_root=root,
        paths=paths,
        build_metadata=metadata,
        holdout_path=holdout_path,
    )
    return root, paths, metadata, holdout_path, context, folds, evaluations


def test_recorded_models_and_results_round_trip_without_changing_global_tracking(
    comparison, tmp_path
):
    root, _, _, _, context, folds, (baseline, model) = comparison
    uri = f"sqlite:///{(root / 'artifacts/mlflow/mlflow.db').as_posix()}"
    original_uri = mlflow.get_tracking_uri()
    unrelated_uri = f"sqlite:///{(tmp_path / 'unrelated.db').as_posix()}"
    mlflow.set_tracking_uri(unrelated_uri)
    unrelated_experiment = MlflowClient(tracking_uri=unrelated_uri).create_experiment(
        "unrelated", artifact_location=(tmp_path / "unrelated-artifacts").as_uri()
    )
    try:
        with mlflow.start_run(experiment_id=unrelated_experiment) as unrelated:
            run_id = record_comparison(
                baseline=baseline,
                model=model,
                fold_results=folds,
                context=context,
                project_root=root,
            )
            assert mlflow.active_run().info.run_id == unrelated.info.run_id
            assert mlflow.get_tracking_uri() == unrelated_uri
        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["git.dirty"] == "true"
        assert run.data.tags["mlflow.source.git.commit"] == context.git_commit
        assert run.data.params["model.C"] == "1.0"
        assert run.data.params["fold_count"] == "2"
        assert (
            run.data.metrics["baseline.detected_events"]
            == baseline.event_summary["detected_events"].item()
        )
        assert "baseline.days_to_breakout_start_median" not in run.data.metrics
        assert (
            sum(
                value
                for key, value in run.data.metrics.items()
                if key.startswith("paired.") and key != "paired.median_days_earlier"
            )
            == 1
        )
        artifact = client.download_artifacts(run_id, "baseline/event_results.parquet")
        assert pl.read_parquet(artifact).equals(baseline.event_results)
        artifact = client.download_artifacts(run_id, "code/src/sonora/untracked.py")
        assert Path(artifact).read_text() == "NEW = True\n"
        for result in folds:
            folder = "folds/" + result.fold.period.replace(" ", "_")
            local = client.download_artifacts(run_id, f"{folder}/model")
            restored = mlflow.sklearn.load_model(local)
            inputs = result.fold.scoring.select(
                pl.col(MODEL_FEATURES).cast(pl.Float64)
            ).to_numpy()
            np.testing.assert_array_equal(
                restored.decision_function(inputs),
                result.model.decision_function(inputs),
            )
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(local).predict(inputs),
                result.model.predict(inputs),
            )
            assert (
                restored.named_steps["scale"].n_samples_seen_
                == result.fold.training_rows
            )
    finally:
        mlflow.set_tracking_uri(original_uri)


@pytest.mark.parametrize(
    "filename",
    [
        "listening_events.parquet",
        "daily_targets.parquet",
        "first_breakout_events.parquet",
    ],
)
def test_changed_inputs_are_rejected_before_fitting(comparison, filename):
    root, paths, metadata, holdout_path, _, _, _ = comparison
    path = next(
        path
        for path in [
            paths.curated_listening_events,
            paths.daily_breakout_targets,
            paths.breakout_events,
        ]
        if path.name == filename
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="does not match"):
        capture_comparison_context(
            project_root=root,
            paths=paths,
            build_metadata=metadata,
            holdout_path=holdout_path,
        )


def test_metadata_changes_require_reloading_inputs(comparison):
    root, paths, metadata, holdout_path, _, _, _ = comparison
    paths.breakout_build_metadata.write_text(json.dumps({**metadata, "updated": True}))
    with pytest.raises(ValueError, match="reload"):
        capture_comparison_context(
            project_root=root,
            paths=paths,
            build_metadata=metadata,
            holdout_path=holdout_path,
        )


def test_capture_preserves_code_as_it_was_when_comparison_was_computed(comparison):
    root, _, _, _, context, _, _ = comparison
    (root / "src/sonora/example.py").write_text("VALUE = 3\n")
    assert context.documents["code/src/sonora/example.py"] == "VALUE = 2\n"
    assert "+VALUE = 2" in context.documents["code/working_tree.patch"]


def test_mismatched_comparison_candidates_do_not_create_a_run(comparison):
    root, _, _, _, context, folds, (baseline, model) = comparison
    baseline = replace(baseline, rankings=baseline.rankings.slice(1))
    with pytest.raises(ValueError, match="same candidates"):
        record_comparison(
            baseline=baseline,
            model=model,
            fold_results=folds,
            context=context,
            project_root=root,
        )
    assert not (root / "artifacts/mlflow").exists()


def test_failed_artifact_recording_marks_run_as_failed(comparison, monkeypatch):
    root, _, _, _, context, folds, (baseline, model) = comparison

    def fail(*args, **kwargs):
        raise OSError("artifact write failed")

    monkeypatch.setattr(MlflowClient, "log_artifacts", fail)
    with pytest.raises(OSError, match="artifact write failed"):
        record_comparison(
            baseline=baseline,
            model=model,
            fold_results=folds,
            context=context,
            project_root=root,
        )
    client = MlflowClient(
        tracking_uri=f"sqlite:///{(root / 'artifacts/mlflow/mlflow.db').as_posix()}"
    )
    runs = client.search_runs(
        [client.get_experiment_by_name("breakout-development-comparison").experiment_id]
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"


def test_reserved_dates_cannot_be_recorded_as_development_results(comparison):
    root, _, _, _, context, folds, (baseline, model) = comparison
    context = replace(
        context,
        holdout={
            **context.holdout,
            "boundaries": {
                **context.holdout["boundaries"],
                "development_end": "2022-06-30",
            },
        },
    )
    with pytest.raises(ValueError, match="reserved scoring dates"):
        record_comparison(
            baseline=baseline,
            model=model,
            fold_results=folds,
            context=context,
            project_root=root,
        )
    assert not (root / "artifacts/mlflow").exists()


def test_three_way_comparison_records_interaction_metrics_review_and_reloadable_models(
    comparison,
):
    root, paths, _, _, context, folds, (baseline, model) = comparison
    periods = [
        EvaluationPeriod(
            r.fold.period, r.fold.scoring_start, r.fold.scoring_end, r.fold.is_partial
        )
        for r in folds
    ]
    interaction_folds = run_logistic_walk_forward(
        pl.read_parquet(paths.daily_breakout_targets),
        periods,
        baseline.rankings.select("user_id", "artist_id", "scoring_date", "score"),
        interaction=True,
    )
    interaction = evaluate_scores(
        pl.concat([r.scores for r in interaction_folds]),
        pl.concat([r.fold.scoring for r in interaction_folds]),
        pl.read_parquet(paths.breakout_events),
        oos_start=periods[0].scoring_start,
        oos_end=periods[-1].scoring_end,
    )
    timing = TimingAudit(
        training_summary=pl.DataFrame({"period": ["2022 H1"], "positive_days": [1]}),
        ranking_summary=pl.DataFrame(
            {"approach": ["baseline"], "top3_day_rate": [0.5]}
        ),
        event_days=baseline.rankings.select(
            "user_id", "artist_id", "scoring_date", "rank"
        ),
    )
    run_id = record_comparison(
        baseline=baseline,
        model=model,
        fold_results=folds,
        context=context,
        project_root=root,
        interaction_model=interaction,
        interaction_fold_results=interaction_folds,
        timing_audit=timing,
    )
    client = MlflowClient(
        tracking_uri=f"sqlite:///{(root / 'artifacts/mlflow/mlflow.db').as_posix()}"
    )
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    for table in ["training_summary", "ranking_summary", "event_days"]:
        saved = pl.read_parquet(
            client.download_artifacts(run_id, f"diagnostics/timing/{table}.parquet")
        )
        assert saved.equals(getattr(timing, table))
    assert run.data.params["model.interaction"] == "False"
    assert run.data.params["interaction.interaction"] == "True"
    assert (
        run.data.params["interaction.terms"] == "share,log1p_count,share_x_log1p_count"
    )
    assert (
        run.data.metrics["interaction.detected_events"]
        == interaction.event_summary["detected_events"].item()
    )
    assert "interaction_vs_model.paired.same_day" in run.data.metrics
    review = pl.read_parquet(
        client.download_artifacts(
            run_id, "interaction/diagnostics/event_review.parquet"
        )
    )
    assert review.height == 1
    result = interaction_folds[0]
    local = client.download_artifacts(run_id, "interaction/folds/2022_H1/model")
    restored = mlflow.sklearn.load_model(local)
    inputs = result.fold.scoring.select(
        pl.col(MODEL_FEATURES).cast(pl.Float64)
    ).to_numpy()
    assert restored.named_steps["classifier"].n_features_in_ == 3
    np.testing.assert_array_equal(
        restored.decision_function(inputs), result.model.decision_function(inputs)
    )


def test_interaction_evaluation_requires_its_fitted_models_before_recording(comparison):
    root, _, _, _, context, folds, (baseline, model) = comparison
    with pytest.raises(ValueError, match="supplied together"):
        record_comparison(
            baseline=baseline,
            model=model,
            fold_results=folds,
            context=context,
            project_root=root,
            interaction_model=model,
        )
    assert not (root / "artifacts/mlflow").exists()
