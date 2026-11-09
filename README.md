# sonora

## Breakout model comparison

Run from the repository root:

```bash
uv sync
uv run marimo edit notebooks/marimo/11_breakout_model_comparison.py
```

Use **Record comparison in MLflow** to save the displayed comparison. Opening the
notebook or exporting HTML does not record a run. Each click creates a new run.

To inspect saved runs:

```bash
export MLFLOW_DISABLE_TELEMETRY=true
uv run mlflow server --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Run metadata and artifacts are stored under
`artifacts/mlflow/`, which is excluded from Git. Keep
`artifacts/breakout/first_ml_holdout.json` in Git to preserve the reserved dates.

In PowerShell, set `$env:MLFLOW_DISABLE_TELEMETRY="true"` instead of using `export`.
