# Airflow Evaluation Pipeline Report

## Architecture

The pipeline is implemented as the `evaluate_agent` Airflow DAG in `dags/evaluate_agent.py`.
It runs the SWE-bench-style evaluation in four explicit tasks:

1. `prepare_run` normalizes Airflow params and writes `runs/<run-id>/config.json`.
2. `run_agent` runs `mini-extra swebench` and writes trajectories plus `run-agent/preds.json`.
3. `run_eval` runs `python -m swebench.harness.run_evaluation` against that `preds.json`.
4. `summarize_and_log` parses SWE-bench reports, writes metrics and a manifest, then logs params/metrics/artifacts to MLflow.

The Airflow file is intentionally thin. Reusable helper logic lives in `pipeline/evaluate_agent.py`, which makes local checks possible without importing Airflow.

## DAG Parameters

The DAG can be triggered from the Airflow UI with these example tiny-run params:

```json
{
  "split": "test",
  "subset": "verified",
  "workers": 1,
  "model": "nebius/moonshotai/Kimi-K2.6",
  "task_slice": "0:1",
  "run_id": "tiny-local-cli-002",
  "cost_limit": 0,
  "dataset_name": "",
  "agent_config": "benchmarks/swebench.yaml",
  "mlflow_tracking_uri": "",
  "mlflow_experiment": "di-mavericks-fire-evaluation"
}
```

If `run_id` is empty, the DAG generates one. If `mlflow_tracking_uri` is empty, the DAG uses a local SQLite MLflow backend at `sqlite:///.../mlflow.db`.

## Artifact Layout

Each run writes a self-contained directory:

```text
runs/<run-id>/
  config.json
  run-agent/
    command.json
    run.log
    preds.json
    trajectories/
  run-eval/
    command.json
    run.log
    logs/
  metrics.json
  manifest.json
  mlflow_command.json
  mlflow.log
  mlflow_payload.json
  mlflow_result.json
```

`manifest.json` records the important relative paths and summary metrics, so the run directory is enough to reconstruct what was executed and what was produced. The committed sample at `runs/tiny-local-cli-002/` is intentionally small but includes config, predictions, trajectory output, SWE-bench evaluation logs/reports, metrics, manifest, and MLflow run metadata. The generated MLflow backend files (`mlflow.db`, `mlruns/`) are ignored; `mlflow_result.json` and `manifest.json` preserve the run ID, tracking URI, and artifact URI evidence. A production version would upload this directory to object storage and log that URI to MLflow.

## Local Development Checks

Use the local-first loop:

```bash
uv run ruff check dags/evaluate_agent.py pipeline/evaluate_agent.py scripts/smoke_evaluate_agent.py pyproject.toml
uv run python -m py_compile dags/evaluate_agent.py pipeline/evaluate_agent.py scripts/smoke_evaluate_agent.py
uv run python scripts/smoke_evaluate_agent.py
```

The smoke script uses sample predictions and SWE-bench reports from `sample/`, builds a temporary run directory, verifies metrics and manifest contents, and logs a run to a temporary SQLite MLflow store.

Airflow itself is not a project dependency. It is provided by `uv tool run apache-airflow`, while DAG tasks re-enter the project environment with `uv run` for `mini-swe-agent`, SWE-bench, and MLflow commands.

## VM Run Instructions

On the VM:

```bash
cd /home/ofir/di-mavericks-fire
~/.local/bin/uv sync
bash run-airflow-standalone.sh
```

Open Airflow on port `8080`, trigger `evaluate_agent`, and use the tiny params shown above first. After the DAG completes, inspect:

```bash
runs/<run-id>/config.json
runs/<run-id>/run-agent/preds.json
runs/<run-id>/run-eval/logs/
runs/<run-id>/metrics.json
runs/<run-id>/manifest.json
runs/<run-id>/mlflow_result.json
```

To open the local MLflow UI for the default SQLite backend:

```bash
~/.local/bin/uv run mlflow ui --backend-store-uri sqlite:////home/ofir/di-mavericks-fire/mlflow.db --port 5000
```

## Verification Evidence

Local checks completed:

- `ruff check` passed for the DAG, helper module, smoke script, and `pyproject.toml`.
- `py_compile` passed for the DAG, helper module, and smoke script.
- `scripts/smoke_evaluate_agent.py` passed against sample artifacts, including MLflow logging.

VM checkpoint completed on `ofir-mlops-hw3`:

- `uv sync` completed after syncing the local implementation.
- Remote `ruff check` passed for the DAG helpers, smoke script, dependency file, and report.
- Remote `py_compile` passed for the DAG, helper module, and smoke script.
- Remote `DagBag` import found `evaluate_agent` and `mini-swe-bench-single` with no import errors.
- Remote smoke check passed using sample predictions/reports and temporary SQLite MLflow tracking.

Completed tiny VM run:

- Airflow command: `airflow dags test evaluate_agent ...`
- Run ID: `tiny-verified-cli-003`
- Artifact root: `/home/ofir/di-mavericks-fire/runs/tiny-verified-cli-003`
- MLflow tracking URI: `sqlite:////home/ofir/di-mavericks-fire/mlflow.db`
- MLflow run ID: `0f462460f31f4b3b98494bd4a5eb8429`
- Metrics: `predictions_count=1`, `evaluated_count=1`, `resolved_count=1`, `resolve_rate=1.0`

Completed tiny local e2e run:

- Airflow command: `airflow dags test evaluate_agent ...`
- Run ID: `tiny-local-cli-002`
- Artifact root: `runs/tiny-local-cli-002`
- MLflow tracking URI: `sqlite:////Users/ofir.n/GIT/DI/di-mavericks-fire/mlflow.db`
- MLflow run ID: `b48d6617d0484270a69585cd1bc85553`
- Metrics: `predictions_count=1`, `evaluated_count=1`, `resolved_count=1`, `resolve_rate=1.0`

Two runtime fixes came from the tiny run:

- `mini-extra swebench` batch mode does not accept `--cost-limit`, so the DAG records `cost_limit` in config/MLflow but does not pass it to the batch agent command.
- SWE-bench writes per-instance `report.json` files under `run-eval/logs/run_evaluation/...`; metrics collection now scans the full `run-eval/` tree instead of only `run-eval/reports/`.
