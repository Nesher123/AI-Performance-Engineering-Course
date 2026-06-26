from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task  # type: ignore[reportMissingImports]
from airflow.operators.python import (
    get_current_context,  # type: ignore[reportMissingImports]
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluate_agent import (  # noqa: E402
    DEFAULT_AGENT_CONFIG,
    DEFAULT_EXPERIMENT,
    DEFAULT_MODEL,
    _agent_command,
    _collect_metrics,
    _eval_command,
    _log_to_mlflow,
    _normalize_config,
    _read_json,
    _run_command,
    _write_json,
    _write_manifest,
)


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["swe-bench", "evaluation", "mlflow"],
    params={
        "split": "test",
        "subset": "verified",
        "workers": 1,
        "model": DEFAULT_MODEL,
        "task_slice": "0:1",
        "run_id": "",
        "cost_limit": 0,
        "dataset_name": "",
        "agent_config": DEFAULT_AGENT_CONFIG,
        "mlflow_tracking_uri": "",
        "mlflow_experiment": DEFAULT_EXPERIMENT,
    },
)
def evaluate_agent_dag():
    @task
    def prepare_run() -> dict[str, Any]:
        context = get_current_context()
        config = _normalize_config(context["params"])
        run_dir = Path(config["run_dir"])
        (run_dir / "run-agent" / "trajectories").mkdir(parents=True, exist_ok=True)
        (run_dir / "run-eval" / "reports").mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "config.json", config)
        return config

    @task
    def run_agent(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        agent_dir = run_dir / "run-agent"
        trajectories_dir = agent_dir / "trajectories"
        env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
        _run_command(
            _agent_command(config, trajectories_dir),
            cwd=PROJECT_ROOT,
            env=env,
            command_path=agent_dir / "command.json",
            log_path=agent_dir / "run.log",
        )

        generated_preds = trajectories_dir / "preds.json"
        if not generated_preds.exists():
            raise FileNotFoundError(f"mini-swe-agent did not create {generated_preds}")
        shutil.copy2(generated_preds, agent_dir / "preds.json")
        return config

    @task
    def run_eval(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        eval_dir = run_dir / "run-eval"
        preds_path = run_dir / "run-agent" / "preds.json"
        if not preds_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {preds_path}")
        _run_command(
            _eval_command(config, preds_path, eval_dir / "reports"),
            cwd=eval_dir,
            env=os.environ.copy(),
            command_path=eval_dir / "command.json",
            log_path=eval_dir / "run.log",
        )
        return config

    @task
    def summarize_and_log(config: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(config["run_dir"])
        metrics = _collect_metrics(run_dir)
        _write_json(run_dir / "metrics.json", metrics)
        manifest_path = _write_manifest(run_dir, config, metrics)
        mlflow_result = _log_to_mlflow(config, metrics, manifest_path)

        manifest = _read_json(manifest_path)
        manifest["mlflow"] = mlflow_result
        _write_json(manifest_path, manifest)
        return {"run_dir": str(run_dir), "metrics": metrics, "mlflow": mlflow_result}

    summarize_and_log(run_eval(run_agent(prepare_run())))


evaluate_agent_dag()
