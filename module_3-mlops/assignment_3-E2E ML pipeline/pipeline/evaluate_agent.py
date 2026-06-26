from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"
DEFAULT_AGENT_CONFIG = "benchmarks/swebench.yaml"
DEFAULT_EXPERIMENT = "di-mavericks-fire-evaluation"
DATASET_BY_SUBSET = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "full": "SWE-bench/SWE-bench",
}


def _uv_bin() -> str:
    return (
        os.environ.get("UV_BIN")
        or shutil.which("uv")
        or str(Path.home() / ".local" / "bin" / "uv")
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.SubprocessError:
        return None


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    command_path: Path,
    log_path: Path,
) -> None:
    _write_json(
        command_path,
        {
            "command": command,
            "cwd": str(cwd),
            "started_at": _utc_now(),
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    command_payload = _read_json(command_path)
    command_payload.update(
        {"finished_at": _utc_now(), "returncode": completed.returncode}
    )
    _write_json(command_path, command_payload)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}. "
            f"See log: {log_path}"
        )


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _has_artifact_content(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.rglob("*"))
    return False


def _normalize_config(params: dict[str, Any]) -> dict[str, Any]:
    subset = str(params.get("subset") or "verified")
    run_id = _safe_id(str(params.get("run_id") or ""))
    if not params.get("run_id"):
        run_id = f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    task_slice = params.get("task_slice")
    if task_slice is not None:
        task_slice = str(task_slice).strip() or None

    cost_limit = params.get("cost_limit")
    if cost_limit is not None and str(cost_limit).strip() != "":
        cost_limit = float(cost_limit)
    else:
        cost_limit = None

    workers = int(params.get("workers") or 1)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    dataset_name = str(
        params.get("dataset_name") or DATASET_BY_SUBSET.get(subset, subset)
    )
    tracking_uri = params.get("mlflow_tracking_uri") or os.environ.get(
        "MLFLOW_TRACKING_URI"
    )
    if not tracking_uri:
        tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

    runs_root = Path(str(params.get("runs_root") or RUNS_ROOT))
    run_dir = runs_root / run_id
    return {
        "run_id": run_id,
        "created_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": _git_commit(),
        "split": str(params.get("split") or "test"),
        "subset": subset,
        "dataset_name": dataset_name,
        "workers": workers,
        "model": str(params.get("model") or DEFAULT_MODEL),
        "task_slice": task_slice,
        "cost_limit": cost_limit,
        "agent_config": str(params.get("agent_config") or DEFAULT_AGENT_CONFIG),
        "mlflow_tracking_uri": str(tracking_uri),
        "mlflow_experiment": str(params.get("mlflow_experiment") or DEFAULT_EXPERIMENT),
        "run_dir": str(run_dir),
    }


def _agent_command(config: dict[str, Any], trajectories_dir: Path) -> list[str]:
    command = [
        _uv_bin(),
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        config["subset"],
        "--split",
        config["split"],
        "--model",
        config["model"],
        "--config",
        config["agent_config"],
        "--workers",
        str(config["workers"]),
        "-o",
        str(trajectories_dir),
    ]
    if config.get("task_slice"):
        command.extend(["--slice", str(config["task_slice"])])
    return command


def _eval_command(
    config: dict[str, Any], preds_path: Path, reports_dir: Path
) -> list[str]:
    return [
        _uv_bin(),
        "run",
        "--project",
        str(PROJECT_ROOT),
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        config["dataset_name"],
        "--split",
        config["split"],
        "--predictions_path",
        str(preds_path),
        "--max_workers",
        str(config["workers"]),
        "--run_id",
        config["run_id"],
        "--report_dir",
        str(reports_dir),
    ]


def _collect_metrics(run_dir: Path) -> dict[str, Any]:
    preds_path = run_dir / "run-agent" / "preds.json"
    eval_dir = run_dir / "run-eval"
    predictions = _read_json(preds_path) if preds_path.exists() else {}

    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted(eval_dir.glob("**/report.json")):
        report = _read_json(report_path)
        for instance_id, result in report.items():
            if isinstance(result, dict):
                reports[instance_id] = result

    resolved = sum(1 for result in reports.values() if result.get("resolved") is True)
    applied = sum(
        1
        for result in reports.values()
        if result.get("patch_successfully_applied") is True
    )
    failed_apply = sum(
        1
        for result in reports.values()
        if result.get("patch_successfully_applied") is False
    )
    empty_patch = sum(
        1 for result in reports.values() if result.get("patch_exists") is False
    )
    total = len(reports)

    return {
        "predictions_count": len(predictions),
        "evaluated_count": total,
        "resolved_count": resolved,
        "unresolved_count": total - resolved,
        "patch_applied_count": applied,
        "patch_failed_apply_count": failed_apply,
        "empty_patch_count": empty_patch,
        "resolve_rate": resolved / total if total else 0.0,
    }


def _write_manifest(
    run_dir: Path, config: dict[str, Any], metrics: dict[str, Any]
) -> Path:
    paths = {
        "config": run_dir / "config.json",
        "agent_predictions": run_dir / "run-agent" / "preds.json",
        "agent_trajectories": run_dir / "run-agent" / "trajectories",
        "agent_command": run_dir / "run-agent" / "command.json",
        "agent_log": run_dir / "run-agent" / "run.log",
        "eval_reports": run_dir / "run-eval" / "reports",
        "eval_logs": run_dir / "run-eval" / "logs",
        "eval_command": run_dir / "run-eval" / "command.json",
        "eval_log": run_dir / "run-eval" / "run.log",
        "metrics": run_dir / "metrics.json",
    }
    manifest = {
        "run_id": config["run_id"],
        "created_at": config["created_at"],
        "finished_at": _utc_now(),
        "project_root": config["project_root"],
        "git_commit": config["git_commit"],
        "metrics": metrics,
        "artifact_root": str(run_dir),
        "paths": {
            key: _relative_to_run(path, run_dir)
            for key, path in paths.items()
            if _has_artifact_content(path)
        },
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _log_to_mlflow(
    config: dict[str, Any], metrics: dict[str, Any], manifest_path: Path
) -> dict[str, str | None]:
    run_dir = Path(config["run_dir"])
    payload_path = run_dir / "mlflow_payload.json"
    result_path = run_dir / "mlflow_result.json"
    payload = {
        "config": config,
        "metrics": metrics,
        "manifest_path": str(manifest_path),
        "metrics_path": str(run_dir / "metrics.json"),
        "config_path": str(run_dir / "config.json"),
        "result_path": str(result_path),
    }
    _write_json(payload_path, payload)

    script = r"""
import json
import sys

import mlflow

payload = json.loads(open(sys.argv[1]).read())
config = payload["config"]
metrics = payload["metrics"]
mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
mlflow.set_experiment(config["mlflow_experiment"])
with mlflow.start_run(run_name=config["run_id"]) as run:
    params = {
        "run_id": config["run_id"],
        "split": config["split"],
        "subset": config["subset"],
        "dataset_name": config["dataset_name"],
        "workers": config["workers"],
        "model": config["model"],
        "task_slice": config.get("task_slice") or "",
        "cost_limit": "" if config.get("cost_limit") is None else config["cost_limit"],
        "git_commit": config.get("git_commit") or "",
        "artifact_root": config["run_dir"],
    }
    mlflow.log_params(params)
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    mlflow.log_metrics(numeric_metrics)
    mlflow.set_tags(
        {
            "pipeline": "airflow-evaluate-agent",
            "artifact_root": config["run_dir"],
            "local_run_id": config["run_id"],
        }
    )
    mlflow.log_artifact(payload["config_path"])
    mlflow.log_artifact(payload["metrics_path"])
    mlflow.log_artifact(payload["manifest_path"])
    result = {
        "mlflow_run_id": run.info.run_id,
        "mlflow_experiment_id": run.info.experiment_id,
        "mlflow_artifact_uri": run.info.artifact_uri,
        "mlflow_tracking_uri": config["mlflow_tracking_uri"],
    }
open(payload["result_path"], "w").write(json.dumps(result, indent=2, sort_keys=True) + "\n")
"""
    _run_command(
        [
            _uv_bin(),
            "run",
            "--project",
            str(PROJECT_ROOT),
            "python",
            "-c",
            script,
            str(payload_path),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "MLFLOW_ALLOW_FILE_STORE": "true"},
        command_path=run_dir / "mlflow_command.json",
        log_path=run_dir / "mlflow.log",
    )
    return _read_json(result_path)
