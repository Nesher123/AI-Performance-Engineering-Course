from __future__ import annotations

import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluate_agent import (  # noqa: E402
    DEFAULT_MODEL,
    _agent_command,
    _collect_metrics,
    _log_to_mlflow,
    _normalize_config,
    _read_json,
    _write_json,
    _write_manifest,
)

SAMPLE_ROOT = PROJECT_ROOT / "sample"


def main() -> None:
    tmp_dir = Path(mkdtemp(prefix="evaluate-agent-smoke-"))
    try:
        runs_root = tmp_dir / "runs"
        mlflow_tracking_uri = f"sqlite:///{tmp_dir / 'mlflow.db'}"
        config = _normalize_config(
            {
                "run_id": "local-smoke",
                "split": "test",
                "subset": "verified",
                "workers": 1,
                "model": DEFAULT_MODEL,
                "task_slice": "0:3",
                "cost_limit": 0,
                "runs_root": str(runs_root),
                "mlflow_tracking_uri": mlflow_tracking_uri,
                "mlflow_experiment": "local-smoke-evaluate-agent",
            }
        )
        run_dir = Path(config["run_dir"])
        agent_command = _agent_command(config, run_dir / "run-agent" / "trajectories")
        assert "--slice" in agent_command
        assert "--cost-limit" not in agent_command

        agent_dir = run_dir / "run-agent"
        eval_dir = run_dir / "run-eval"
        trajectories_dir = agent_dir / "trajectories"
        reports_dir = eval_dir / "logs" / "run_evaluation"

        shutil.copytree(SAMPLE_ROOT / "trajectories", trajectories_dir)
        shutil.copy2(trajectories_dir / "preds.json", agent_dir / "preds.json")
        shutil.copytree(SAMPLE_ROOT / "logs" / "run_evaluation", reports_dir)
        _write_json(run_dir / "config.json", config)
        _write_json(agent_dir / "command.json", {"command": ["sample-agent"]})
        (agent_dir / "run.log").write_text("sample agent log\n")
        _write_json(eval_dir / "command.json", {"command": ["sample-eval"]})
        (eval_dir / "run.log").write_text("sample eval log\n")

        metrics = _collect_metrics(run_dir)
        assert metrics["predictions_count"] == 3, metrics
        assert metrics["evaluated_count"] == 3, metrics
        assert metrics["resolved_count"] == 1, metrics
        assert metrics["resolve_rate"] == 1 / 3, metrics

        _write_json(run_dir / "metrics.json", metrics)
        manifest_path = _write_manifest(run_dir, config, metrics)
        manifest = _read_json(manifest_path)
        assert manifest["paths"]["agent_predictions"] == "run-agent/preds.json"
        assert manifest["paths"]["eval_logs"] == "run-eval/logs"
        assert manifest["metrics"] == metrics

        mlflow_result = _log_to_mlflow(config, metrics, manifest_path)
        assert mlflow_result["mlflow_run_id"]
        assert mlflow_result["mlflow_tracking_uri"] == mlflow_tracking_uri
        assert (run_dir / "mlflow_result.json").exists()

        print(f"smoke ok: {run_dir}")
    except Exception:
        print(f"smoke failed, preserved temp dir: {tmp_dir}")
        raise
    else:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
