"""Export MLflow experiment runs to submission-friendly JSON and CSV."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docextract.train.run_config import _MANDATED_FIELDS

logger = logging.getLogger(__name__)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_name",
    "status",
    "start_time",
    "end_time",
    *_MANDATED_FIELDS,
    "base_model",
    "seed",
    "torch_dtype",
    "train_time_sec",
    "peak_gpu_mem_mb",
    "trainable_params",
    "total_params",
    "train_loss",
    "eval_loss",
)


def _ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()


def _run_payload(run: Any) -> dict[str, Any]:
    params = dict(run.data.params)
    metrics = {key: value for key, value in run.data.metrics.items()}
    tags = dict(run.data.tags)
    mandated = {field: params.get(field) for field in _MANDATED_FIELDS}
    if mandated.get("run_id") is None and "custom_run_id" in params:
        mandated["run_id"] = params["custom_run_id"]
    extras = {
        key: value
        for key, value in params.items()
        if key not in _MANDATED_FIELDS and key != "custom_run_id"
    }
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": _ms_to_iso(run.info.start_time),
        "end_time": _ms_to_iso(run.info.end_time),
        "params": {**mandated, **extras},
        "metrics": metrics,
        "tags": tags,
    }


def _summary_row(payload: dict[str, Any]) -> dict[str, str]:
    params = payload["params"]
    metrics = payload["metrics"]
    row: dict[str, str] = {
        "run_id": str(payload["run_id"]),
        "run_name": str(payload.get("run_name") or ""),
        "status": str(payload.get("status") or ""),
        "start_time": str(payload.get("start_time") or ""),
        "end_time": str(payload.get("end_time") or ""),
    }
    for field in _MANDATED_FIELDS:
        row[field] = str(params.get(field, ""))
    for field in ("base_model", "seed", "torch_dtype"):
        row[field] = str(params.get(field, ""))
    for field in ("train_time_sec", "peak_gpu_mem_mb", "trainable_params", "total_params"):
        value = metrics.get(field, params.get(field, ""))
        row[field] = str(value) if value != "" else ""
    for field in ("train_loss", "eval_loss"):
        row[field] = str(metrics.get(field, ""))
    return row


def _write_summary_csv(output_dir: Path, rows: list[dict[str, str]]) -> Path:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def export_mlflow_runs(
    output_dir: Path,
    *,
    experiment_name: str = "docextract",
    tracking_uri: str | None = None,
) -> int:
    """Export all runs for an MLflow experiment.

    Args:
        output_dir: Directory for per-run JSON files and ``summary.csv``.
        experiment_name: MLflow experiment name to query.
        tracking_uri: Optional tracking URI override.

    Returns:
        Number of runs exported (0 when the experiment or runs are missing).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    experiment = client.get_experiment_by_name(experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if experiment is None:
        logger.warning("Experiment %r not found; writing empty summary only", experiment_name)
        _write_summary_csv(output_dir, [])
        return 0

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        logger.warning("No runs found for experiment %r", experiment_name)
        _write_summary_csv(output_dir, [])
        return 0

    summary_rows: list[dict[str, str]] = []
    for run in runs:
        payload = _run_payload(run)
        out_path = output_dir / f"{payload['run_id']}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_rows.append(_summary_row(payload))

    summary_path = _write_summary_csv(output_dir, summary_rows)
    logger.info("Exported %d run(s) to %s (summary: %s)", len(runs), output_dir, summary_path)
    return len(runs)
