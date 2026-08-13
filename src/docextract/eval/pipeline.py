"""Evaluation pipeline: dataset → inference (stub) → metrics → results.json."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docextract.data.dataset import Split, validate_eval_split
from docextract.eval.inference import is_loadable_model, load_predictor
from docextract.eval.metrics import (
    SCALAR_FIELDS,
    compute_exact_match,
    compute_precision_recall_f1,
    compute_schema_validity_rate,
)

logger = logging.getLogger(__name__)


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Load evaluation records from a JSONL file.

    Each line must be a JSON object with at least a ``target`` field.
    """
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _stub_predict(record: dict[str, Any]) -> dict[str, Any]:
    """Stub inference — returns an empty prediction.

    Placeholder for the real model call; kept so the pipeline structure
    (loading, scoring, reporting) is exercised end-to-end.
    """
    _ = record
    return {}


def _use_stub_inference(model_path: Path, use_stub: bool | None) -> bool:
    if use_stub is not None:
        return use_stub
    return not is_loadable_model(model_path)


def _aggregate_field_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Micro-average per-field precision/recall/F1 across all examples."""
    fields = (*SCALAR_FIELDS, "line_items")
    totals = {field: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for field in fields}
    n = len(records)
    if n == 0:
        return totals
    for record, pred in zip(records, predictions, strict=True):
        prf = compute_precision_recall_f1(pred, record.get("target", {}))
        for field in fields:
            totals[field]["precision"] += prf[field]["precision"]
            totals[field]["recall"] += prf[field]["recall"]
            totals[field]["f1"] += prf[field]["f1"]
    return {
        field: {metric: value / n for metric, value in scores.items()}
        for field, scores in totals.items()
    }


def run_evaluation(
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    split: Split,
    *,
    use_stub: bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Run the evaluation pipeline and write ``results.json``.

    Args:
        model_path: Path to the (merged/adapted) model artifact.
        dataset_path: JSONL evaluation dataset with ``target`` fields.
        output_dir: Directory to write ``results.json`` into.
        split: Which split is being evaluated.

    Returns:
        Path to the written ``results.json``.
    """
    validate_eval_split(split)
    records = _load_records(dataset_path)
    if _use_stub_inference(model_path, use_stub):
        logger.warning("Using stub inference for model path %s", model_path)
        predictions = [_stub_predict(record) for record in records]
    else:
        predictor = load_predictor(model_path, local_files_only=local_files_only)
        predictions = []
        for index, record in enumerate(records, start=1):
            predictions.append(predictor.predict(record))
            if index == 1 or index % 5 == 0 or index == len(records):
                logger.info("Inference progress: %d/%d", index, len(records))

    schema_validity_rate = compute_schema_validity_rate(predictions)

    em_overalls: list[float] = []
    prf_overalls: list[dict[str, float]] = []
    for record, pred in zip(records, predictions, strict=True):
        gold = record.get("target", {})
        em_overalls.append(compute_exact_match(pred, gold)["_overall"])
        prf_overalls.append(compute_precision_recall_f1(pred, gold)["_overall"])

    n = len(records)
    em_mean = sum(em_overalls) / n if n else 0.0
    p_mean = sum(o["precision"] for o in prf_overalls) / n if n else 0.0
    r_mean = sum(o["recall"] for o in prf_overalls) / n if n else 0.0
    f1_mean = sum(o["f1"] for o in prf_overalls) / n if n else 0.0

    now = datetime.now(UTC)
    results = {
        "run_id": f"eval-{now.strftime('%Y%m%d-%H%M%S')}",
        "split": split.value,
        "model_path": str(model_path),
        "schema_validity_rate": schema_validity_rate,
        "exact_match": em_mean,
        "precision_recall_f1": {"precision": p_mean, "recall": r_mean, "f1": f1_mean},
        "field_precision_recall_f1": _aggregate_field_metrics(records, predictions),
        "num_examples": n,
        "timestamp": now.isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(
        "Eval on %s: %d examples, validity=%.2f, EM=%.2f, F1=%.2f",
        split.value,
        n,
        schema_validity_rate,
        em_mean,
        f1_mean,
    )
    return out_path
