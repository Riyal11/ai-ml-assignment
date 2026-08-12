"""Tests for the evaluation pipeline (stub inference)."""

import json
from pathlib import Path
from typing import Any

import pytest

from docextract.data.dataset import Split
from docextract.eval.pipeline import run_evaluation


def make_record(idx: int) -> dict[str, Any]:
    """Return a minimal eval record with a target JSON object."""
    return {
        "example_id": f"ex-{idx}",
        "document": f"Invoice INV-{idx:04d} from Vendor {idx}.",
        "target": {
            "invoice_number": f"INV-{idx:04d}",
            "vendor_name": f"Vendor {idx}",
            "invoice_date": "2025-06-15",
            "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 10}],
            "subtotal": 10,
            "tax_amount": 2,
            "total_amount": 12,
            "currency": "USD",
        },
        "language": "en",
        "split": "validation",
    }


@pytest.fixture
def eval_dataset(tmp_path: Path) -> Path:
    """Write a small JSONL eval dataset and return its path."""
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(make_record(i)) + "\n")
    return path


def test_run_evaluation_writes_results(eval_dataset: Path, tmp_path: Path) -> None:
    out_path = run_evaluation(
        model_path=tmp_path / "model",
        dataset_path=eval_dataset,
        output_dir=tmp_path / "out",
        split=Split.VALIDATION,
    )
    assert out_path.exists()
    results = json.loads(out_path.read_text(encoding="utf-8"))
    for key in (
        "run_id",
        "split",
        "model_path",
        "schema_validity_rate",
        "exact_match",
        "precision_recall_f1",
        "num_examples",
        "timestamp",
    ):
        assert key in results


def test_results_num_examples_matches_dataset(eval_dataset: Path, tmp_path: Path) -> None:
    out_path = run_evaluation(
        model_path=tmp_path / "model",
        dataset_path=eval_dataset,
        output_dir=tmp_path / "out",
        split=Split.VALIDATION,
    )
    results = json.loads(out_path.read_text(encoding="utf-8"))
    assert results["num_examples"] == 3


def test_results_split_recorded(eval_dataset: Path, tmp_path: Path) -> None:
    out_path = run_evaluation(
        model_path=tmp_path / "model",
        dataset_path=eval_dataset,
        output_dir=tmp_path / "out",
        split=Split.GOLDEN,
    )
    results = json.loads(out_path.read_text(encoding="utf-8"))
    assert results["split"] == "golden"


def test_results_rates_are_bounded_floats(eval_dataset: Path, tmp_path: Path) -> None:
    out_path = run_evaluation(
        model_path=tmp_path / "model",
        dataset_path=eval_dataset,
        output_dir=tmp_path / "out",
        split=Split.VALIDATION,
    )
    results = json.loads(out_path.read_text(encoding="utf-8"))
    for metric in ("schema_validity_rate", "exact_match"):
        value = results[metric]
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
    prf = results["precision_recall_f1"]
    for submetric in ("precision", "recall", "f1"):
        assert 0.0 <= prf[submetric] <= 1.0
