"""Tests for forgetting metric helpers."""

import json
from pathlib import Path

import pytest

from docextract.eval.metrics import compute_forgetting_score


def test_compute_forgetting_score_retention() -> None:
    retention = compute_forgetting_score([0.85], [0.755])
    assert retention == pytest.approx(0.755 / 0.85)


def test_relative_drop_from_results(tmp_path: Path) -> None:
    base = {"precision_recall_f1": {"f1": 0.85}}
    ft = {"precision_recall_f1": {"f1": 0.755}}
    retention = compute_forgetting_score(
        [base["precision_recall_f1"]["f1"]],
        [ft["precision_recall_f1"]["f1"]],
    )
    relative_drop = 1.0 - retention
    ft["relative_benchmark_drop"] = relative_drop
    out = tmp_path / "results.json"
    out.write_text(json.dumps(ft), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["relative_benchmark_drop"] == pytest.approx(1.0 - (0.755 / 0.85))
