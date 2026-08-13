"""Tests for evaluation inference helpers."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from docextract.eval.inference import (
    HfBaseModelPredictor,
    is_loadable_model,
    load_predictor,
    resolve_adapter_path,
)


def test_is_loadable_model_hub_id() -> None:
    assert is_loadable_model(Path("Qwen/Qwen3-4B-Instruct-2507")) is True


def test_is_loadable_model_windows_path_not_hub(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    assert is_loadable_model(missing) is False


def test_is_loadable_model_local_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert is_loadable_model(model_dir) is True


def test_resolve_adapter_path_prefers_parent(tmp_path: Path) -> None:
    adapter_root = tmp_path / "run-001"
    adapter_root.mkdir()
    (adapter_root / "adapter_config.json").write_text("{}", encoding="utf-8")
    nested = adapter_root / "final"
    nested.mkdir()
    assert resolve_adapter_path(nested) == adapter_root


def test_load_predictor_stub() -> None:
    predictor = load_predictor(Path("missing"), use_stub=True)
    assert predictor.predict({"document": "x", "language": "en"}) == {}


def test_predict_raw_returns_text_and_dict() -> None:
    record = {
        "document": "Invoice INV-1",
        "language": "en",
        "target": {
            "invoice_number": "1",
            "vendor_name": "Acme",
            "invoice_date": "2025-01-01",
            "line_items": [{"description": "x", "quantity": 1, "unit_price": 1}],
            "subtotal": 1,
            "tax_amount": 0,
            "total_amount": 1,
            "currency": "USD",
        },
    }
    predictor = MagicMock(spec=HfBaseModelPredictor)
    predictor.predict_raw.return_value = ('{"invoice_number": "1"}', {"invoice_number": "1"})
    raw, parsed = predictor.predict_raw(record)
    assert raw.startswith("{")
    assert parsed["invoice_number"] == "1"


def test_load_predictor_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePredictor:
        def predict(self, record: dict[str, Any]) -> dict[str, Any]:
            return {}

        def predict_raw(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            return "{}", {}

    monkeypatch.setattr(
        "docextract.eval.inference.HfBaseModelPredictor",
        lambda *args, **kwargs: FakePredictor(),
    )
    predictor = load_predictor(Path("org/model-name"))
    assert predictor.predict({"document": "x", "language": "en"}) == {}
