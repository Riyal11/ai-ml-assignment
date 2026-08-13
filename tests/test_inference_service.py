"""Tests for the API inference service."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from docextract.api.inference import InferenceService

_VALID_INVOICE: dict[str, Any] = {
    "invoice_number": "10670",
    "vendor_name": "SuperStore",
    "invoice_date": "2012-08-15",
    "line_items": [{"description": "Phone", "quantity": 1, "unit_price": 10.0}],
    "subtotal": 10.0,
    "tax_amount": 0.0,
    "total_amount": 10.0,
    "currency": "USD",
}


class _FakePredictor:
    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        return _VALID_INVOICE

    def predict_raw(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return '{"invoice_number": "10670"}', _VALID_INVOICE


def test_stub_load_and_extract(tmp_path: Path) -> None:
    async def _run() -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        service = InferenceService(model_dir)
        await service.load()
        assert service.is_loaded
        response = await service.extract_invoice("Invoice text", "en")
        assert response.raw_output
        assert response.invoice is not None
        await service.unload()
        assert not service.is_loaded

    asyncio.run(_run())


def test_real_predictor_extract() -> None:
    async def _run() -> None:
        service = InferenceService(Path("Qwen/Qwen3-4B-Instruct-2507"))
        service._predictor = _FakePredictor()  # noqa: SLF001
        service._loaded = True  # noqa: SLF001
        response = await service.extract_invoice("Invoice text", "en")
        assert response.is_valid
        assert response.invoice == _VALID_INVOICE

    asyncio.run(_run())


def test_load_missing_path_raises(tmp_path: Path) -> None:
    async def _run() -> None:
        service = InferenceService(tmp_path / "missing")
        with pytest.raises(RuntimeError, match="model path not found"):
            await service.load()

    asyncio.run(_run())
