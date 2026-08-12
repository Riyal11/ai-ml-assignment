"""Celery task for asynchronous model quantization."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from celery import Celery

logger = logging.getLogger(__name__)

_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("docextract", broker=_BROKER_URL)

_METHOD_ALIASES = {
    "gguf": "gguf-q4_k_m",
    "gguf-q4_k_m": "gguf-q4_k_m",
}


def _normalize_method(method: str) -> str:
    """Map short method names to quantize script identifiers."""
    return _METHOD_ALIASES.get(method, method)


def _run_quantization(model_path: str, output_dir: str, method: str) -> dict[str, Any]:
    """Execute quantization synchronously and return a result payload."""
    from scripts.quantize import quantize_model

    resolved_method = _normalize_method(method)
    output_path = quantize_model(
        Path(model_path),
        Path(output_dir),
        resolved_method,
    )
    return {
        "output_path": str(output_path),
        "method": resolved_method,
        "model_path": model_path,
    }


@celery_app.task(bind=True, max_retries=3)  # type: ignore[untyped-decorator]
def quantize_model_task(
    self: Any,
    model_path: str,
    output_dir: str,
    method: str = "gguf-q4_k_m",
) -> dict[str, Any]:
    """Quantize a merged HuggingFace model asynchronously.

    When ``CELERY_BROKER_URL`` is unset, callers should invoke
    ``run_quantize_sync`` instead — this task expects a running Celery worker
    and Redis broker.

    Args:
        self: Celery task instance (bound).
        model_path: Path to merged HuggingFace model directory.
        output_dir: Directory for quantized artifacts.
        method: Quantization method (``gguf``, ``gguf-q4_k_m``, ``awq``, ``gptq``).

    Returns:
        Result dict with ``output_path`` on success.

    Raises:
        Exception: Re-raised after updating task state to ``FAILURE``.
    """
    try:
        self.update_state(state="STARTED", meta={"model_path": model_path, "method": method})
        result = _run_quantization(model_path, output_dir, method)
        return result
    except Exception as exc:
        logger.exception("Quantization task failed for %s", model_path)
        raise self.retry(exc=exc, countdown=5) from exc


def run_quantize_sync(model_path: str, output_dir: str, method: str) -> dict[str, Any]:
    """Run quantization synchronously when Celery is unavailable.

    Logs a warning when ``CELERY_BROKER_URL`` is not configured and executes
    the quantize script logic in-process.

    Args:
        model_path: Path to merged HuggingFace model directory.
        output_dir: Directory for quantized artifacts.
        method: Quantization method identifier.

    Returns:
        Result dict with ``output_path`` on success.
    """
    if not os.getenv("CELERY_BROKER_URL"):
        logger.warning("CELERY_BROKER_URL not set; running quantization synchronously in-process")
    return _run_quantization(model_path, output_dir, method)


def celery_enabled() -> bool:
    """Return whether Celery async dispatch should be used."""
    return bool(os.getenv("CELERY_BROKER_URL"))
