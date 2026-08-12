"""Inference service wrapper for the FastAPI application."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from docextract.api.models import ExtractResponse
from docextract.data.format_sft import build_extraction_prompt
from docextract.data.validation import validate_invoice

logger = logging.getLogger(__name__)

_DUMMY_RESPONSE = '{"invoice_number": "DUMMY-001"}'


class InferenceService:
    """Load and run the extraction model for API requests."""

    def __init__(self, model_path: Path, quantization: str = "none") -> None:
        """Initialize the service without loading weights.

        Args:
            model_path: Path to merged HF model directory or GGUF file.
            quantization: Serving backend (``none``, ``gguf``, ``awq``).
        """
        self.model: Any = None
        self.model_path = model_path
        self.quantization = quantization
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has completed startup loading."""
        return self._loaded

    async def load(self) -> None:
        """Load model weights based on the configured quantization backend.

        Raises:
            RuntimeError: If the model path does not exist.
        """
        if not self.model_path.exists():
            raise RuntimeError(f"model path not found: {self.model_path}")

        # TODO: load transformers model for ``none`` or llama-cpp for ``gguf``.
        logger.info(
            "Loading model from %s (quantization=%s) [stub]",
            self.model_path,
            self.quantization,
        )
        self.model = object()
        self._loaded = True

    async def unload(self) -> None:
        """Release model resources on application shutdown."""
        logger.info("Unloading inference model")
        self.model = None
        self._loaded = False

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Generate a completion for chat messages.

        Args:
            messages: OpenAI-style chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stream: When ``True``, return an async token iterator.

        Returns:
            Full completion text, or an async iterator of text chunks when streaming.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("inference model is not loaded")

        _ = (messages, max_tokens, temperature)
        if stream:

            async def _stream() -> AsyncIterator[str]:
                yield _DUMMY_RESPONSE

            return _stream()

        return _DUMMY_RESPONSE

    async def extract_invoice(self, document: str, language: str) -> ExtractResponse:
        """Extract and validate invoice JSON from a source document.

        Args:
            document: Raw document text.
            language: Language code (``en`` or ``hi``).

        Returns:
            Parsed, validated extraction response.
        """
        prompt = build_extraction_prompt(document, language)
        messages = [{"role": "user", "content": prompt}]
        raw_output = await self.generate(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            stream=False,
        )
        if not isinstance(raw_output, str):
            msg = "expected non-streaming generate() to return str"
            raise TypeError(msg)

        invoice: dict[str, Any] | None = None
        validation_errors: list[str] = []
        is_valid = False

        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                invoice = parsed
                is_valid, errors = validate_invoice(parsed)
                validation_errors = [
                    f"{err.get('source', 'unknown')}: {err.get('message', '')}" for err in errors
                ]
            else:
                validation_errors = ["parsed output is not a JSON object"]
        except json.JSONDecodeError as exc:
            validation_errors = [f"json decode error: {exc.msg}"]

        return ExtractResponse(
            invoice=invoice,
            raw_output=raw_output,
            is_valid=is_valid,
            validation_errors=validation_errors,
        )


_inference_service: InferenceService | None = None


def set_inference_service(service: InferenceService) -> None:
    """Register the global inference service instance for dependency injection."""
    global _inference_service
    _inference_service = service


def get_inference_service() -> InferenceService:
    """Return the active inference service for FastAPI dependencies.

    Returns:
        The configured ``InferenceService``.

    Raises:
        RuntimeError: If the service has not been initialized.
    """
    if _inference_service is None:
        raise RuntimeError("inference service not initialized")
    return _inference_service
