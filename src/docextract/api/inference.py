"""Inference service wrapper for the FastAPI application."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, cast

from docextract.api.models import ExtractResponse
from docextract.data.format_sft import build_extraction_prompt
from docextract.data.validation import validate_invoice
from docextract.eval.inference import InvoicePredictor, is_loadable_model, load_predictor

logger = logging.getLogger(__name__)

_DUMMY_RESPONSE = '{"invoice_number": "DUMMY-001"}'


class _RawPredictor(Protocol):
    """Predictor that can return raw model text alongside parsed JSON."""

    def predict_raw(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]: ...


class InferenceService:
    """Load and run the extraction model for API requests."""

    def __init__(self, model_path: Path, quantization: str = "none") -> None:
        """Initialize the service without loading weights.

        Args:
            model_path: Path to merged HF model directory, Hub model ID, or GGUF file.
            quantization: Serving backend (``none``, ``gguf``, ``awq``).
        """
        self.model: Any = None
        self.model_path = model_path
        self.quantization = quantization
        self._loaded = False
        self._predictor: InvoicePredictor | None = None
        self._local_files_only = False

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has completed startup loading."""
        return self._loaded

    async def load(self, *, local_files_only: bool = False) -> None:
        """Load model weights based on the configured quantization backend.

        Args:
            local_files_only: When ``True``, load HuggingFace weights from cache only.

        Raises:
            RuntimeError: If the model path does not exist and is not a Hub ID.
        """
        self._local_files_only = local_files_only
        if is_loadable_model(self.model_path):
            logger.info(
                "Loading inference model from %s (quantization=%s)",
                self.model_path.as_posix(),
                self.quantization,
            )
            self._predictor = load_predictor(
                self.model_path,
                local_files_only=local_files_only,
            )
            self.model = self._predictor
            self._loaded = True
            return

        if not self.model_path.exists():
            raise RuntimeError(f"model path not found: {self.model_path}")

        logger.info(
            "Loading model from %s (quantization=%s) [stub]",
            self.model_path,
            self.quantization,
        )
        self.model = object()
        self._predictor = None
        self._loaded = True

    async def unload(self) -> None:
        """Release model resources on application shutdown."""
        logger.info("Unloading inference model")
        self.model = None
        self._predictor = None
        self._loaded = False

    def _predict_raw(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Run raw generation when a real predictor is loaded."""
        if self._predictor is None:
            return _DUMMY_RESPONSE, {}
        predictor = cast(_RawPredictor, self._predictor)
        return predictor.predict_raw(record)

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

        if self._predictor is not None:
            document = ""
            language = "en"
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "user" and content:
                    document = content
            record = {"document": document, "language": language}
            raw_output, _parsed = self._predict_raw(record)
            return raw_output

        return _DUMMY_RESPONSE

    async def extract_invoice(self, document: str, language: str) -> ExtractResponse:
        """Extract and validate invoice JSON from a source document.

        Args:
            document: Raw document text.
            language: Language code (``en`` or ``hi``).

        Returns:
            Parsed, validated extraction response.
        """
        raw_output: str
        invoice: dict[str, Any] | None = None
        validation_errors: list[str] = []
        is_valid = False

        if self._predictor is not None:
            record = {"document": document, "language": language}
            raw_output, parsed = self._predict_raw(record)
            if parsed:
                invoice = parsed
                is_valid, errors = validate_invoice(parsed)
                validation_errors = [
                    f"{err.get('source', 'unknown')}: {err.get('message', '')}" for err in errors
                ]
            else:
                validation_errors = ["parsed output is not a valid invoice JSON object"]
            return ExtractResponse(
                invoice=invoice,
                raw_output=raw_output,
                is_valid=is_valid,
                validation_errors=validation_errors,
            )

        prompt = build_extraction_prompt(document, language)
        messages = [{"role": "user", "content": prompt}]
        generated = await self.generate(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            stream=False,
        )
        if not isinstance(generated, str):
            msg = "expected non-streaming generate() to return str"
            raise TypeError(msg)
        raw_output = generated

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
    """Return the active inference service for dependency injection.

    Returns:
        The configured ``InferenceService``.

    Raises:
        RuntimeError: If the service has not been initialized.
    """
    if _inference_service is None:
        raise RuntimeError("inference service not initialized")
    return _inference_service
