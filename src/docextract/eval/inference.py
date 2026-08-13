"""Hugging Face + PEFT inference for invoice extraction evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from docextract.data.format_sft import build_inference_messages
from docextract.data.tokenizer_utils import apply_chat_template, load_tokenizer
from docextract.eval.schema_validator import validate_output_text

logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 1024


class InvoicePredictor(Protocol):
    """Protocol for per-record invoice JSON prediction."""

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a parsed invoice dict (possibly empty on failure)."""
        ...


def resolve_adapter_path(model_path: Path) -> Path:
    """Resolve a model directory that contains ``adapter_config.json``.

    Accepts either the adapter root (``artifacts/run-001``) or a conventional
    ``final`` subdirectory when the trainer wrote adapters one level up.
    """
    if (model_path / "adapter_config.json").is_file():
        return model_path
    parent = model_path.parent
    if (parent / "adapter_config.json").is_file():
        logger.warning(
            "Adapter not found at %s; using parent directory %s",
            model_path,
            parent,
        )
        return parent
    return model_path


class HfInvoicePredictor:
    """Run QLoRA/LoRA adapter inference with 4-bit base weights."""

    def __init__(
        self,
        model_path: Path,
        *,
        base_model: str | None = None,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        local_files_only: bool = False,
    ) -> None:
        adapter_path = resolve_adapter_path(model_path)
        adapter_config_path = adapter_path / "adapter_config.json"
        if not adapter_config_path.is_file():
            msg = f"no adapter_config.json under {model_path} (resolved {adapter_path})"
            raise FileNotFoundError(msg)

        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        resolved_base = base_model or str(adapter_config["base_model_name_or_path"])
        logger.info("Loading base model %s with adapter from %s", resolved_base, adapter_path)

        bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device_map: dict[str, int] | str = {"": 0} if torch.cuda.is_available() else "auto"
        load_kwargs: dict[str, Any] = {
            "quantization_config": bnb_config,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16,
            "local_files_only": local_files_only,
        }
        base = AutoModelForCausalLM.from_pretrained(resolved_base, **load_kwargs)
        self._model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._tokenizer = load_tokenizer(resolved_base, local_files_only=local_files_only)
        logger.info("Model ready; starting inference")
        self._max_new_tokens = max_new_tokens

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Generate and parse invoice JSON for one evaluation record."""
        messages = build_inference_messages(
            str(record.get("document", "")),
            str(record.get("language", "en")),
        )
        prompt = apply_chat_template(
            messages,
            self._tokenizer,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        _valid, parsed, errors = validate_output_text(text)
        if parsed is None:
            logger.debug("Prediction parse failed: %s", errors)
            return {}
        return parsed


class StubInvoicePredictor:
    """Stub predictor that always returns an empty dict."""

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        _ = record
        return {}


def load_predictor(
    model_path: Path,
    *,
    use_stub: bool = False,
    local_files_only: bool = False,
) -> InvoicePredictor:
    """Load a real HF predictor or a stub when ``use_stub`` is True."""
    if use_stub:
        return StubInvoicePredictor()
    return HfInvoicePredictor(model_path, local_files_only=local_files_only)
