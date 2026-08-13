"""CLI to merge a PEFT adapter into a base Hugging Face model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from docextract.data.tokenizer_utils import load_tokenizer
from docextract.eval.inference import resolve_adapter_path

logger = logging.getLogger(__name__)

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_OUTPUT_DIR = Path("artifacts/merged")


def merge_adapter(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    local_files_only: bool = False,
) -> Path:
    """Merge a LoRA/QLoRA adapter into the base weights and save full model.

    Args:
        base_model: Hugging Face model ID or local path for the base weights.
        adapter_path: Directory containing ``adapter_config.json``.
        output_dir: Destination directory for the merged model + tokenizer.
        device: ``auto``, ``cpu``, or ``cuda`` load device map strategy.
        local_files_only: Load only from the local Hugging Face cache.

    Returns:
        Path to the output directory.

    Raises:
        FileNotFoundError: If the adapter directory is missing.
    """
    resolved_adapter = resolve_adapter_path(adapter_path)
    adapter_config_path = resolved_adapter / "adapter_config.json"
    if not adapter_config_path.is_file():
        msg = f"adapter_config.json not found under {adapter_path}"
        raise FileNotFoundError(msg)

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    resolved_base = str(adapter_config.get("base_model_name_or_path", base_model))

    if device == "cpu":
        device_map: str | dict[str, str] = "cpu"
        torch_dtype = torch.bfloat16
    elif device == "cuda":
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = "auto"
        torch_dtype = torch.bfloat16

    logger.info("Loading base model %s (device=%s)", resolved_base, device)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )
    logger.info("Loading adapter from %s", resolved_adapter)
    model = PeftModel.from_pretrained(
        model,
        str(resolved_adapter),
        local_files_only=local_files_only,
    )
    logger.info("Merging adapter into base weights")
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s", output_dir)
    merged.save_pretrained(output_dir)

    tokenizer = load_tokenizer(resolved_base, local_files_only=local_files_only)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a base model")
    parser.add_argument(
        "base_model",
        nargs="?",
        default=_DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {_DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "adapter_path",
        type=Path,
        help="PEFT adapter directory (e.g. artifacts/run-001)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for merge load (use cpu while GPU training is running)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load base model/tokenizer from HF cache only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Merge adapter CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        out_path = merge_adapter(
            args.base_model,
            args.adapter_path,
            args.output_dir,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        logger.info("Merged model saved to %s", out_path)
        return 0
    except FileNotFoundError:
        logger.exception("Merge failed")
        return 1
    except Exception:
        logger.exception("Merge failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
