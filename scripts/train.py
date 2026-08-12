"""CLI entrypoint for fine-tuning runs."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from docextract.train.run_config import RunConfig, run_config_to_dict, validate_mandatory_fields
from docextract.train.trainer import run_training

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for training runs."""
    parser = argparse.ArgumentParser(description="Fine-tune docextract with LoRA/QLoRA/DoRA")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML run config")
    parser.add_argument("--method", choices=["lora", "qlora", "dora"], default=None)
    parser.add_argument("--r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=float, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument(
        "--max-seq-length", type=int, default=None, help="Max token sequence length"
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="LoRA target modules (space-separated, e.g. q_proj k_proj v_proj)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--base-model", type=str, default=None, help="Base model ID")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL dataset path (messages format)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Adapter output directory")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping")
    return data


def _apply_overrides(base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Overlay CLI arguments onto a base config dict."""
    overrides: dict[str, Any] = {}
    if args.method is not None:
        overrides["method"] = args.method
    if args.r is not None:
        overrides["r"] = args.r
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["gradient_accumulation_steps"] = args.grad_accum
    if args.max_seq_length is not None:
        overrides["max_seq_length"] = args.max_seq_length
    if args.target_modules is not None:
        overrides["target_modules"] = args.target_modules
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    return {**base, **overrides}


def main(argv: list[str] | None = None) -> int:
    """Parse args, build a validated ``RunConfig``, and run training."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    base: dict[str, Any] = _load_yaml(args.config) if args.config else {}
    config_dict = _apply_overrides(base, args)
    config = RunConfig(**config_dict)
    validate_mandatory_fields(config)

    logger.info("Run ID: %s", config.run_id)
    logger.info("Method: %s", config.method)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Run config: %s", json.dumps(run_config_to_dict(config), indent=2, default=str))

    try:
        out_path = run_training(config, args.dataset)
        logger.info("Status: succeeded (adapter at %s)", out_path)
        return 0
    except Exception:
        logger.exception("Training run %s failed", config.run_id)
        logger.info("Status: failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
