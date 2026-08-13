"""CLI entrypoint for golden-set / benchmark evaluation."""

import argparse
import logging
import sys
from pathlib import Path

from docextract.data.dataset import Split
from docextract.eval.pipeline import run_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an invoice extraction model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to PEFT adapter directory (or artifacts/run-NNN/final)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL evaluation dataset with document/target fields",
    )
    parser.add_argument(
        "--split",
        choices=[member.value for member in Split],
        required=True,
        help="Split label recorded in results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write results.json",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Force stub predictions (no model load)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load base model/tokenizer from HF cache only (no Hub API calls)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out_path = run_evaluation(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        split=Split(args.split),
        use_stub=args.stub,
        local_files_only=args.local_files_only,
    )
    logging.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
