"""Convert extraction JSONL to SFT messages JSONL for training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from docextract.data.format_sft import convert_extraction_jsonl_to_sft

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/train/invoices.jsonl")
DEFAULT_OUTPUT = Path("data/train/sft.jsonl")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert extraction JSONL (document/target) to SFT messages JSONL",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source extraction JSONL (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination SFT JSONL (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if not args.input.is_file():
        logger.error("Input file not found: %s", args.input)
        return 1

    count = convert_extraction_jsonl_to_sft(args.input, args.output)
    logger.info("Prepared %d SFT examples at %s", count, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
