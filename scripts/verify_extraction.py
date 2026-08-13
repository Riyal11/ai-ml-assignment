"""Verify extracted invoice JSONL records against schema validators."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from docextract.data.validation import validate_invoice

logger = logging.getLogger(__name__)
DEFAULT_JSONL = Path("data/train/invoices.jsonl")


def verify_jsonl(path: Path) -> int:
    """Validate all targets in a JSONL file.

    Returns:
        Number of invalid records found.
    """
    if not path.is_file():
        logger.error("JSONL file not found: %s", path)
        return 0

    invalid_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            is_valid, errors = validate_invoice(record["target"])
            if not is_valid:
                invalid_count += 1
                example_id = record.get("example_id", f"line-{line_no}")
                logger.warning("INVALID: %s", example_id)
                for err in errors:
                    logger.warning("  - %s", err["message"])
    return invalid_count


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Verify invoice extraction JSONL")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"Path to JSONL file (default: {DEFAULT_JSONL})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    invalid = verify_jsonl(args.jsonl)
    if invalid:
        logger.error("Found %d invalid record(s)", invalid)
        return 1
    logger.info("All records valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
