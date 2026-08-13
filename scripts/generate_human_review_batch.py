"""Generate a human-review batch JSONL from golden-set examples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from docextract.eval.inference import load_predictor

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = Path("data/golden/invoices.jsonl")
_DEFAULT_OUTPUT = Path("docs/human_review_batch.jsonl")


def _load_records(dataset_path: Path, limit: int) -> list[dict[str, Any]]:
    """Load up to ``limit`` JSONL records from a golden dataset."""
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def generate_review_batch(
    dataset_path: Path,
    output_path: Path,
    *,
    limit: int = 20,
    model_path: Path | None = None,
    local_files_only: bool = False,
) -> int:
    """Write a human-review batch with gold targets and optional predictions.

    Args:
        dataset_path: Golden JSONL source.
        output_path: Destination JSONL for reviewers.
        limit: Maximum number of examples to include.
        model_path: Optional adapter path to attach model predictions.
        local_files_only: Pass through to the HF predictor when set.

    Returns:
        Number of records written.
    """
    records = _load_records(dataset_path, limit)
    predictor = None
    if model_path is not None:
        logger.info("Loading predictor from %s", model_path)
        predictor = load_predictor(model_path, local_files_only=local_files_only)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row: dict[str, Any] = {
                "example_id": record["example_id"],
                "language": record.get("language", "en"),
                "document": record["document"],
                "gold": record["target"],
            }
            if predictor is not None:
                row["prediction"] = predictor.predict(record)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(records)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate human review batch JSONL")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"Golden JSONL source (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Review batch output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples to include (default: 20)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional adapter path to attach model predictions",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model from HF cache only when --model-path is set",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    count = generate_review_batch(
        args.dataset,
        args.output,
        limit=args.limit,
        model_path=args.model_path,
        local_files_only=args.local_files_only,
    )
    logger.info("Wrote %d review examples to %s", count, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
