"""Generate a human-review batch JSON file from golden-set examples."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from docextract.api.inference import InferenceService

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_DATASET = Path("data/golden/invoices.jsonl")
_DEFAULT_OUTPUT = Path("docs/human_review_batch.json")
_PREVIEW_CHARS = 240


def _load_records(dataset_path: Path, limit: int) -> list[dict[str, Any]]:
    """Load up to ``limit`` JSONL records from a golden dataset.

    Args:
        dataset_path: Path to the golden JSONL file.
        limit: Maximum number of records to load.

    Returns:
        Parsed record dicts in file order.
    """
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def _document_preview(document: str, max_chars: int = _PREVIEW_CHARS) -> str:
    """Return a single-line preview of the source document text."""
    collapsed = " ".join(document.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


async def _run_extractions(
    service: InferenceService,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run ``extract_invoice`` for each golden record.

    Args:
        service: Loaded inference service with the base model.
        records: Golden-set records with ``document``, ``target``, and metadata.

    Returns:
        Review batch rows ready for JSON serialization.
    """
    batch: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        example_id = str(record["example_id"])
        language = str(record.get("language", "en"))
        document = str(record["document"])
        gold = record["target"]
        logger.info("Extracting %s (%d/%d)", example_id, index, len(records))
        response = await service.extract_invoice(document, language)
        batch.append(
            {
                "example_id": example_id,
                "document_preview": _document_preview(document),
                "gold_json": gold,
                "prediction_json": response.invoice,
                "raw_output": response.raw_output,
                "is_valid": response.is_valid,
                "validation_errors": response.validation_errors,
            }
        )
    return batch


async def generate_human_review_batch(
    *,
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int = 20,
    local_files_only: bool = False,
) -> int:
    """Generate the human review batch file.

    Args:
        model_path: HuggingFace Hub ID or local model path.
        dataset_path: Golden JSONL source.
        output_path: Destination JSON array path.
        limit: Number of examples to include.
        local_files_only: Load model weights from cache only when set.

    Returns:
        Number of examples written.
    """
    records = _load_records(dataset_path, limit)
    if not records:
        msg = f"no records found in {dataset_path}"
        raise ValueError(msg)

    service = InferenceService(model_path=model_path, quantization="none")
    await service.load(local_files_only=local_files_only)
    batch = await _run_extractions(service, records)
    await service.unload()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(batch)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate human review batch JSON")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(_DEFAULT_MODEL),
        help=f"Base model Hub ID or path (default: {_DEFAULT_MODEL})",
    )
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
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of examples to include (default: 20)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HuggingFace weights from local cache only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        count = asyncio.run(
            generate_human_review_batch(
                model_path=args.model_path,
                dataset_path=args.dataset,
                output_path=args.output,
                limit=args.limit,
                local_files_only=args.local_files_only,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("Human review batch generation failed")
        return 1

    logger.info("Wrote %d review examples to %s", count, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
