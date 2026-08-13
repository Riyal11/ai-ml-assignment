"""Extract SuperStore invoice PDFs into validated JSONL dataset splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docextract.data.dataset import DocumentRecord, Split
from docextract.data.superstore_extractor import (
    ParsedSuperstoreInvoice,
    parse_superstore_invoice_text,
    to_schema_target,
)
from docextract.data.validation import validate_invoice

logger = logging.getLogger(__name__)

DEFAULT_PDF_DIR = Path("data/raw/superstore")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_GOLDEN_SIZE = 50
DEFAULT_BENCHMARK_SIZE = 20
DEFAULT_VALIDATION_RATIO = 0.1


@dataclass
class ExtractionResult:
    """One extracted PDF and its validation metadata."""

    example_id: str
    source_pdf: str
    document: str
    parsed: ParsedSuperstoreInvoice
    target: dict[str, Any]
    split: Split
    schema_valid: bool
    validation_errors: list[dict[str, Any]]


def extract_pdf_content(pdf_path: Path) -> tuple[str, list[list[list[str | None]]]]:
    """Return plain text and tables from a PDF file."""
    try:
        import pdfplumber
    except ImportError as exc:
        msg = "pdfplumber is required. Install with: uv pip install pdfplumber"
        raise RuntimeError(msg) from exc

    text_parts: list[str] = []
    tables: list[list[list[str | None]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            for table in page.extract_tables() or []:
                tables.append(table)
    return "\n".join(text_parts), tables


def assign_splits(
    results: list[ExtractionResult],
    *,
    golden_size: int,
    benchmark_size: int,
    validation_ratio: float,
) -> None:
    """Assign train/validation/golden/benchmark splits deterministically."""
    ordered = sorted(results, key=lambda item: item.parsed.invoice_number)
    remaining_start = golden_size + benchmark_size
    remaining = ordered[remaining_start:]
    validation_count = int(len(remaining) * validation_ratio) if remaining else 0

    for idx, result in enumerate(ordered):
        if idx < golden_size:
            result.split = Split.GOLDEN
        elif idx < golden_size + benchmark_size:
            result.split = Split.BENCHMARK
        elif idx < remaining_start + validation_count:
            result.split = Split.VALIDATION
        else:
            result.split = Split.TRAIN


def record_from_result(result: ExtractionResult) -> DocumentRecord:
    """Convert an extraction result into a ``DocumentRecord``."""
    return DocumentRecord(
        example_id=result.example_id,
        document=result.document,
        target=result.target,
        language="en",
        split=result.split,
    )


def write_split_jsonl(results: list[ExtractionResult], output_dir: Path) -> dict[str, int]:
    """Write one JSONL file per split under ``output_dir``."""
    counts = {split.value: 0 for split in Split}
    buckets: dict[Split, list[ExtractionResult]] = {split: [] for split in Split}
    for result in results:
        buckets[result.split].append(result)

    for split, bucket in buckets.items():
        split_dir = output_dir / split.value
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "invoices.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for result in bucket:
                record = record_from_result(result)
                line = {
                    "example_id": record.example_id,
                    "document": record.document,
                    "target": record.target,
                    "language": record.language,
                    "split": record.split.value,
                }
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                counts[split.value] += 1
        logger.info("Wrote %d records to %s", len(bucket), out_path)
    return counts


def write_manifest(
    results: list[ExtractionResult],
    *,
    pdf_dir: Path,
    output_dir: Path,
    split_counts: dict[str, int],
    golden_size: int,
    benchmark_size: int,
    validation_ratio: float,
) -> Path:
    """Write a manifest summarizing extraction and validation."""
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source": "superstore_pdfs",
        "pdf_dir": str(pdf_dir),
        "output_dir": str(output_dir),
        "split_config": {
            "golden_size": golden_size,
            "benchmark_size": benchmark_size,
            "validation_ratio": validation_ratio,
        },
        "counts": split_counts,
        "records": [
            {
                "example_id": result.example_id,
                "source_pdf": result.source_pdf,
                "invoice_number": result.parsed.invoice_number,
                "order_id": result.parsed.order_id,
                "customer_name": result.parsed.customer_name,
                "discount_percent": (
                    float(result.parsed.discount.percent) if result.parsed.discount else None
                ),
                "discount_amount": (
                    float(result.parsed.discount.amount) if result.parsed.discount else None
                ),
                "shipping": float(result.parsed.shipping),
                "total_validation_warning": result.parsed.total_validation_warning,
                "schema_valid": result.schema_valid,
                "validation_errors": result.validation_errors,
                "split": result.split.value,
            }
            for result in results
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest to %s", manifest_path)
    return manifest_path


def extract_superstore_invoices(
    pdf_dir: Path,
    output_dir: Path,
    *,
    golden_size: int = DEFAULT_GOLDEN_SIZE,
    benchmark_size: int = DEFAULT_BENCHMARK_SIZE,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> list[ExtractionResult]:
    """Extract all PDFs in ``pdf_dir`` and write JSONL splits plus manifest."""
    if not pdf_dir.is_dir():
        logger.warning("PDF directory does not exist: %s", pdf_dir)
        write_manifest(
            [],
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            split_counts={split.value: 0 for split in Split},
            golden_size=golden_size,
            benchmark_size=benchmark_size,
            validation_ratio=validation_ratio,
        )
        return []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        write_manifest(
            [],
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            split_counts={split.value: 0 for split in Split},
            golden_size=golden_size,
            benchmark_size=benchmark_size,
            validation_ratio=validation_ratio,
        )
        return []

    results: list[ExtractionResult] = []
    for pdf_path in pdf_files:
        try:
            document, tables = extract_pdf_content(pdf_path)
            parsed = parse_superstore_invoice_text(document, tables)
            target = to_schema_target(parsed)
            valid, errors = validate_invoice(target)
            if not valid:
                logger.warning(
                    "Schema validation failed for %s: %s",
                    pdf_path.name,
                    errors,
                )
            example_id = f"superstore-{parsed.invoice_number}"
            results.append(
                ExtractionResult(
                    example_id=example_id,
                    source_pdf=pdf_path.name,
                    document=document,
                    parsed=parsed,
                    target=target,
                    split=Split.TRAIN,
                    schema_valid=valid,
                    validation_errors=errors,
                )
            )
        except Exception:
            logger.exception("Failed to extract %s", pdf_path.name)
            continue

    if not results:
        logger.warning("No invoices were successfully extracted from %s", pdf_dir)
        write_manifest(
            [],
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            split_counts={split.value: 0 for split in Split},
            golden_size=golden_size,
            benchmark_size=benchmark_size,
            validation_ratio=validation_ratio,
        )
        return []

    assign_splits(
        results,
        golden_size=golden_size,
        benchmark_size=benchmark_size,
        validation_ratio=validation_ratio,
    )
    split_counts = write_split_jsonl(results, output_dir)
    write_manifest(
        results,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        split_counts=split_counts,
        golden_size=golden_size,
        benchmark_size=benchmark_size,
        validation_ratio=validation_ratio,
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract SuperStore invoice PDFs into JSONL dataset splits",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing SuperStore PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dataset output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--golden-size", type=int, default=DEFAULT_GOLDEN_SIZE)
    parser.add_argument("--benchmark-size", type=int, default=DEFAULT_BENCHMARK_SIZE)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Fraction of non-held-out invoices assigned to validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    results = extract_superstore_invoices(
        args.pdf_dir,
        args.output_dir,
        golden_size=args.golden_size,
        benchmark_size=args.benchmark_size,
        validation_ratio=args.validation_ratio,
    )
    logger.info("Extracted %d invoices", len(results))
    return 0 if results or not args.pdf_dir.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
