"""Generate synthetic Hindi invoice examples for evaluation only.

Does not modify training data. Output is written to ``hindi_eval.jsonl`` under
the configured golden directory plus a manifest at ``data/hindi_eval_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from docextract.data.validation import validate_invoice

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("data/golden")
_DEFAULT_OUTPUT_NAME = "hindi_eval.jsonl"
_DEFAULT_MANIFEST = Path("data/hindi_eval_manifest.json")
_TOTAL_EXAMPLES = 50
_MAX_REGEN_ATTEMPTS = 25
_TAX_RATE = Decimal("0.18")

_VENDORS: tuple[str, ...] = (
    "भारत स्टोर्स",
    "दिल्ली ट्रेडर्स",
    "मुंबई सप्लायर्स",
    "चेन्नई मार्ट",
    "कोलकाता डिपो",
    "पुणे इंटरप्राइजेज",
    "अहमदाबाद ट्रेडिंग",
)

_PRODUCTS: tuple[str, ...] = (
    "ऑफिस चेयर",
    "फाइल कैबिनेट",
    "लैपटॉप बैग",
    "प्रिंटर",
    "डेस्क लैंप",
    "मार्कर पेन",
    "नोटबुक",
    "कैलकुलेटर",
    "माउस पैड",
    "कीबोर्ड",
)

_HINDI_MONTHS: tuple[tuple[str, int], ...] = (
    ("जनवरी", 1),
    ("फरवरी", 2),
    ("मार्च", 3),
    ("अप्रैल", 4),
    ("मई", 5),
    ("जून", 6),
    ("जुलाई", 7),
    ("अगस्त", 8),
    ("सितंबर", 9),
    ("अक्टूबर", 10),
    ("नवंबर", 11),
    ("दिसंबर", 12),
)

_INVOICE_LABELS: tuple[str, ...] = (
    "बिल क्रमांक",
    "इनवॉइस नंबर",
    "चालान संख्या",
    "बिल नंबर",
)

_DATE_LABELS: tuple[str, ...] = ("दिनांक", "तारीख")

_ITEM_LABELS: tuple[str, ...] = ("वस्तु", "सामान")

_QTY_LABELS: tuple[str, ...] = ("मात्रा", "संख्या")

_PRICE_LABELS: tuple[str, ...] = ("दर", "मूल्य")

_TOTAL_LABELS: tuple[str, ...] = ("कुल", "योग")


@dataclass(frozen=True)
class HindiDate:
    """A Hindi calendar date with ISO normalization."""

    day: int
    month_name: str
    month: int
    year: int

    @property
    def iso(self) -> str:
        """Return the date as ``YYYY-MM-DD``."""
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    @property
    def hindi_text(self) -> str:
        """Return the date in Hindi day-month-year form."""
        return f"{self.day} {self.month_name} {self.year}"


@dataclass(frozen=True)
class GeneratedLineItem:
    """One invoice line item with computed extension."""

    description: str
    quantity: int
    unit_price: Decimal

    @property
    def extension(self) -> Decimal:
        """Line total before tax."""
        return self.unit_price * Decimal(self.quantity)


def _money_quantize(value: Decimal) -> Decimal:
    """Round money to two decimal places."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _format_inr(amount: Decimal) -> str:
    """Format a rupee amount for display in invoice text."""
    normalized = _money_quantize(amount)
    if normalized == normalized.to_integral_value():
        return f"₹{int(normalized)}"
    return f"₹{normalized}"


def _random_unit_price(rng: random.Random) -> Decimal:
    """Pick a plausible unit price, sometimes with paise."""
    base = Decimal(rng.randint(150, 8500))
    if rng.random() < 0.35:
        return _money_quantize(base + Decimal("0.50"))
    return base


def _pick_date(rng: random.Random) -> HindiDate:
    """Sample a Hindi date from fixed anchors and random valid calendar days."""
    anchors: tuple[tuple[int, str, int, int], ...] = (
        (15, "मार्च", 3, 2024),
        (22, "अप्रैल", 4, 2024),
        (5, "जनवरी", 1, 2024),
        (30, "दिसंबर", 12, 2023),
        (1, "नवंबर", 11, 2024),
        (8, "जून", 6, 2024),
        (19, "अगस्त", 8, 2024),
        (3, "सितंबर", 9, 2024),
    )
    if rng.random() < 0.45:
        day, month_name, month, year = rng.choice(anchors)
        return HindiDate(day=day, month_name=month_name, month=month, year=year)

    month_name, month = rng.choice(_HINDI_MONTHS)
    year = rng.choice((2023, 2024, 2024, 2024))
    day = rng.randint(1, 28)
    return HindiDate(day=day, month_name=month_name, month=month, year=year)


def _build_line_items(rng: random.Random, count: int) -> list[GeneratedLineItem]:
    """Create ``count`` unique product line items."""
    products = rng.sample(_PRODUCTS, k=count)
    items: list[GeneratedLineItem] = []
    for product in products:
        quantity = rng.randint(1, 12)
        unit_price = _random_unit_price(rng)
        items.append(
            GeneratedLineItem(
                description=product,
                quantity=quantity,
                unit_price=unit_price,
            )
        )
    return items


def _compute_totals(items: list[GeneratedLineItem]) -> tuple[Decimal, Decimal, Decimal]:
    """Compute subtotal, tax, and total from line items."""
    subtotal = _money_quantize(sum((item.extension for item in items), Decimal("0")))
    tax_amount = _money_quantize(subtotal * _TAX_RATE)
    total_amount = _money_quantize(subtotal + tax_amount)
    return subtotal, tax_amount, total_amount


def _target_from_items(
    invoice_number: str,
    vendor_name: str,
    invoice_date: str,
    items: list[GeneratedLineItem],
    subtotal: Decimal,
    tax_amount: Decimal,
    total_amount: Decimal,
) -> dict[str, Any]:
    """Build a schema-compatible target dict."""
    return {
        "invoice_number": invoice_number,
        "vendor_name": vendor_name,
        "invoice_date": invoice_date,
        "line_items": [
            {
                "description": item.description,
                "quantity": item.quantity,
                "unit_price": float(item.unit_price),
            }
            for item in items
        ],
        "subtotal": float(subtotal),
        "tax_amount": float(tax_amount),
        "total_amount": float(total_amount),
        "currency": "INR",
    }


def _render_line_block(
    rng: random.Random,
    index: int,
    item: GeneratedLineItem,
) -> str:
    """Render one line-item block in natural Hindi."""
    item_label = rng.choice(_ITEM_LABELS)
    qty_label = rng.choice(_QTY_LABELS)
    price_label = rng.choice(_PRICE_LABELS)
    prefix = f"{item_label} {index}: " if index > 1 else f"{item_label}: "
    return (
        f"{prefix}{item.description}, {qty_label}: {item.quantity}, "
        f"{price_label}: {_format_inr(item.unit_price)}"
    )


def _render_document(
    rng: random.Random,
    invoice_number: str,
    vendor_name: str,
    hindi_date: HindiDate,
    items: list[GeneratedLineItem],
    subtotal: Decimal,
    tax_amount: Decimal,
    total_amount: Decimal,
    *,
    complex_narrative: bool,
) -> str:
    """Render realistic Devanagari invoice text."""
    invoice_label = rng.choice(_INVOICE_LABELS)
    date_label = rng.choice(_DATE_LABELS)
    total_label = rng.choice(_TOTAL_LABELS)

    lines = [
        f"{invoice_label}: {invoice_number}",
        f"विक्रेता: {vendor_name}",
        f"{date_label}: {hindi_date.hindi_text}",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(_render_line_block(rng, index, item))

    if complex_narrative:
        discount = _money_quantize(Decimal(rng.randint(100, 900)))
        shipping = _money_quantize(Decimal(rng.randint(80, 450)))
        lines.append(f"छूट: {_format_inr(discount)} (विशेष प्रस्ताव)")
        lines.append(f"शिपिंग शुल्क: {_format_inr(shipping)}")

    lines.append(
        f"{total_label}: {_format_inr(subtotal)} + कर: {_format_inr(tax_amount)} = "
        f"{_format_inr(total_amount)} INR"
    )
    lines.append("भुगतान शर्तें: 15 दिनों में")
    return "\n".join(lines)


def _line_item_count_for_index(index: int) -> int:
    """Return line-item count by complexity bucket for example ``index`` (1-based)."""
    if index <= 20:
        return 1
    if index <= 40:
        return 2
    return 3


def _generate_example(rng: random.Random, index: int) -> dict[str, Any]:
    """Generate one evaluation record."""
    example_id = f"hi-eval-{index:03d}"
    invoice_number = f"INV-H-E{index:03d}"
    vendor_name = rng.choice(_VENDORS)
    hindi_date = _pick_date(rng)
    line_count = _line_item_count_for_index(index)
    items = _build_line_items(rng, line_count)
    subtotal, tax_amount, total_amount = _compute_totals(items)
    target = _target_from_items(
        invoice_number,
        vendor_name,
        hindi_date.iso,
        items,
        subtotal,
        tax_amount,
        total_amount,
    )
    document = _render_document(
        rng,
        invoice_number,
        vendor_name,
        hindi_date,
        items,
        subtotal,
        tax_amount,
        total_amount,
        complex_narrative=line_count == 3,
    )
    return {
        "example_id": example_id,
        "document": document,
        "target": target,
        "language": "hi",
        "split": "golden",
    }


def _generate_valid_example(rng: random.Random, index: int) -> dict[str, Any]:
    """Generate an example, regenerating until validation passes."""
    for attempt in range(1, _MAX_REGEN_ATTEMPTS + 1):
        record = _generate_example(rng, index)
        is_valid, errors = validate_invoice(record["target"])
        if is_valid:
            return record
        logger.error(
            "Validation failed for %s (attempt %d/%d): %s",
            record["example_id"],
            attempt,
            _MAX_REGEN_ATTEMPTS,
            errors,
        )
    msg = f"failed to generate valid Hindi eval example for index {index}"
    raise RuntimeError(msg)


def generate_hindi_eval_dataset(
    rng: random.Random,
    total: int = _TOTAL_EXAMPLES,
) -> list[dict[str, Any]]:
    """Generate ``total`` validated Hindi evaluation records."""
    return [_generate_valid_example(rng, index) for index in range(1, total + 1)]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write the Hindi eval manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_generated": len(records),
        "split": "golden",
        "sample_example_ids": [
            records[0]["example_id"],
            records[24]["example_id"],
            records[-1]["example_id"],
        ],
        "notes": (
            "Evaluation only. Model trained on English data; Hindi capability relies "
            "on Qwen3-4B base model + schema transfer."
        ),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Hindi invoices for evaluation only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for hindi_eval.jsonl (default: data/golden/)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path for hindi_eval_manifest.json (default: data/hindi_eval_manifest.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=_TOTAL_EXAMPLES,
        help="Number of examples to generate (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate Hindi eval JSONL and manifest."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    rng = random.Random(args.seed)  # noqa: S311
    records = generate_hindi_eval_dataset(rng, total=args.count)

    output_path = args.output_dir / _DEFAULT_OUTPUT_NAME
    write_jsonl(records, output_path)
    write_manifest(records, args.manifest_path)

    logger.info("Wrote %d examples to %s", len(records), output_path)
    logger.info("Wrote manifest to %s", args.manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
