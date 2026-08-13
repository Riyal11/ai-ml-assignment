"""Tests for SuperStore invoice extraction."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.data.dataset import Split
from docextract.data.superstore_extractor import (
    normalize_invoice_date,
    parse_line_items_from_table,
    parse_money,
    parse_superstore_invoice_text,
    to_schema_target,
    validate_total_amount,
)
from docextract.data.validation import validate_invoice
from scripts.extract_superstore_invoices import (
    assign_splits,
    extract_superstore_invoices,
    write_manifest,
    write_split_jsonl,
)

SAMPLE_INVOICE_WITH_DISCOUNT = """
SuperStore
INVOICE #25880
Date: Nov172012

Bill To: Claire Gute
Minneapolis, MN

Item    Quantity    Rate    Amount
Staples 3           $4.17   $12.51
Paper   2           $15.00  $30.00

Subtotal: $42.51
Discount (20%): $8.50
Shipping: $5.00
Total: $39.01
Order ID: US-2012-25880
"""

SAMPLE_INVOICE_HASH_STYLE = """
SuperStore
# 36258
Order Date Mar 06 2012

BillTo: Sean Miller
Denver, CO

Product Name    Quantity    Unit Cost    Amount
Chairs          4           $120.00      $480.00

Subtotal: $480.00
Shipping: $12.45
Balance Due: $492.45
Order ID: CA-2012-36258
"""

SAMPLE_INVOICE_ROW_ID = """
SuperStore
Row ID: 99112
Date: Mar 6 2012

Bill To: Tamara Chand
Indianapolis, IN

Product Name    Qty    Rate/Unit Cost    Amount
Binders         10     $2.50             $25.00

Subtotal: $25.00
Total Amount Payable: $25.00
Order ID: IN-2012-99112
"""


def test_parse_money_strips_currency_symbols() -> None:
    assert parse_money("$1,880.45") == Decimal("1880.45")
    assert parse_money("$319.68") == Decimal("319.68")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Nov172012", "2012-11-17"),
        ("Mar 06 2012", "2012-03-06"),
        ("Mar 6 2012", "2012-03-06"),
    ],
)
def test_normalize_invoice_date(raw: str, expected: str) -> None:
    assert normalize_invoice_date(raw) == expected


def test_parse_invoice_with_discount_and_bill_to_variation() -> None:
    parsed = parse_superstore_invoice_text(SAMPLE_INVOICE_WITH_DISCOUNT)
    assert parsed.invoice_number == "25880"
    assert parsed.invoice_date == "2012-11-17"
    assert parsed.customer_name.startswith("Claire Gute")
    assert parsed.order_id == "US-2012-25880"
    assert parsed.discount is not None
    assert parsed.discount.percent == Decimal("20")
    assert parsed.discount.amount == Decimal("8.50")
    assert len(parsed.line_items) == 2
    assert parsed.total_validation_warning is None

    target = to_schema_target(parsed)
    valid, errors = validate_invoice(target)
    assert valid, errors
    assert target["tax_amount"] == 0.0
    assert target["currency"] == "USD"


def test_parse_invoice_hash_and_order_date_variation() -> None:
    parsed = parse_superstore_invoice_text(SAMPLE_INVOICE_HASH_STYLE)
    assert parsed.invoice_number == "36258"
    assert parsed.invoice_date == "2012-03-06"
    assert parsed.customer_name.startswith("Sean Miller")
    assert parsed.discount is None
    assert parsed.total_amount == Decimal("492.45")


def test_parse_invoice_row_id_and_total_label_variation() -> None:
    parsed = parse_superstore_invoice_text(SAMPLE_INVOICE_ROW_ID)
    assert parsed.invoice_number == "99112"
    assert parsed.total_amount == Decimal("25.00")
    assert parsed.order_id == "IN-2012-99112"


def test_validate_total_amount_logs_mismatch() -> None:
    warning = validate_total_amount(
        subtotal=Decimal("100.00"),
        discount=None,
        shipping=Decimal("5.00"),
        total_amount=Decimal("90.00"),
    )
    assert warning is not None
    assert "computed total" in warning


def test_parse_line_items_from_table() -> None:
    table = [
        ["Product Name", "Quantity", "Rate", "Amount"],
        ["Staples", "3", "$4.17", "$12.51"],
        ["Paper", "2", "$15.00", "$30.00"],
    ]
    items = parse_line_items_from_table(table)
    assert len(items) == 2
    assert items[0]["description"] == "Staples"
    assert items[0]["quantity"] == 3
    assert items[0]["unit_price"] == 4.17


def test_assign_splits_is_deterministic() -> None:
    from scripts.extract_superstore_invoices import ExtractionResult

    parsed_a = parse_superstore_invoice_text(SAMPLE_INVOICE_HASH_STYLE)
    parsed_b = parse_superstore_invoice_text(SAMPLE_INVOICE_WITH_DISCOUNT)
    results = [
        ExtractionResult(
            example_id="superstore-36258",
            source_pdf="a.pdf",
            document="doc-a",
            parsed=parsed_a,
            target=to_schema_target(parsed_a),
            split=Split.TRAIN,
            schema_valid=True,
            validation_errors=[],
        ),
        ExtractionResult(
            example_id="superstore-25880",
            source_pdf="b.pdf",
            document="doc-b",
            parsed=parsed_b,
            target=to_schema_target(parsed_b),
            split=Split.TRAIN,
            schema_valid=True,
            validation_errors=[],
        ),
    ]
    assign_splits(results, golden_size=1, benchmark_size=1, validation_ratio=0.0)
    by_id = {result.example_id: result.split for result in results}
    assert by_id["superstore-25880"] == Split.GOLDEN
    assert by_id["superstore-36258"] == Split.BENCHMARK


def test_write_split_jsonl_and_manifest(tmp_path: Path) -> None:
    from scripts.extract_superstore_invoices import ExtractionResult

    parsed = parse_superstore_invoice_text(SAMPLE_INVOICE_ROW_ID)
    result = ExtractionResult(
        example_id="superstore-99112",
        source_pdf="invoice.pdf",
        document=SAMPLE_INVOICE_ROW_ID,
        parsed=parsed,
        target=to_schema_target(parsed),
        split=Split.TRAIN,
        schema_valid=True,
        validation_errors=[],
    )
    counts = write_split_jsonl([result], tmp_path)
    assert counts["train"] == 1
    train_path = tmp_path / "train" / "invoices.jsonl"
    line = json.loads(train_path.read_text(encoding="utf-8").strip())
    assert line["example_id"] == "superstore-99112"
    assert line["split"] == "train"

    manifest_path = write_manifest(
        [result],
        pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path,
        split_counts=counts,
        golden_size=50,
        benchmark_size=20,
        validation_ratio=0.1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["records"][0]["order_id"] == "IN-2012-99112"
    assert manifest["records"][0]["schema_valid"] is True


def test_extract_superstore_invoices_missing_dir_writes_empty_manifest(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "missing"
    output_dir = tmp_path / "out"
    results = extract_superstore_invoices(pdf_dir, output_dir)
    assert results == []
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 0
