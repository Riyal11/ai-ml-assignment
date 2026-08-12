"""Tests for raw model output schema validation."""

import json
from typing import Any

from docextract.eval.schema_validator import (
    extract_json_from_text,
    validate_output_text,
    validate_schema,
)


def valid_invoice() -> dict[str, Any]:
    """Return a schema-valid invoice dict."""
    return {
        "invoice_number": "INV-0001",
        "vendor_name": "Acme Supplies",
        "invoice_date": "2025-06-15",
        "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 10}],
        "subtotal": 20,
        "tax_amount": 4,
        "total_amount": 24,
        "currency": "USD",
    }


def test_validate_schema_valid_invoice() -> None:
    is_valid, errors = validate_schema(valid_invoice())
    assert is_valid is True
    assert errors == []


def test_validate_schema_missing_field() -> None:
    data = valid_invoice()
    del data["currency"]
    is_valid, errors = validate_schema(data)
    assert is_valid is False
    assert errors
    assert any("currency" in e for e in errors)


def test_validate_output_text_clean_json() -> None:
    is_valid, parsed, errors = validate_output_text(json.dumps(valid_invoice()))
    assert is_valid is True
    assert parsed is not None
    assert parsed["currency"] == "USD"
    assert errors == []


def test_validate_output_text_markdown_fences() -> None:
    text = f"```json\n{json.dumps(valid_invoice())}\n```"
    is_valid, parsed, _ = validate_output_text(text)
    assert is_valid is True
    assert parsed is not None
    assert parsed["invoice_number"] == "INV-0001"


def test_validate_output_text_trailing_commas() -> None:
    raw = json.dumps(valid_invoice())
    # Insert a trailing comma before the final closing brace.
    trailing = raw[:-1] + ",}"
    is_valid, parsed, _ = validate_output_text(trailing)
    assert is_valid is True
    assert parsed is not None
    assert parsed["total_amount"] == 24


def test_extract_json_from_text_with_prefix_suffix() -> None:
    text = 'Sure! Here is the result: {"invoice_number": "INV-1"} Done.'
    extracted = extract_json_from_text(text)
    assert extracted == '{"invoice_number": "INV-1"}'
