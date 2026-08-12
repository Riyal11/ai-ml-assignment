"""Tests for JSON Schema + Pydantic validation helpers."""

from docextract.data.validation import (
    load_json_schema,
    validate_dict_against_json_schema,
    validate_invoice,
    validate_invoice_pydantic,
)


def valid_invoice() -> dict[str, object]:
    """Return a schema-valid invoice dict."""
    return {
        "invoice_number": "INV-0001",
        "vendor_name": "Acme Supplies",
        "invoice_date": "2025-06-15",
        "line_items": [
            {"description": "Widget", "quantity": 2, "unit_price": 10},
            {"description": "Gadget", "quantity": 1, "unit_price": 5},
        ],
        "subtotal": 25,
        "tax_amount": 5,
        "total_amount": 30,
        "currency": "USD",
    }


def test_load_schema() -> None:
    schema = load_json_schema()
    assert schema["title"] == "Invoice"
    assert set(schema["required"]) == {
        "invoice_number",
        "vendor_name",
        "invoice_date",
        "line_items",
        "subtotal",
        "tax_amount",
        "total_amount",
        "currency",
    }


def test_valid_dict_passes_json_schema() -> None:
    ok, errors = validate_dict_against_json_schema(valid_invoice())
    assert ok is True
    assert errors == []


def test_missing_field_fails_json_schema() -> None:
    data = valid_invoice()
    del data["currency"]
    ok, errors = validate_dict_against_json_schema(data)
    assert ok is False
    assert errors


def test_invalid_date_format_fails_json_schema() -> None:
    data = valid_invoice()
    data["invoice_date"] = "15/06/2025"
    ok, errors = validate_dict_against_json_schema(data)
    assert ok is False
    assert any(e["path"] == ["invoice_date"] for e in errors)


def test_extra_field_fails_json_schema() -> None:
    data = valid_invoice()
    data["unknown_field"] = "surprise"
    ok, errors = validate_dict_against_json_schema(data)
    assert ok is False


def test_valid_pydantic_invoice() -> None:
    ok, errors = validate_invoice_pydantic(valid_invoice())
    assert ok is True
    assert errors == []


def test_invalid_currency_fails_pydantic() -> None:
    data = valid_invoice()
    data["currency"] = "usd"
    ok, errors = validate_invoice_pydantic(data)
    assert ok is False
    assert any("currency" in str(e["path"]) for e in errors)


def test_unified_valid_both_pass() -> None:
    ok, errors = validate_invoice(valid_invoice())
    assert ok is True
    assert errors == []


def test_unified_missing_field_tagged_json_schema() -> None:
    data = valid_invoice()
    del data["tax_amount"]
    ok, errors = validate_invoice(data)
    assert ok is False
    assert any("json_schema" in str(e) for e in errors)


def test_unified_bad_currency_tagged_pydantic_only() -> None:
    data = valid_invoice()
    data["currency"] = "usd"
    ok, errors = validate_invoice(data)
    assert ok is False
    pydantic_hits = [e for e in errors if e["source"] == "pydantic"]
    assert pydantic_hits


def test_unified_both_sources_catch_error() -> None:
    data = valid_invoice()
    data["currency"] = "usd"
    del data["tax_amount"]
    ok, errors = validate_invoice(data)
    assert ok is False
    sources = {e["source"] for e in errors}
    assert sources == {"json_schema", "pydantic"}


def test_errors_have_message_path_source() -> None:
    data = valid_invoice()
    data["currency"] = "usd"
    ok, errors = validate_invoice(data)
    assert ok is False
    assert errors
    for e in errors:
        assert "message" in e
        assert "source" in e
