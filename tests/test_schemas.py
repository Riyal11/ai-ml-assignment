"""Tests for Pydantic v2 invoice models."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from docextract.data.schemas import Invoice, LineItem


def valid_invoice() -> dict[str, object]:
    """Return a schema-valid invoice dict (English, no whitespace issues)."""
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


def valid_hindi_invoice() -> dict[str, object]:
    """Return a valid invoice with Hindi vendor text."""
    return {
        "invoice_number": "INV-0002",
        "vendor_name": "एक्मे सप्लाईज़",
        "invoice_date": "2025-06-16",
        "line_items": [{"description": "विजेट", "quantity": 3, "unit_price": 15}],
        "subtotal": 45,
        "tax_amount": 9,
        "total_amount": 54,
        "currency": "INR",
    }


def test_valid_invoice_accepted() -> None:
    invoice = Invoice.model_validate(valid_invoice())
    assert invoice.invoice_number == "INV-0001"
    assert invoice.currency == "USD"
    assert len(invoice.line_items) == 2


def test_missing_required_field_rejected() -> None:
    data = valid_invoice()
    del data["vendor_name"]
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_invalid_date_format_rejected() -> None:
    data = valid_invoice()
    data["invoice_date"] = "15/06/2025"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_invalid_date_value_rejected() -> None:
    data = valid_invoice()
    data["invoice_date"] = "2025-02-31"  # impossible date
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_lowercase_currency_rejected() -> None:
    data = valid_invoice()
    data["currency"] = "usd"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_short_currency_rejected() -> None:
    data = valid_invoice()
    data["currency"] = "US"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_string_in_numeric_field_rejected() -> None:
    data = valid_invoice()
    data["line_items"][0]["quantity"] = "abc"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_missing_line_item_field_rejected() -> None:
    data = valid_invoice()
    del data["line_items"][0]["quantity"]
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_empty_line_items_rejected() -> None:
    data = valid_invoice()
    data["line_items"] = []
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_extra_field_rejected() -> None:
    data = valid_invoice()
    data["unknown_field"] = "surprise"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_hindi_text_accepted() -> None:
    invoice = Invoice.model_validate(valid_hindi_invoice())
    assert invoice.vendor_name == "एक्मे सप्लाईज़"
    assert invoice.currency == "INR"


def test_stripped_strings() -> None:
    data = valid_invoice()
    data["vendor_name"] = "  Acme Supplies  "
    invoice = Invoice.model_validate(data)
    assert invoice.vendor_name == "Acme Supplies"


def test_decimal_field_types() -> None:
    data = valid_invoice()
    data["subtotal"] = "not-a-number"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_json_schema_export_contains_required() -> None:
    schema = Invoice.model_json_schema()
    required = {
        "invoice_number",
        "vendor_name",
        "invoice_date",
        "line_items",
        "subtotal",
        "tax_amount",
        "total_amount",
        "currency",
    }
    assert required.issubset(set(schema["required"]))
    assert "line_items" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_line_item_strict_decimals() -> None:
    item = LineItem.model_validate(
        {"description": "  Widget  ", "quantity": 2, "unit_price": Decimal("10.50")}
    )
    assert item.description == "Widget"
    assert item.quantity == Decimal("2")
    assert item.unit_price == Decimal("10.50")
