"""Tests for invoice extraction evaluation metrics."""

from typing import Any

import pytest

from docextract.eval.metrics import (
    compute_exact_match,
    compute_forgetting_score,
    compute_precision_recall_f1,
    compute_schema_validity_rate,
)


def base_invoice() -> dict[str, Any]:
    """Return a base gold invoice dict for metric tests."""
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


def test_em_perfect_match() -> None:
    em = compute_exact_match(base_invoice(), base_invoice())
    assert em["_overall"] == 1.0
    for field in (
        "invoice_number",
        "vendor_name",
        "invoice_date",
        "subtotal",
        "tax_amount",
        "total_amount",
        "currency",
    ):
        assert em[field] == 1.0


def test_em_partial_match() -> None:
    pred = base_invoice()
    pred["vendor_name"] = "Wrong Vendor"
    em = compute_exact_match(pred, base_invoice())
    assert em["vendor_name"] == 0.0
    assert em["invoice_number"] == 1.0
    assert em["_overall"] == 6 / 7


def test_em_date_normalization() -> None:
    pred = base_invoice()
    pred["invoice_date"] = "2025-6-5"
    gold = base_invoice()
    gold["invoice_date"] = "2025-06-05"
    em = compute_exact_match(pred, gold)
    assert em["invoice_date"] == 1.0


def test_em_money_normalization() -> None:
    pred = base_invoice()
    pred["total_amount"] = "24.0"
    em = compute_exact_match(pred, base_invoice())
    assert em["total_amount"] == 1.0


def test_prf_perfect() -> None:
    prf = compute_precision_recall_f1(base_invoice(), base_invoice())
    overall = prf["_overall"]
    assert overall["precision"] == 1.0
    assert overall["recall"] == 1.0
    assert overall["f1"] == 1.0


def test_prf_partial_scalar() -> None:
    pred = base_invoice()
    pred["currency"] = "EUR"
    prf = compute_precision_recall_f1(pred, base_invoice())
    assert prf["currency"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overall = prf["_overall"]
    assert 0.0 < overall["f1"] < 1.0


def test_prf_line_items_aligned_match() -> None:
    prf = compute_precision_recall_f1(base_invoice(), base_invoice())
    assert prf["line_items"]["f1"] == 1.0


def test_prf_line_items_extra_pred_is_fp() -> None:
    pred = base_invoice()
    pred["line_items"] = pred["line_items"] + [
        {"description": "Extra", "quantity": 1, "unit_price": 5}
    ]
    prf = compute_precision_recall_f1(pred, base_invoice())
    assert prf["line_items"]["precision"] < 1.0
    assert prf["line_items"]["recall"] == 1.0


def test_prf_line_items_missing_pred_is_fn() -> None:
    pred = base_invoice()
    gold = base_invoice()
    gold["line_items"] = gold["line_items"] + [
        {"description": "Extra", "quantity": 1, "unit_price": 5}
    ]
    prf = compute_precision_recall_f1(pred, gold)
    assert prf["line_items"]["recall"] < 1.0
    assert prf["line_items"]["precision"] == 1.0


def test_schema_validity_rate_all_valid() -> None:
    rate = compute_schema_validity_rate([base_invoice(), base_invoice()])
    assert rate == 1.0


def test_schema_validity_rate_mixed() -> None:
    bad = base_invoice()
    del bad["currency"]
    rate = compute_schema_validity_rate([base_invoice(), bad])
    assert rate == 0.5


def test_forgetting_score_retention_and_drop() -> None:
    assert compute_forgetting_score([0.8, 0.8], [0.8, 0.8]) == pytest.approx(1.0)
    assert compute_forgetting_score([0.8, 0.8], [0.6, 0.6]) == pytest.approx(0.75)
