"""Tests for extraction-to-SFT JSONL conversion."""

import json
from pathlib import Path

from docextract.data.format_sft import (
    load_sft_examples_from_jsonl,
    sft_example_from_dict,
)


def test_sft_example_from_extraction_record() -> None:
    raw = {
        "example_id": "superstore-1",
        "document": "Invoice #1 from Acme",
        "target": {
            "invoice_number": "1",
            "vendor_name": "Acme",
            "invoice_date": "2025-01-01",
            "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 10}],
            "subtotal": 10,
            "tax_amount": 0,
            "total_amount": 10,
            "currency": "USD",
        },
        "language": "en",
        "split": "train",
    }
    example = sft_example_from_dict(raw)
    assert "messages" in example
    assert example["messages"][1]["content"] == "Invoice #1 from Acme"
    assert json.loads(example["messages"][2]["content"])["invoice_number"] == "1"


def test_load_sft_examples_from_jsonl_accepts_extraction_format(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "ex-1",
                "document": "doc",
                "target": {"invoice_number": "INV-1"},
                "language": "en",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_sft_examples_from_jsonl(path)
    assert len(examples) == 1
    assert "messages" in examples[0]
