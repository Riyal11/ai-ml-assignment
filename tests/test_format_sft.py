"""Tests for SFT chat-template formatting."""

import json

import pytest

from docextract.data.dataset import DocumentRecord, Split
from docextract.data.format_sft import (
    build_extraction_prompt,
    format_sft_dataset,
    format_sft_example,
    save_sft_jsonl,
)


def make_record(language: str = "en", target: dict[str, object] | None = None) -> DocumentRecord:
    """Build a minimal valid DocumentRecord for formatting tests."""
    return DocumentRecord(
        example_id="ex-1",
        document="Invoice INV-1 from Acme for widgets.",
        target=target
        or {
            "invoice_number": "INV-1",
            "vendor_name": "Acme",
            "invoice_date": "2025-01-15",
            "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 5}],
            "subtotal": 10,
            "tax_amount": 1,
            "total_amount": 11,
            "currency": "USD",
        },
        language=language,
        split=Split.TRAIN,
    )


def test_build_extraction_prompt_contains_json_and_schema_fields() -> None:
    prompt = build_extraction_prompt("some doc text", "en")
    assert "JSON" in prompt
    for field in ("invoice_number", "vendor_name", "invoice_date", "currency"):
        assert field in prompt


def test_format_sft_example_message_structure() -> None:
    example = format_sft_example(make_record())
    messages = example["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"].startswith("You are an invoice")
    assert messages[1]["content"] == make_record().document


def test_assistant_message_is_pure_json() -> None:
    record = make_record()
    example = format_sft_example(record)
    assistant_content = example["messages"][2]["content"]
    assert "```" not in assistant_content
    assert "Here is the extracted" not in assistant_content
    parsed = json.loads(assistant_content)
    assert parsed["invoice_number"] == "INV-1"
    assert parsed["currency"] == "USD"


def test_format_sft_dataset_preserves_length() -> None:
    records = [make_record() for _ in range(3)]
    examples = format_sft_dataset(records)
    assert len(examples) == 3
    for example, record in zip(examples, records, strict=True):
        assert example["messages"][2]["content"] == json.dumps(
            record.target, ensure_ascii=False, indent=2
        )


def test_save_sft_jsonl_writes_valid_jsonl(tmp_path) -> None:
    records = [make_record() for _ in range(2)]
    out_path = tmp_path / "sft" / "train.jsonl"
    save_sft_jsonl(format_sft_dataset(records), out_path)
    assert out_path.exists()
    with out_path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 2
    for line in lines:
        example = json.loads(line)
        assert "messages" in example


def test_hindi_record_includes_language_hint() -> None:
    example = format_sft_example(make_record(language="hi"))
    system_content = example["messages"][0]["content"]
    assert "Hindi" in system_content


def test_missing_target_field_still_formats() -> None:
    record = make_record(target={"invoice_number": "INV-9"})
    example = format_sft_example(record)
    assert json.loads(example["messages"][2]["content"]) == {"invoice_number": "INV-9"}


def test_english_record_has_no_hindi_hint() -> None:
    example = format_sft_example(make_record(language="en"))
    assert "Hindi" not in example["messages"][0]["content"]


@pytest.mark.parametrize("language", ["hi", "Hi", "HI"])
def test_language_hint_case_insensitive(language: str) -> None:
    example = format_sft_example(make_record(language=language))
    assert "Hindi" in example["messages"][0]["content"]
