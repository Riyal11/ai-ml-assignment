"""Data layer: schemas, validation, dataset isolation, SFT formatting."""

from docextract.data.dataset import DocumentRecord, Split, guard_train_path
from docextract.data.format_sft import (
    build_extraction_prompt,
    format_sft_dataset,
    format_sft_example,
    save_sft_jsonl,
)
from docextract.data.schemas import Invoice, LineItem
from docextract.data.validation import validate_invoice

__all__ = [
    "DocumentRecord",
    "Invoice",
    "LineItem",
    "Split",
    "build_extraction_prompt",
    "format_sft_dataset",
    "format_sft_example",
    "guard_train_path",
    "save_sft_jsonl",
    "validate_invoice",
]
