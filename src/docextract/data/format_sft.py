"""Chat-templated SFT prompt formatting for document extraction."""

import json
import logging
from pathlib import Path
from typing import Any

from docextract.data.dataset import DocumentRecord
from docextract.data.validation import load_json_schema

logger = logging.getLogger(__name__)

_OUTPUT_RULES = (
    "Output strict JSON matching the schema. No markdown code fences, "
    "no explanations, no introductory or trailing text."
)


def _system_instructions(language: str) -> str:
    """Build the system message: extraction instructions plus target schema."""
    schema = load_json_schema()
    schema_text = json.dumps(schema["properties"], indent=2, ensure_ascii=False)

    if language.lower() == "hi":
        language_hint = (
            "The source document is in Hindi or mixes Hindi and English. "
            "Extract fields into the same JSON structure."
        )
    else:
        language_hint = "The source document is in English."

    return (
        "You are an invoice data extraction engine. "
        f"Extract the required fields from the document into strict JSON. "
        f"Target JSON schema:\n{schema_text}\n"
        f"{language_hint}\n"
        f"{_OUTPUT_RULES}"
    )


def build_extraction_prompt(document_text: str, language: str) -> str:
    """Build a single full prompt (system instructions + document text).

    Args:
        document_text: Raw source document text.
        language: Language code of the document (e.g. ``"en"``, ``"hi"``).

    Returns:
        The full instruction prompt including the target schema and document.
    """
    return f"{_system_instructions(language)}\n\nDocument:\n{document_text}"


def format_sft_example(record: DocumentRecord) -> dict[str, Any]:
    """Format one ``DocumentRecord`` into a chat message list.

    Args:
        record: A single (document, target) example.

    Returns:
        A dict with a ``messages`` key, compatible with
        ``transformers`` ``apply_chat_template``. The assistant message
        contains only the target JSON as text.
    """
    target_json: str = json.dumps(record.target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": _system_instructions(record.language)},
            {"role": "user", "content": record.document},
            {"role": "assistant", "content": target_json},
        ]
    }


def format_sft_dataset(records: list[DocumentRecord]) -> list[dict[str, Any]]:
    """Format a list of records into SFT chat examples.

    Args:
        records: Records to format.

    Returns:
        One formatted example per input record.
    """
    return [format_sft_example(record) for record in records]


def save_sft_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    """Write formatted examples to JSONL, one example per line.

    Args:
        examples: Formatted examples (from ``format_sft_dataset``).
        path: Destination file. Parent directories are created on demand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Wrote %d SFT examples to %s", len(examples), path)