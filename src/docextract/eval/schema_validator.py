"""Schema validation helpers for raw model output text."""

import json
import re
from typing import Any

import jsonschema

from docextract.data.validation import load_json_schema

_MD_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json_from_text(text: str) -> str | None:
    """Extract the first balanced ``{...}`` JSON object substring.

    Strips markdown code fences first, then scans for the first ``{``
    and its matching closing ``}`` (string-aware, escape-aware).

    Args:
        text: Raw model output text.

    Returns:
        The JSON substring, or ``None`` if no balanced object is found.
    """
    cleaned = text.strip()
    fence = _MD_FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


def validate_schema(output_json: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a dict against the invoice JSON Schema.

    Args:
        output_json: Parsed JSON object.

    Returns:
        ``(is_valid, errors)`` where ``errors`` is a list of
        human-readable ``path: message`` strings.
    """
    schema = load_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = [
        f"{err.json_path or '$'}: {err.message}"
        for err in sorted(validator.iter_errors(output_json), key=lambda e: e.json_path)
    ]
    return (len(errors) == 0, errors)


def validate_output_text(text: str) -> tuple[bool, dict[str, Any] | None, list[str]]:
    """Parse raw model output text as JSON and validate it.

    Handles markdown fences and trailing commas.

    Args:
        text: Raw model output text.

    Returns:
        ``(is_valid, parsed_dict_or_None, errors)``.
    """
    json_text = extract_json_from_text(text)
    if json_text is None:
        return (False, None, ["no JSON object found in output"])

    parsed: Any = None
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        fixed = _TRAILING_COMMA.sub(r"\1", json_text)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError as exc:
            return (False, None, [f"invalid JSON: {exc.msg}"])

    if not isinstance(parsed, dict):
        return (False, None, ["top-level JSON value is not an object"])

    is_valid, errors = validate_schema(parsed)
    return (is_valid, parsed, errors)
