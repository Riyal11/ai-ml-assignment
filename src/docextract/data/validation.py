"""Validation helpers combining JSON Schema and Pydantic checks."""

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from docextract.data.schemas import Invoice

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "configs" / "schema" / "invoice_schema.json"


def load_json_schema() -> dict[str, Any]:
    """Load the invoice JSON Schema from ``configs/schema``."""
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        loaded: dict[str, Any] = json.load(f)
        return loaded


def validate_dict_against_json_schema(record: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Validate a plain dict against the invoice JSON Schema.

    Returns ``(valid, errors)`` where each error is a dict with
    ``message`` and ``path`` keys.
    """
    schema = load_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors: list[dict[str, Any]] = []
    for err in validator.iter_errors(record):
        errors.append(
            {
                "message": err.message,
                "path": list(err.absolute_path) if err.absolute_path else [],
            }
        )
    return (len(errors) == 0, errors)


def validate_invoice_pydantic(record: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Validate a plain dict against the Pydantic ``Invoice`` model.

    Returns ``(valid, errors)`` where each error is a dict with
    ``message`` and ``path`` keys.
    """
    try:
        Invoice.model_validate(record)
    except ValidationError as exc:
        errors: list[dict[str, Any]] = []
        for err in exc.errors():
            errors.append(
                {
                    "message": err["msg"],
                    "path": list(err["loc"]) if err["loc"] else [],
                }
            )
        return (False, errors)
    return (True, [])


def validate_invoice(record: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Run both JSON Schema and Pydantic validation.

    Returns ``(valid, errors)``. Errors are tagged with a ``source``
    key of either ``"json_schema"`` or ``"pydantic"``. A record is
    valid only if both validators pass.
    """
    js_ok, js_errors = validate_dict_against_json_schema(record)
    pd_ok, pd_errors = validate_invoice_pydantic(record)

    tagged: list[dict[str, Any]] = []
    for err in js_errors:
        tagged.append({**err, "source": "json_schema"})
    for err in pd_errors:
        tagged.append({**err, "source": "pydantic"})
    return (js_ok and pd_ok, tagged)
