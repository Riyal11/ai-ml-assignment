"""Evaluation metrics for invoice extraction quality."""

from decimal import Decimal, InvalidOperation
from typing import Any

from docextract.eval.schema_validator import validate_schema

SCALAR_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "vendor_name",
    "invoice_date",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
)

_ITEM_SUBFIELDS: tuple[str, ...] = ("description", "quantity", "unit_price")
_MONEY_FIELDS: frozenset[str] = frozenset(
    {"subtotal", "tax_amount", "total_amount", "quantity", "unit_price"}
)


def _normalize_money(value: Any) -> str:
    """Canonicalize a numeric value for comparison (``Decimal``-based)."""
    try:
        return format(Decimal(str(value)).normalize(), "f")
    except (InvalidOperation, ValueError):
        return str(value).strip()


def _normalize_scalar(field: str, value: Any) -> str:
    """Normalize a scalar field value for comparison.

    Dates are padded to ``YYYY-MM-DD``; money-like fields are compared
    as canonical ``Decimal`` strings; everything else is stripped text.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if field == "invoice_date":
        parts = text.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        return text
    if field in _MONEY_FIELDS:
        return _normalize_money(value)
    return text


def compute_exact_match(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    """Compute field-level exact match on the 7 scalar invoice fields.

    Args:
        pred: Predicted JSON object.
        gold: Gold-standard JSON object.

    Returns:
        ``{field: 1.0 or 0.0, ..., "_overall": mean}``.
    """
    scores: dict[str, float] = {}
    for field in SCALAR_FIELDS:
        pred_val = _normalize_scalar(field, pred.get(field))
        gold_val = _normalize_scalar(field, gold.get(field))
        scores[field] = 1.0 if pred_val == gold_val else 0.0
    scores["_overall"] = sum(scores[f] for f in SCALAR_FIELDS) / len(SCALAR_FIELDS)
    return scores


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Derive precision/recall/F1 from raw counts (0.0 when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_precision_recall_f1(
    pred: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Compute per-field precision/recall/F1.

    Scalar fields contribute one TP or one FP+FN each. ``line_items``
    are aligned by index and scored per subfield; unmatched predicted
    items count as FP, unmatched gold items as FN (per subfield).

    Args:
        pred: Predicted JSON object.
        gold: Gold-standard JSON object.

    Returns:
        ``{field: {"precision", "recall", "f1"}, "_overall": {...}}``
        where ``_overall`` is micro-averaged across all fields.
    """
    counts: dict[str, dict[str, int]] = {
        field: {"tp": 0, "fp": 0, "fn": 0} for field in (*SCALAR_FIELDS, "line_items")
    }

    for field in SCALAR_FIELDS:
        if _normalize_scalar(field, pred.get(field)) == _normalize_scalar(field, gold.get(field)):
            counts[field]["tp"] += 1
        else:
            counts[field]["fp"] += 1
            counts[field]["fn"] += 1

    pred_items = pred.get("line_items") or []
    gold_items = gold.get("line_items") or []
    if not isinstance(pred_items, list):
        pred_items = []
    if not isinstance(gold_items, list):
        gold_items = []

    for i in range(max(len(pred_items), len(gold_items))):
        item_counts = counts["line_items"]
        if i < len(pred_items) and i < len(gold_items):
            pred_item = pred_items[i] if isinstance(pred_items[i], dict) else {}
            gold_item = gold_items[i] if isinstance(gold_items[i], dict) else {}
            for sub in _ITEM_SUBFIELDS:
                if _normalize_scalar(sub, pred_item.get(sub)) == _normalize_scalar(
                    sub, gold_item.get(sub)
                ):
                    item_counts["tp"] += 1
                else:
                    item_counts["fp"] += 1
                    item_counts["fn"] += 1
        elif i < len(pred_items):
            item_counts["fp"] += len(_ITEM_SUBFIELDS)
        else:
            item_counts["fn"] += len(_ITEM_SUBFIELDS)

    result: dict[str, dict[str, float]] = {
        field: _prf(c["tp"], c["fp"], c["fn"]) for field, c in counts.items()
    }
    overall_tp = sum(c["tp"] for c in counts.values())
    overall_fp = sum(c["fp"] for c in counts.values())
    overall_fn = sum(c["fn"] for c in counts.values())
    result["_overall"] = _prf(overall_tp, overall_fp, overall_fn)
    return result


def compute_schema_validity_rate(predictions: list[dict[str, Any]]) -> float:
    """Compute the fraction of predictions that validate against the schema.

    Args:
        predictions: Parsed prediction dicts.

    Returns:
        Validity rate in ``[0.0, 1.0]`` (0.0 for an empty list).
    """
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if validate_schema(p)[0])
    return valid / len(predictions)


def compute_forgetting_score(base_scores: list[float], ft_scores: list[float]) -> float:
    """Compute the catastrophic-forgetting retention ratio.

    Args:
        base_scores: Per-example scores of the untouched base model.
        ft_scores: Per-example scores of the fine-tuned model.

    Returns:
        ``mean(ft_scores) / mean(base_scores)``; 1.0 means no drop.
        Returns 0.0 for empty inputs or a zero base mean.
    """
    if not base_scores or not ft_scores:
        return 0.0
    base_mean = sum(base_scores) / len(base_scores)
    if base_mean == 0.0:
        return 0.0
    ft_mean = sum(ft_scores) / len(ft_scores)
    return ft_mean / base_mean
