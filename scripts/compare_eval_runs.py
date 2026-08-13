"""Compare two evaluation ``results.json`` files side by side."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    "schema_validity_rate",
    "exact_match",
    "precision_recall_f1",
    "num_examples",
)


def _load_results(path: Path) -> dict[str, Any]:
    """Load a results JSON file."""
    if not path.is_file():
        msg = f"results file not found: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare key metrics between two evaluation result dicts."""
    left_f1 = float(left.get("precision_recall_f1", {}).get("f1", 0.0))
    right_f1 = float(right.get("precision_recall_f1", {}).get("f1", 0.0))
    winner = "left" if left_f1 > right_f1 else "right" if right_f1 > left_f1 else "tie"

    field_comparison: dict[str, dict[str, float]] = {}
    left_fields = left.get("field_precision_recall_f1", {})
    right_fields = right.get("field_precision_recall_f1", {})
    if isinstance(left_fields, dict) and isinstance(right_fields, dict):
        for field in sorted(set(left_fields) | set(right_fields)):
            left_field = left_fields.get(field, {})
            right_field = right_fields.get(field, {})
            if isinstance(left_field, dict) and isinstance(right_field, dict):
                field_comparison[field] = {
                    "left_f1": float(left_field.get("f1", 0.0)),
                    "right_f1": float(right_field.get("f1", 0.0)),
                    "delta_f1": float(right_field.get("f1", 0.0))
                    - float(left_field.get("f1", 0.0)),
                }

    return {
        "left": {key: left.get(key) for key in _METRIC_KEYS},
        "right": {key: right.get(key) for key in _METRIC_KEYS},
        "delta": {
            "schema_validity_rate": float(right.get("schema_validity_rate", 0.0))
            - float(left.get("schema_validity_rate", 0.0)),
            "exact_match": float(right.get("exact_match", 0.0))
            - float(left.get("exact_match", 0.0)),
            "f1": right_f1 - left_f1,
        },
        "field_comparison": field_comparison,
        "recommended": winner,
        "left_model_path": left.get("model_path"),
        "right_model_path": right.get("model_path"),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Compare two evaluation results.json files")
    parser.add_argument("left", type=Path, help="First results.json (e.g. run-001)")
    parser.add_argument("right", type=Path, help="Second results.json (e.g. run-002)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full comparison as JSON instead of a summary table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    comparison = compare_results(_load_results(args.left), _load_results(args.right))
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        left = comparison["left"]
        right = comparison["right"]
        delta = comparison["delta"]
        print(f"{'Metric':<24} {'Left':>10} {'Right':>10} {'Delta':>10}")
        print("-" * 58)
        print(
            f"{'schema_validity_rate':<24} "
            f"{left['schema_validity_rate']:>10.4f} "
            f"{right['schema_validity_rate']:>10.4f} "
            f"{delta['schema_validity_rate']:>+10.4f}"
        )
        print(
            f"{'exact_match':<24} "
            f"{left['exact_match']:>10.4f} "
            f"{right['exact_match']:>10.4f} "
            f"{delta['exact_match']:>+10.4f}"
        )
        left_f1 = left["precision_recall_f1"]["f1"]
        right_f1 = right["precision_recall_f1"]["f1"]
        print(f"{'f1':<24} {left_f1:>10.4f} {right_f1:>10.4f} {delta['f1']:>+10.4f}")
        print(
            f"{'num_examples':<24} " f"{left['num_examples']:>10} " f"{right['num_examples']:>10}"
        )
        print()
        print(f"Recommended: {comparison['recommended']}")
        print(f"Left model:  {comparison['left_model_path']}")
        print(f"Right model: {comparison['right_model_path']}")
        if comparison["field_comparison"]:
            print()
            print(f"{'Field':<16} {'Left F1':>10} {'Right F1':>10} {'Delta':>10}")
            print("-" * 50)
            for field, scores in comparison["field_comparison"].items():
                print(
                    f"{field:<16} {scores['left_f1']:>10.4f} "
                    f"{scores['right_f1']:>10.4f} {scores['delta_f1']:>+10.4f}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
