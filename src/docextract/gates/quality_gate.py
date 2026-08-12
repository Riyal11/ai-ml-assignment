"""Quality gate: compare evaluation results against acceptance criteria."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Operator = Literal["gte", "lte"]

_QUANTITATIVE_SECTIONS: frozenset[str] = frozenset(
    {
        "Structured Output Validity",
        "General Capability (Catastrophic Forgetting)",
    }
)

_RESULT_KEY_MAP: dict[str, tuple[str, ...]] = {
    "JSON Schema Validity Rate": ("schema_validity_rate",),
    "Field-Level Exact Match (EM)": ("exact_match",),
    "Field-Level F1": ("precision_recall_f1", "f1"),
    "Base vs. Fine-Tuned on General Benchmark": ("relative_benchmark_drop",),
}

_THRESHOLD_PATTERN = re.compile(r"([≥≤])\s*([\d.]+)\s*(%)?")


@dataclass(frozen=True)
class Criterion:
    """A single quantitative acceptance threshold parsed from markdown."""

    name: str
    operator: Operator
    threshold: float


def parse_threshold(threshold_text: str) -> tuple[Operator, float]:
    """Parse a threshold cell such as ``**≥ 90%**`` or ``**≤ 5% relative drop**``.

    Args:
        threshold_text: Raw markdown table cell for the threshold column.

    Returns:
        A tuple of ``(operator, normalized_threshold)`` where percentages are
        converted to fractions (e.g. ``90%`` → ``0.90``).

    Raises:
        ValueError: If the threshold text cannot be parsed.
    """
    text = threshold_text.replace("**", "").strip()
    match = _THRESHOLD_PATTERN.search(text)
    if match is None:
        raise ValueError(f"cannot parse threshold from {threshold_text!r}")

    op_char = match.group(1)
    value = float(match.group(2))
    has_percent = match.group(3) is not None
    if has_percent or value > 1.0:
        value /= 100.0

    operator: Operator = "gte" if op_char == "≥" else "lte"
    return operator, value


def parse_criteria(criteria_path: Path) -> list[Criterion]:
    """Parse quantitative thresholds from ``acceptance_criteria.md`` tables.

    Only sections listed in ``_QUANTITATIVE_SECTIONS`` are considered; the
    human-review section is intentionally skipped.

    Args:
        criteria_path: Path to the acceptance criteria markdown file.

    Returns:
        Parsed criteria in document order.

    Raises:
        FileNotFoundError: If ``criteria_path`` does not exist.
    """
    if not criteria_path.is_file():
        raise FileNotFoundError(f"criteria file not found: {criteria_path}")

    criteria: list[Criterion] = []
    in_quantitative = False

    for line in criteria_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section_title = line[3:].strip()
            in_quantitative = section_title in _QUANTITATIVE_SECTIONS
            continue
        if not in_quantitative or not line.startswith("|"):
            continue
        if line.startswith("|--") or line.startswith("| Metric"):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 2:
            continue

        operator, threshold = parse_threshold(parts[1])
        criteria.append(Criterion(name=parts[0], operator=operator, threshold=threshold))

    return criteria


def _get_nested_value(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return a numeric metric from nested ``results.json`` keys."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool):
        return None
    if isinstance(current, int | float):
        return float(current)
    return None


def _load_results(results_path: Path) -> dict[str, Any]:
    """Load and parse a ``results.json`` evaluation artifact.

    Args:
        results_path: Path to the results JSON file.

    Returns:
        Parsed results dict.

    Raises:
        FileNotFoundError: If ``results_path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not results_path.is_file():
        raise FileNotFoundError(f"results file not found: {results_path}")
    data: object = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"results file must contain a JSON object: {results_path}")
    return data


def _check_value(actual: float, operator: Operator, threshold: float) -> bool:
    """Return whether ``actual`` satisfies the threshold comparison."""
    if operator == "gte":
        return actual >= threshold
    return actual <= threshold


def _format_comparison(
    criterion: Criterion,
    actual: float,
) -> str:
    """Format a human-readable failure message for one criterion."""
    symbol = "≥" if criterion.operator == "gte" else "≤"
    return (
        f"{criterion.name}: actual={actual:.4f} does not meet "
        f"threshold {symbol} {criterion.threshold:.4f}"
    )


def run_gate(results_path: Path, criteria_path: Path) -> tuple[bool, list[str]]:
    """Compare evaluation metrics against acceptance-criteria thresholds.

    Args:
        results_path: Path to ``results.json`` from the eval pipeline.
        criteria_path: Path to ``acceptance_criteria.md``.

    Returns:
        ``(all_passed, failure_messages)`` where ``failure_messages`` is empty
        when every quantitative criterion passes.
    """
    results = _load_results(results_path)
    criteria = parse_criteria(criteria_path)
    failures: list[str] = []

    for criterion in criteria:
        key_path = _RESULT_KEY_MAP.get(criterion.name)
        if key_path is None:
            failures.append(f"{criterion.name}: no results.json mapping defined")
            continue

        actual = _get_nested_value(results, key_path)
        if actual is None:
            failures.append(
                f"{criterion.name}: metric missing at " f"results.json key {'.'.join(key_path)}"
            )
            continue

        if not _check_value(actual, criterion.operator, criterion.threshold):
            failures.append(_format_comparison(criterion, actual))

    all_passed = not failures
    if all_passed:
        logger.info("Quality gate passed (%d criteria)", len(criteria))
    else:
        logger.error("Quality gate failed (%d/%d criteria)", len(failures), len(criteria))
        for message in failures:
            logger.error("%s", message)

    return all_passed, failures


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the quality gate."""
    parser = argparse.ArgumentParser(
        description="Fail CI when evaluation metrics miss acceptance thresholds",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results.json from the evaluation pipeline",
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        required=True,
        help="Path to acceptance_criteria.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quality gate CLI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when all criteria pass, ``1`` otherwise.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    passed, _ = run_gate(args.results, args.criteria)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
