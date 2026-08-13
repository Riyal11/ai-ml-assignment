"""Compute catastrophic-forgetting metrics from base vs fine-tuned eval results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docextract.eval.metrics import compute_forgetting_score


def _f1(results: dict[str, Any]) -> float:
    return float(results["precision_recall_f1"]["f1"])


def compute_relative_benchmark_drop(
    base_results: dict[str, Any],
    ft_results: dict[str, Any],
) -> dict[str, float]:
    """Compute retention ratio and relative F1 drop between two result dicts."""
    base_f1 = _f1(base_results)
    ft_f1 = _f1(ft_results)
    retention = compute_forgetting_score([base_f1], [ft_f1])
    relative_drop = 1.0 - retention if base_f1 > 0 else 1.0
    return {
        "base_f1": base_f1,
        "fine_tuned_f1": ft_f1,
        "retention_ratio": retention,
        "relative_benchmark_drop": relative_drop,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute forgetting metrics from base and fine-tuned results.json files",
    )
    parser.add_argument("base_results", type=Path, help="Base model results.json")
    parser.add_argument("ft_results", type=Path, help="Fine-tuned model results.json")
    parser.add_argument(
        "--write-ft",
        action="store_true",
        help="Merge relative_benchmark_drop into the fine-tuned results file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    base = json.loads(args.base_results.read_text(encoding="utf-8"))
    ft = json.loads(args.ft_results.read_text(encoding="utf-8"))
    metrics = compute_relative_benchmark_drop(base, ft)

    print(f"Base F1:              {metrics['base_f1']:.4f}")
    print(f"Fine-tuned F1:        {metrics['fine_tuned_f1']:.4f}")
    print(f"Retention ratio:      {metrics['retention_ratio']:.4f}")
    drop = metrics["relative_benchmark_drop"]
    print(f"Relative drop:        {drop:.4f} ({drop * 100:.1f}%)")

    if args.write_ft:
        ft["relative_benchmark_drop"] = metrics["relative_benchmark_drop"]
        ft["forgetting_retention_ratio"] = metrics["retention_ratio"]
        args.ft_results.write_text(json.dumps(ft, indent=2), encoding="utf-8")
        print(f"Updated {args.ft_results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
