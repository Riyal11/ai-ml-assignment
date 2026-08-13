"""CLI entrypoint for exporting MLflow runs."""

import argparse
import logging
import sys
from pathlib import Path

from docextract.experiments.mlflow_export import export_mlflow_runs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MLflow runs for submission")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/mlflow_export"),
        help="Directory for per-run JSON and summary.csv",
    )
    parser.add_argument(
        "--experiment-name",
        default="docextract",
        help="MLflow experiment name to export",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Optional MLflow tracking URI override",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    export_mlflow_runs(
        args.output_dir,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
