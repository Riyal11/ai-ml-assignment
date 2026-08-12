"""Tests for the evaluation quality gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from docextract.gates.quality_gate import main, parse_criteria, parse_threshold, run_gate

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CRITERIA = Path(__file__).resolve().parents[1] / "docs" / "acceptance_criteria.md"
PASS_RESULTS = FIXTURES / "sample_results_pass.json"
FAIL_RESULTS = FIXTURES / "sample_results_fail.json"


def test_parse_threshold_percent() -> None:
    operator, value = parse_threshold("**≥ 90%**")
    assert operator == "gte"
    assert value == pytest.approx(0.90)


def test_parse_threshold_decimal() -> None:
    operator, value = parse_threshold("**≥ 0.85**")
    assert operator == "gte"
    assert value == pytest.approx(0.85)


def test_parse_threshold_relative_drop() -> None:
    operator, value = parse_threshold("**≤ 5% relative drop**")
    assert operator == "lte"
    assert value == pytest.approx(0.05)


def test_parse_threshold_invalid_raises() -> None:
    with pytest.raises(ValueError, match="cannot parse threshold"):
        parse_threshold("not a threshold")


def test_parse_criteria_reads_quantitative_sections() -> None:
    criteria = parse_criteria(CRITERIA)
    names = [criterion.name for criterion in criteria]
    assert "JSON Schema Validity Rate" in names
    assert "Field-Level F1" in names
    assert "Base vs. Fine-Tuned on General Benchmark" in names
    assert "Correctness (field values match document)" not in names


def test_run_gate_pass() -> None:
    passed, failures = run_gate(PASS_RESULTS, CRITERIA)
    assert passed is True
    assert failures == []


def test_run_gate_fail_on_low_f1() -> None:
    passed, failures = run_gate(FAIL_RESULTS, CRITERIA)
    assert passed is False
    assert any("Field-Level F1" in message for message in failures)


def test_run_gate_missing_results_raises() -> None:
    with pytest.raises(FileNotFoundError, match="results file not found"):
        run_gate(Path("nonexistent/results.json"), CRITERIA)


def test_run_gate_missing_metric_in_results(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results = json.loads(PASS_RESULTS.read_text(encoding="utf-8"))
    del results["relative_benchmark_drop"]
    results_path.write_text(json.dumps(results), encoding="utf-8")

    passed, failures = run_gate(results_path, CRITERIA)
    assert passed is False
    assert any("relative_benchmark_drop" in message for message in failures)


def test_run_gate_rejects_non_object_results(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        run_gate(results_path, CRITERIA)


def test_run_gate_unmapped_metric(tmp_path: Path) -> None:
    criteria_path = tmp_path / "criteria.md"
    criteria_path.write_text(
        "## Structured Output Validity\n\n"
        "| Metric | Threshold | Rationale |\n"
        "|--------|-----------|-----------|\n"
        "| Custom Metric | **≥ 90%** | test |\n",
        encoding="utf-8",
    )
    passed, failures = run_gate(PASS_RESULTS, criteria_path)
    assert passed is False
    assert any("no results.json mapping defined" in message for message in failures)


def test_run_gate_missing_criteria_raises() -> None:
    with pytest.raises(FileNotFoundError, match="criteria file not found"):
        run_gate(PASS_RESULTS, Path("nonexistent/acceptance_criteria.md"))


def test_cli_exit_zero_on_pass() -> None:
    assert main(["--results", str(PASS_RESULTS), "--criteria", str(CRITERIA)]) == 0


def test_cli_exit_one_on_fail() -> None:
    assert main(["--results", str(FAIL_RESULTS), "--criteria", str(CRITERIA)]) == 1


def test_script_exits_one_on_failure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "quality_gate.py"),
            "--results",
            str(FAIL_RESULTS),
            "--criteria",
            str(CRITERIA),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
