"""Tests for MLflow export helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from docextract.experiments.mlflow_export import _ms_to_iso, _run_payload, export_mlflow_runs


def test_ms_to_iso_none() -> None:
    assert _ms_to_iso(None) is None


def test_ms_to_iso_converts() -> None:
    iso = _ms_to_iso(1_000)
    assert iso is not None
    assert "1970" in iso


def test_run_payload_maps_custom_run_id() -> None:
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="abc",
            run_name="test-run",
            status="FINISHED",
            start_time=1_000,
            end_time=2_000,
        ),
        data=SimpleNamespace(
            params={"custom_run_id": "custom-1", "method": "qlora"},
            metrics=SimpleNamespace(items=lambda: [("train_loss", 0.1)]),
            tags={"mlflow.runName": "test-run"},
        ),
    )
    payload = _run_payload(run)
    assert payload["params"]["run_id"] == "custom-1"
    assert payload["metrics"]["train_loss"] == 0.1


def test_export_mlflow_runs_empty_experiment(tmp_path: Path) -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    with (
        patch("mlflow.set_tracking_uri"),
        patch(
            "mlflow.tracking.MlflowClient",
            return_value=client,
        ),
    ):
        count = export_mlflow_runs(tmp_path)
    assert count == 0
    assert (tmp_path / "summary.csv").is_file()


def test_export_mlflow_runs_writes_runs(tmp_path: Path) -> None:
    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-1",
            run_name="qlora",
            status="FINISHED",
            start_time=1_000,
            end_time=2_000,
        ),
        data=SimpleNamespace(
            params={"method": "qlora", "run_id": "custom"},
            metrics=SimpleNamespace(items=lambda: [("train_loss", 0.2)]),
            tags={},
        ),
    )
    experiment = SimpleNamespace(name="docextract", experiment_id="1")
    client = MagicMock()
    client.get_experiment_by_name.return_value = experiment
    client.search_runs.return_value = [run]
    with (
        patch("mlflow.set_tracking_uri"),
        patch(
            "mlflow.tracking.MlflowClient",
            return_value=client,
        ),
    ):
        count = export_mlflow_runs(tmp_path)
    assert count == 1
    assert (tmp_path / "run-1.json").is_file()
    assert (tmp_path / "summary.csv").is_file()
