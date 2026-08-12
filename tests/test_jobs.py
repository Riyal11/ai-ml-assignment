"""Tests for quantization job endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from docextract.api.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a TestClient with a stub-loaded inference service."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    application = create_app(model_path=model_dir)
    with TestClient(application) as test_client:
        yield test_client


def test_quantize_job_endpoint_accepts_request(client: TestClient) -> None:
    with patch(
        "docextract.api.routers.jobs.run_quantize_sync",
        return_value={"output_path": "artifacts/gguf/model.Q4_K_M.gguf"},
    ):
        response = client.post(
            "/v1/jobs/quantize",
            json={"model_path": "artifacts/merged-model", "method": "gguf"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] in {"queued", "completed"}


def test_job_status_endpoint_returns_status(client: TestClient) -> None:
    with patch(
        "docextract.api.routers.jobs.run_quantize_sync",
        return_value={"output_path": "artifacts/gguf/model.Q4_K_M.gguf"},
    ):
        create_response = client.post(
            "/v1/jobs/quantize",
            json={"model_path": "artifacts/merged-model", "method": "gguf"},
        )
    job_id = create_response.json()["job_id"]
    status_response = client.get(f"/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["job_id"] == job_id
    assert payload["status"] == "SUCCESS"
    assert payload["result"] is not None


def test_invalid_method_returns_400(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs/quantize",
        json={"model_path": "artifacts/merged-model", "method": "invalid-method"},
    )
    assert response.status_code == 400
