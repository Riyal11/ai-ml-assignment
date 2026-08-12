"""Tests for the FastAPI application."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from docextract.api.main import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a TestClient with a loaded stub inference service."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    application = create_app(model_path=model_dir)
    with TestClient(application) as test_client:
        yield test_client


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_models_returns_list(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["id"] == "docextract-qwen3-4b"


def test_chat_completions_non_stream_returns_valid_json(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "docextract-qwen3-4b",
            "messages": [{"role": "user", "content": "Extract invoice JSON"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"]
    assert "usage" in payload


def test_chat_completions_stream_returns_sse_chunks(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "docextract-qwen3-4b",
            "messages": [{"role": "user", "content": "Extract invoice JSON"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "data: " in body
    assert "data: [DONE]" in body


def test_extract_returns_extract_response(client: TestClient) -> None:
    response = client.post(
        "/extract",
        json={"document": "Invoice INV-001 from Acme", "language": "en"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "raw_output" in payload
    assert "is_valid" in payload
    assert "validation_errors" in payload
    assert payload["invoice"] is not None


def test_invalid_request_returns_422(client: TestClient) -> None:
    response = client.post("/extract", json={"language": "en"})
    assert response.status_code == 422


def test_model_not_loaded_returns_503(client: TestClient) -> None:
    client.app.state.inference._loaded = False
    response = client.post(
        "/extract",
        json={"document": "Invoice INV-001", "language": "en"},
    )
    assert response.status_code == 503


def test_chat_completions_internal_error_returns_500(client: TestClient) -> None:
    with patch.object(
        client.app.state.inference,
        "generate",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "docextract-qwen3-4b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
    assert response.status_code == 500
