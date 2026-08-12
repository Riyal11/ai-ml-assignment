"""Pydantic models for the OpenAI-compatible API and extraction endpoint."""

from typing import Any

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    messages: list[dict[str, str]]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float = 0.7
    response_format: dict[str, str] | None = None


class Usage(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """A single non-streaming completion choice."""

    index: int
    message: dict[str, str]
    finish_reason: str | None = None


class ChoiceStream(BaseModel):
    """A single streaming completion chunk."""

    index: int
    delta: dict[str, str]
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response body."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ExtractRequest(BaseModel):
    """Request body for the invoice extraction endpoint."""

    document: str
    language: str = "en"
    schema_strict: bool = True


class ExtractResponse(BaseModel):
    """Response body for the invoice extraction endpoint."""

    invoice: dict[str, Any] | None
    raw_output: str
    is_valid: bool
    validation_errors: list[str]


class QuantizeJobRequest(BaseModel):
    """Request body for async quantization jobs."""

    model_path: str
    method: str = "gguf-q4_k_m"
    output_dir: str = "artifacts/gguf"


class QuantizeJobResponse(BaseModel):
    """Response when a quantization job is queued or completed."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response for quantization job status polling."""

    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_loaded: bool


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[dict[str, Any]] = Field(default_factory=list)
