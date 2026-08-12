"""FastAPI application with OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from docextract.api.inference import (
    InferenceService,
    set_inference_service,
)
from docextract.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    ModelListResponse,
    Usage,
)
from docextract.api.routers import jobs as jobs_router

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(os.getenv("DOCEXTRACT_MODEL_PATH", "artifacts/merged-model"))
_DEFAULT_MODEL_ID = os.getenv("DOCEXTRACT_MODEL_ID", "docextract-qwen3-4b")
_DEFAULT_QUANTIZATION = os.getenv("DOCEXTRACT_QUANTIZATION", "none")


def _require_loaded_service(request: Request) -> InferenceService:
    """Return the app inference service or raise HTTP 503.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Loaded inference service.

    Raises:
        HTTPException: When the model is not loaded.
    """
    service = getattr(request.app.state, "inference", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )
    return cast(InferenceService, service)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the inference model on startup and release it on shutdown."""
    model_path = Path(app.state.model_path)
    service = InferenceService(model_path=model_path, quantization=app.state.quantization)
    try:
        await service.load()
    except RuntimeError:
        logger.exception("Failed to load model from %s", model_path)
        service = InferenceService(model_path=model_path, quantization=app.state.quantization)
    app.state.inference = service
    set_inference_service(service)
    yield
    if hasattr(app.state.inference, "unload"):
        await app.state.inference.unload()


def create_app(
    model_path: Path | None = None,
    quantization: str | None = None,
    model_id: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_path: Optional override for model artifact path.
        quantization: Optional override for quantization backend.
        model_id: Optional override for served model identifier.

    Returns:
        Configured FastAPI application.
    """
    application = FastAPI(title="docextract", lifespan=lifespan)
    application.state.model_path = model_path or _DEFAULT_MODEL_PATH
    application.state.quantization = quantization or _DEFAULT_QUANTIZATION
    application.state.model_id = model_id or _DEFAULT_MODEL_ID
    application.include_router(jobs_router.router)

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Return service health and model load status."""
        service = getattr(request.app.state, "inference", None)
        return HealthResponse(
            status="ok",
            model_loaded=bool(service and service.is_loaded),
        )

    @application.get("/v1/models", response_model=ModelListResponse)
    async def list_models(request: Request) -> ModelListResponse:
        """Return the OpenAI-compatible model list."""
        model_id = request.app.state.model_id
        return ModelListResponse(
            data=[
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "docextract",
                }
            ]
        )

    @application.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        service: InferenceService = Depends(_require_loaded_service),  # noqa: B008
    ) -> Any:
        """Create a chat completion in OpenAI-compatible format."""
        max_tokens = body.max_tokens or 256
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"

        if body.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    service=service,
                    completion_id=completion_id,
                    request=body,
                    max_tokens=max_tokens,
                    model_id=body.model,
                ),
                media_type="text/event-stream",
            )

        try:
            output = await service.generate(
                messages=body.messages,
                max_tokens=max_tokens,
                temperature=body.temperature,
                stream=False,
            )
        except Exception as exc:
            logger.exception("Chat completion failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="chat completion failed",
            ) from exc

        if not isinstance(output, str):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="unexpected non-streaming response type",
            )

        prompt_tokens = sum(len(msg.get("content", "")) for msg in body.messages)
        completion_tokens = len(output)
        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message={"role": "assistant", "content": output},
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @application.post("/extract", response_model=ExtractResponse)
    async def extract(
        body: ExtractRequest,
        service: InferenceService = Depends(_require_loaded_service),  # noqa: B008
    ) -> ExtractResponse:
        """Extract invoice JSON from a document."""
        try:
            return await service.extract_invoice(body.document, body.language)
        except Exception as exc:
            logger.exception("Extraction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="extraction failed",
            ) from exc

    return application


async def _stream_chat_completion(
    service: InferenceService,
    completion_id: str,
    request: ChatCompletionRequest,
    max_tokens: int,
    model_id: str,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events for a streaming chat completion."""
    created = int(time.time())
    try:
        stream_result = await service.generate(
            messages=request.messages,
            max_tokens=max_tokens,
            temperature=request.temperature,
            stream=True,
        )
        if isinstance(stream_result, str):
            chunks = [stream_result]
        else:
            chunks = [chunk async for chunk in stream_result]

        for index, chunk in enumerate(chunks):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk} if index == 0 else {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        final_payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("Streaming chat completion failed")
        error_payload = {"error": "stream failed"}
        yield f"data: {json.dumps(error_payload)}\n\n"


app = create_app()
