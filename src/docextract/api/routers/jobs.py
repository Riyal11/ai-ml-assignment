"""FastAPI router for background job management."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from docextract.api.models import JobStatusResponse, QuantizeJobRequest, QuantizeJobResponse
from docextract.jobs.quantize_task import (
    celery_enabled,
    quantize_model_task,
    run_quantize_sync,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

_SYNC_JOBS: dict[str, dict[str, Any]] = {}
_SUPPORTED_METHODS = {"gguf", "gguf-q4_k_m", "awq", "gptq"}


@router.post("/quantize", response_model=QuantizeJobResponse)
async def create_quantize_job(request: QuantizeJobRequest) -> QuantizeJobResponse:
    """Queue or run a model quantization job.

    Args:
        request: Quantization parameters.

    Returns:
        Job identifier and initial status.

    Raises:
        HTTPException: 400 for unsupported methods, 500 for internal failures.
    """
    if request.method not in _SUPPORTED_METHODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported quantization method: {request.method}",
        )

    try:
        if celery_enabled():
            async_result = quantize_model_task.apply_async(
                args=[request.model_path, request.output_dir, request.method],
            )
            return QuantizeJobResponse(job_id=async_result.id, status="queued")

        job_id = uuid.uuid4().hex
        result = run_quantize_sync(request.model_path, request.output_dir, request.method)
        _SYNC_JOBS[job_id] = {"status": "SUCCESS", "result": result, "error": None}
        return QuantizeJobResponse(job_id=job_id, status="completed")
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create quantization job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to start quantization job",
        ) from exc


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the status of a quantization job.

    Args:
        job_id: Celery task id or synchronous job id.

    Returns:
        Job status and optional result payload.

    Raises:
        HTTPException: 404 when the job id is unknown.
    """
    if job_id in _SYNC_JOBS:
        record = _SYNC_JOBS[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=record["status"],
            result=record.get("result"),
            error=record.get("error"),
        )

    if celery_enabled():
        from celery.result import AsyncResult

        from docextract.jobs.quantize_task import celery_app

        async_result = AsyncResult(job_id, app=celery_app)
        if async_result.state == "PENDING" and async_result.result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"job not found: {job_id}",
            )
        if async_result.failed():
            return JobStatusResponse(
                job_id=job_id,
                status="FAILURE",
                result=None,
                error=str(async_result.result),
            )
        if async_result.successful():
            return JobStatusResponse(
                job_id=job_id,
                status="SUCCESS",
                result=async_result.result,
                error=None,
            )
        return JobStatusResponse(job_id=job_id, status=async_result.state, result=None, error=None)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"job not found: {job_id}")
