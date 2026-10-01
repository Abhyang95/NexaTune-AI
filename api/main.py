import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
)
from api.inference_service import inference_service
from api.logging_config import configure_logging, logger


# ============================================================
# Configuration
# ============================================================

configure_logging()


def get_cors_origins() -> list[str]:
    """
    Read allowed CORS origins from the environment.

    Example:
        NEXATUNE_CORS_ORIGINS=http://localhost:7860,http://127.0.0.1:7860

    If the environment variable is not set, preserve the
    existing local Gradio development configuration.
    """

    configured_origins = os.getenv(
        "NEXATUNE_CORS_ORIGINS",
        "http://localhost:7860,http://127.0.0.1:7860",
    )

    origins = [
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    ]

    return origins


CORS_ORIGINS = get_cors_origins()


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="NexaTune AI API",
    description="Domain-adapted customer support LLM inference API",
    version="1.0.0",
)


# ============================================================
# Exception Handling
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "Request validation failed | method=%s | path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "message": (
                "Customer message must be between "
                "1 and 4000 characters."
            ),
        },
    )


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# Request Logging
# ============================================================

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        latency = time.perf_counter() - start_time

        logger.info(
            "HTTP request | method=%s | path=%s | status=%d | latency=%.2fs",
            request.method,
            request.url.path,
            response.status_code,
            latency,
        )

        return response

    except Exception:
        latency = time.perf_counter() - start_time

        logger.exception(
            "HTTP request failed | method=%s | path=%s | latency=%.2fs",
            request.method,
            request.url.path,
            latency,
        )

        raise


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():
    logger.info(
        "NexaTune API startup | CORS origins=%s",
        CORS_ORIGINS,
    )

    inference_service.load()


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "NexaTune AI",
        "model_loaded": inference_service.loaded,
    }


# ============================================================
# Model Information
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "base_model": "Qwen/Qwen3-1.7B-Base",
        "fine_tuning": "QLoRA",
        "lora_rank": 16,
        "quantization": "4-bit NF4",
        "domain": "Customer Support",
    }


# ============================================================
# Generation
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(request: GenerateRequest):

    if not inference_service.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Fine-tuned NexaTune adapter is not "
                "available yet."
            ),
        )

    result = inference_service.generate(
        request.message
    )

    return GenerateResponse(
        response=result["response"],
        model="NexaTune-Qwen3-1.7B",
        guardrail_flagged=result["guardrail_flagged"],
        guardrail_matches=result["guardrail_matches"],
        guardrail_categories=result["guardrail_categories"],
        original_response=result["original_response"],
    )