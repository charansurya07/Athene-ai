"""
Athene AI — Multimodal 7-Agent Research Engine
FastAPI application entry point.

Run with:
    uvicorn research_ai_backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_ai_backend.config import get_settings
from research_ai_backend.config import router as research_router
from research_ai_backend.config import router as voice_router

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("athene_ai")

app = FastAPI(
    title="Athene AI — Multimodal 7-Agent Research Engine",
    description=(
        "FastAPI + LangGraph backend powering the Athene AI frontend."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routers here
app.include_router(research_router)
app.include_router(voice_router)


@app.on_event("startup")
async def on_startup():
    logger.info("Athene AI backend starting up (env=%s)", settings.app_env)

    if not settings.anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY is not set — agent LLM calls will fail until it is configured."
        )


@app.get("/")
async def root():
    return {
        "service": "Athene AI backend",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}