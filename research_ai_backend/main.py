"""
Athene AI — Multimodal 7-Agent Research Engine
FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as research_router
from app.api.websocket_voice import router as voice_router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("athene_ai")

app = FastAPI(
    title="Athene AI — Multimodal 7-Agent Research Engine",
    description=(
        "FastAPI + LangGraph backend powering the Athene AI frontend: "
        "ingestion, planning, retrieval, verification, recommendation, "
        "knowledge-graph extraction and report writing."
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

app.include_router(research_router)
app.include_router(voice_router)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Athene AI backend starting up (env=%s)", settings.app_env)
    if not settings.anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY is not set — agent LLM calls will fail until it is configured in .env"
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Athene AI backend",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
