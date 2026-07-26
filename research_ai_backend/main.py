"""
Athene AI — FastAPI Backend
Run with:
    uvicorn main:app --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("athene_ai")

app = FastAPI(
    title="Athene AI",
    description="Multimodal AI Research Backend",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Athene AI Backend Started Successfully")


@app.get("/")
async def root():
    return {
        "service": "Athene AI Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "Backend is running successfully"
    }


@app.get("/api/v1/info")
async def info():
    return {
        "name": "Athene AI",
        "version": "0.1.0",
        "framework": "FastAPI"
    }