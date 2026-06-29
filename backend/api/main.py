"""
FastAPI application entry point for the GraphRAG Procurement API.

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

# SSL bypass MUST be first — before any library that makes HTTPS connections.
# Defined once in core/ssl_patch.py; importing it applies the patch.
import core.ssl_patch  # noqa: F401


import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import pipeline, graph, traceability, chat, status, workspaces

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GraphRAG Procurement API",
    version="2.0.0",
    description="Multi-workspace knowledge graph API for procurement document analysis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workspaces.router)
app.include_router(pipeline.router)
app.include_router(graph.router)
app.include_router(traceability.router)
app.include_router(chat.router)
app.include_router(status.router)


@app.on_event("startup")
async def startup() -> None:
    try:
        from db.postgres import init_db
        await asyncio.to_thread(init_db)
        logger.info("PostgreSQL workspace table ready")
    except Exception as exc:
        logger.warning("Could not initialize PostgreSQL (workspace CRUD unavailable): %s", exc)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "graphrag-api", "version": "2.0.0"}
