"""Application entrypoint.  Run with:  uvicorn main:app --reload"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.core.clients import get_embedder, get_groq, get_supabase
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.services.ocr_service import ocr_available

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings()  # fail fast on missing env vars
    get_supabase()
    get_groq()
    await run_in_threadpool(get_embedder)  # warm up models so first request is fast
    if get_settings().ocr_enabled:
        await run_in_threadpool(ocr_available)
    yield


app = FastAPI(
    title="AI-Powered B2B Knowledge & Exam Preparation Engine",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = get_settings().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # Browsers reject credentials with a wildcard origin.
    allow_credentials="*" not in _origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)

# Serve the web UI at "/" (mounted last so API and /docs routes take precedence).
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
