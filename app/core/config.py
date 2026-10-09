"""Application settings loaded from environment variables / .env."""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    supabase_url: str = field(default_factory=lambda: _require("SUPABASE_URL"))
    supabase_service_role_key: str = field(
        default_factory=lambda: _require("SUPABASE_SERVICE_ROLE_KEY")
    )
    groq_api_key: str = field(default_factory=lambda: _require("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = 384

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "75")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "25")))
    insert_batch_size: int = 100

    ocr_enabled: bool = field(
        default_factory=lambda: os.getenv("OCR_ENABLED", "true").lower() in {"1", "true", "yes"}
    )
    # EasyOCR language codes, e.g. "en", "en,az", "en,ru", "en,tr"
    ocr_languages: list[str] = field(default_factory=lambda: _csv("OCR_LANGUAGES", "en"))
    ocr_dpi: int = field(default_factory=lambda: int(os.getenv("OCR_DPI", "200")))
    ocr_max_pages: int = field(default_factory=lambda: int(os.getenv("OCR_MAX_PAGES", "50")))

    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "8")))
    rag_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_MATCH_THRESHOLD", "0.30"))
    )
    rag_max_context_chars: int = 12_000

    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "*"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
