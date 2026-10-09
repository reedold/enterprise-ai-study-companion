"""Lazily-created, process-wide singletons for external clients and the embedding model."""

import logging
from functools import lru_cache

from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@lru_cache
def get_groq() -> Groq:
    return Groq(api_key=get_settings().groq_api_key, timeout=60.0, max_retries=2)


@lru_cache
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    logger.info("Loading embedding model %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = get_dim()
    if dim != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding model dimension {dim} does not match DB vector({settings.embedding_dim})"
        )
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Return L2-normalised embeddings (cosine similarity == dot product)."""
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(float).tolist()
