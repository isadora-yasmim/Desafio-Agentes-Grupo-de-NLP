"""
config.py / settings.py
-----------------------
Configurações centrais carregadas de variáveis de ambiente (.env).
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # ── Embeddings ───────────────────────────────────────────────────────────
    # "openai" usa text-embedding-3-small (1536 dims)
    # "huggingface" usa multilingual-e5-base (768 dims)
    EMBEDDING_BACKEND: str = "openai"
    EMBEDDING_DIMENSIONS: int = 1536  # mude para 768 se usar HuggingFace

    # ── Supabase ─────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""   # service_role key (não a anon!)
    SUPABASE_TABLE: str = "documents"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "documents"
    QDRANT_API_KEY: str | None = None

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_K_SEMANTIC: int = 20   # chunks para busca semântica
    RETRIEVAL_K_BM25: int = 20       # chunks para BM25
    RETRIEVAL_K_FINAL: int = 5       # chunks após reranking
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"   # gratuito

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_WINDOW_SIZE: int = 3
    CHUNK_WINDOW_OVERLAP: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Instância global
settings = get_settings()
