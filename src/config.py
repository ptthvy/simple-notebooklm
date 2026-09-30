"""Cấu hình tập trung. Ghi đè bằng biến môi trường RAG_* hoặc file .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_prefix="RAG_",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Đường dẫn dữ liệu ---
    data_dir: Path = ROOT / "data"
    storage_dir: Path = ROOT / "storage" / "qdrant"
    qdrant_url: str | None = None  # nếu đặt: dùng Qdrant server thay vì chế độ local
    qdrant_collection: str = "rag_chunks"
    max_upload_mb: int = Field(default=50, ge=1)

    # --- Chunking & retrieval ---
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    # --- Reranking (Cross-Encoder) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_initial_k: int = Field(default=15, ge=1, le=128)
    rerank_top_k: int = Field(default=5, ge=1, le=64)

    # --- Embedding ---
    embedding_model: str = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"

    # --- LLM ---
    llm_provider: Literal["hf_local", "gemini", "vllm"] = "gemini"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    hf_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    hf_device: int = -1  # -1 = CPU, >= 0 = CUDA index
    hf_max_new_tokens: int = Field(default=2048, ge=1)

    gemini_model: str = "gemini-3.6-flash"
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str | None = None  # mặc định dùng hf_model

    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Học liệu ---
    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    summarize_max_chunks: int = Field(default=200, ge=1)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    max_generation_chunks: int = Field(default=24, ge=1)
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)

    api_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.hf_device < -1:
            raise ValueError("hf_device must be -1 for CPU or >= 0 for CUDA.")
        if self.rerank_top_k > self.rerank_initial_k:
            raise ValueError("rerank_top_k must be <= rerank_initial_k.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
