"""Các chiến lược chunking để so sánh: Recursive (4 cấu hình) và Semantic (3 ngưỡng).

SemanticChunker được cài lại tại đây (cùng thuật toán với langchain_experimental) để tránh xung đột
phiên bản `langchain-community` giữa `langchain-experimental` và `ragas`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
]

_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]

_DEFAULT_AMOUNT = {"percentile": 95.0, "standard_deviation": 3.0, "interquartile": 1.5}
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")

BASELINE_STRATEGY_ID = "rc_1000_150"


@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict[str, object]

    @property
    def method(self) -> str:
        return "Semantic" if self.strategy_id.startswith("semantic") else "Recursive"

    @property
    def label(self) -> str:
        if self.method == "Recursive":
            return f"{self.params['chunk_size']}/{self.params['chunk_overlap']}"
        return {"percentile": "Percentile", "standard_deviation": "Std_dev", "interquartile": "Interquartile"}[
            str(self.params["breakpoint_type"])
        ]


@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] | None = None

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)


@dataclass(frozen=True)
class SemanticChunkerWrapper:
    """Chia theo độ tương đồng ngữ nghĩa giữa các câu liên tiếp.

    Điểm cắt = nơi khoảng cách cosine giữa 2 câu kề nhau vượt ngưỡng:
      - percentile:          phân vị thứ `amount` (mặc định 95)
      - standard_deviation:  mean + amount * std (mặc định 3)
      - interquartile:       mean + amount * IQR (mặc định 1.5)
    """

    embeddings: Embeddings
    breakpoint_type: str = "percentile"
    breakpoint_amount: float | None = None
    buffer_size: int = 1
    max_chunk_chars: int = 3000  # chunk dài hơn sẽ bị cắt thêm để không vượt giới hạn embedding

    def _threshold(self, distances: np.ndarray) -> float:
        amount = self.breakpoint_amount if self.breakpoint_amount is not None else _DEFAULT_AMOUNT[self.breakpoint_type]
        if self.breakpoint_type == "percentile":
            return float(np.percentile(distances, amount))
        if self.breakpoint_type == "standard_deviation":
            return float(np.mean(distances) + amount * np.std(distances))
        if self.breakpoint_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            return float(np.mean(distances) + amount * (q3 - q1))
        raise ValueError(f"Unknown breakpoint_type '{self.breakpoint_type}'")

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) <= 1:
            return self._cap([text])

        n, b = len(sentences), self.buffer_size
        combined = [" ".join(sentences[max(0, i - b) : min(n, i + b + 1)]) for i in range(n)]
        vecs = np.asarray(self.embeddings.embed_documents(combined), dtype=float)
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        distances = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)

        threshold = self._threshold(distances)
        chunks, start = [], 0
        for idx, dist in enumerate(distances):
            if dist > threshold:
                chunks.append(" ".join(sentences[start : idx + 1]))
                start = idx + 1
        chunks.append(" ".join(sentences[start:]))
        return self._cap([c for c in chunks if c.strip()])

    def _cap(self, chunks: list[str]) -> list[str]:
        out: list[str] = []
        fallback = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_chars // 2, chunk_overlap=100, separators=DEFAULT_SEPARATORS
        )
        for c in chunks:
            out.extend(fallback.split_text(c) if len(c) > self.max_chunk_chars else [c])
        return out

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            for piece in self.split_text(doc.page_content):
                result.append(Document(page_content=piece, metadata=dict(doc.metadata)))
        return result


def get_strategies(
    embeddings: Embeddings | None = None,
    ids: list[str] | None = None,
) -> list[ChunkingStrategy]:
    """Trả về danh sách chiến lược (lọc theo `ids` nếu có). Semantic cần `embeddings`."""
    strategies: list[ChunkingStrategy] = []
    for sid, size, overlap in _RECURSIVE_CONFIGS:
        strategies.append(
            ChunkingStrategy(
                sid,
                RecursiveChunker(chunk_size=size, chunk_overlap=overlap),
                {"chunk_size": size, "chunk_overlap": overlap},
            )
        )

    wanted_semantic = [s for s in _SEMANTIC_CONFIGS if ids is None or s[0] in ids]
    if wanted_semantic:
        if embeddings is None:
            from ..store import get_embeddings

            embeddings = get_embeddings()
        for sid, btype in wanted_semantic:
            strategies.append(
                ChunkingStrategy(
                    sid,
                    SemanticChunkerWrapper(embeddings=embeddings, breakpoint_type=btype),
                    {"breakpoint_type": btype},
                )
            )

    if ids is not None:
        unknown = set(ids) - {s.strategy_id for s in strategies}
        if unknown:
            raise ValueError(f"Chiến lược không tồn tại: {sorted(unknown)}")
        strategies = [s for s in strategies if s.strategy_id in ids]
    return strategies
