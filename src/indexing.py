"""Nạp PDF -> chia chunk -> vector hóa -> lưu Qdrant."""
from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .schemas import ChunkMetadata
from .store import ensure_collection, get_vector_store


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or settings.data_dir
    return sorted(p for p in root.glob("*.pdf") if p.is_file())


def _document_id(path: Path) -> str:
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path):
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)

    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def _splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def build_chunks(pdf_paths, chunk_size=None, chunk_overlap=None, chunker=None):
    """Đọc PDF và chia chunk. `chunker` là object có `split_documents` (dùng cho evaluation)."""
    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(Path(path)))
    page_docs = [d for d in page_docs if d.page_content and d.page_content.strip()]

    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    per_doc_counter: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1

        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks, collection_name: str | None = None) -> int:
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def ingest(recreate=False, collection_name=None, chunker=None, chunk_size=None, chunk_overlap=None) -> int:
    pdfs = discover_pdfs()
    if not pdfs:
        raise FileNotFoundError(f"Không có file PDF nào trong {settings.data_dir}")
    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(pdfs, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return index_chunks(chunks, collection_name=collection_name)


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Chỉ hỗ trợ file .pdf")
    if len(file_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError(f"File vượt quá {settings.max_upload_mb} MB")

    dest = settings.data_dir / safe_name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)

    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    if not chunks:
        raise ValueError("PDF không có text để index (có thể là file scan, cần OCR).")
    return {"filename": safe_name, "chunks_indexed": index_chunks(chunks)}
