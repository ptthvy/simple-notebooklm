"""Embedding + Qdrant vector store."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterator

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings
from .schemas import DocumentInfo

INDEXED_PAYLOAD_FIELDS = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}


def _device_str() -> str:
    return "cpu" if settings.hf_device < 0 else f"cuda:{settings.hf_device}"


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": _device_str()},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir))


def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name or settings.qdrant_collection,
        embedding=get_embeddings(),
    )


def ensure_collection(recreate: bool = False, collection_name: str | None = None) -> None:
    client = get_client()
    name = collection_name or settings.qdrant_collection
    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    payload_schema = client.get_collection(name).payload_schema or {}
    for field, schema in INDEXED_PAYLOAD_FIELDS.items():
        if payload_schema.get(field) is None:
            client.create_payload_index(name, field_name=field, field_schema=schema)


def collection_exists(collection_name: str | None = None) -> bool:
    return get_client().collection_exists(collection_name or settings.qdrant_collection)


def collection_count(collection_name: str | None = None) -> int:
    name = collection_name or settings.qdrant_collection
    if not get_client().collection_exists(name):
        return 0
    return get_client().count(collection_name=name, exact=True).count


def scroll_all(
    collection_name: str, scroll_filter: qmodels.Filter | None = None, batch: int = 256
) -> Iterator[list]:
    """Duyệt toàn bộ point (theo trang) thỏa filter."""
    client = get_client()
    if not client.collection_exists(collection_name):
        return
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            yield points
        if offset is None:
            break


def list_documents(collection_name: str | None = None) -> list[DocumentInfo]:
    name = collection_name or settings.qdrant_collection
    pages: dict[str, set[int]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    names: dict[str, str] = {}

    for batch in scroll_all(name):
        for point in batch:
            meta = (point.payload or {}).get("metadata") or {}
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            names[doc_id] = meta.get("filename", doc_id)
            pages[doc_id].add(int(meta.get("page", 0)))
            counts[doc_id] += 1

    return sorted(
        (
            DocumentInfo(filename=names[d], document_id=d, pages=len(pages[d]), chunks=counts[d])
            for d in counts
        ),
        key=lambda x: x.filename.lower(),
    )
