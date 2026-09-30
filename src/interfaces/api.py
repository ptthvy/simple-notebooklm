"""REST API bằng FastAPI. Chạy: uvicorn src.interfaces.api:app --port 8000"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..filters import MetadataFilter, filters_to_dict
from ..indexing import save_and_ingest_pdf
from ..learning import LLMOutputError, NoContentError
from ..learning import generate_flashcards as generate_flashcards_learning
from ..learning import generate_quiz as generate_quiz_learning
from ..learning import summarize as summarize_learning
from ..rag import answer
from ..schemas import DocumentInfo, FlashcardSet, QuizSet, RagAnswer, Summary, UploadResponse
from ..store import list_documents


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None
    rerank: bool = False


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    pass


app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)


@app.exception_handler(NoContentError)
async def _no_content(_, exc: NoContentError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(LLMOutputError)
async def _bad_llm_output(_, exc: LLMOutputError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def _runtime_error(_, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Chỉ hỗ trợ file .pdf")
    content = await file.read()
    return await run_in_threadpool(save_and_ingest_pdf, content, file.filename or "")


@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(req.question, k=req.k, filters=filters_to_dict(req.filters), rerank=req.rerank)


@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    return summarize_learning(
        document=req.document, query=req.query, filters=filters_to_dict(req.filters), k=req.k
    )


@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    return generate_quiz_learning(
        document=req.document, query=req.query, filters=filters_to_dict(req.filters), count=req.count, k=req.k
    )


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest):
    return generate_flashcards_learning(
        document=req.document, query=req.query, filters=filters_to_dict(req.filters), count=req.count, k=req.k
    )
