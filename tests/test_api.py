import pytest
from fastapi.testclient import TestClient

from src.interfaces.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_and_empty_documents(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/documents").json() == []


def test_upload_then_use_every_endpoint(client, sample_pdf, fake_llm):
    r = client.post("/upload", files={"file": ("sample.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200 and r.json()["chunks_indexed"] > 0

    docs = client.get("/documents").json()
    assert docs[0]["filename"] == "sample.pdf" and docs[0]["pages"] == 3

    r = client.post("/ask", json={"question": "What is RAG?", "filters": {"filenames": ["sample.pdf"]}})
    assert r.status_code == 200 and r.json()["citations"]

    assert client.post("/summarize", json={"document": "sample.pdf"}).json()["summary"]
    assert len(client.post("/quiz", json={"count": 5}).json()["items"]) == 2
    assert len(client.post("/flashcards", json={}).json()["cards"]) == 2


def test_error_mapping(client, sample_pdf, fake_llm, monkeypatch):
    client.post("/upload", files={"file": ("sample.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert client.post("/quiz", json={"document": "missing.pdf"}).status_code == 404
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/upload", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 422
    monkeypatch.setattr("src.learning.invoke_llm", lambda p, provider=None: "no json")
    assert client.post("/flashcards", json={}).status_code == 502
