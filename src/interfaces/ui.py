"""Giao diện Streamlit. Chạy (sau khi bật API): streamlit run src/interfaces/ui.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # để import được `src.*`

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import settings  # noqa: E402
from src.export import export  # noqa: E402
from src.interfaces.styles import GLOBAL_CSS  # noqa: E402
from src.schemas import FlashcardSet, QuizSet, RagAnswer, Summary  # noqa: E402

_API = settings.api_url
_LETTERS = "ABCD"


def _api(method: str, path: str, **kwargs):
    return httpx.request(method, f"{_API}{path}", timeout=300, **kwargs)


def _call(method: str, path: str, **kwargs):
    """Gọi API, hiển thị lỗi thân thiện; trả về JSON hoặc None."""
    try:
        resp = _api(method, path, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Không kết nối được API ({_API}): {exc}. Hãy chạy: uvicorn src.interfaces.api:app --port 8000")
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        st.error(f"Lỗi {resp.status_code}: {detail}")
        return None
    return resp.json()


def _scope_payload(filenames: list[str], page: int) -> dict:
    filters: dict = {}
    if filenames:
        filters["filenames"] = filenames
    if page and len(filenames) == 1:
        filters["page"] = page
    return {"filters": filters or None}


def _sources(citations) -> None:
    if citations:
        chips = "".join(
            f'<span class="src-chip">[{c["source_marker"]}] {c["filename"]}, tr. {c["page"]}</span>' for c in citations
        )
        st.markdown(chips, unsafe_allow_html=True)


def _downloads(model, stem: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Tải JSON", export(model, fmt="json"), f"{stem}.json", "application/json", key=f"{stem}-json")
    c2.download_button("Tải Markdown", export(model, fmt="md"), f"{stem}.md", "text/markdown", key=f"{stem}-md")


def _sidebar() -> tuple[list[str], int]:
    st.sidebar.header("Tài liệu")
    up = st.sidebar.file_uploader("Tải lên PDF", type=["pdf"])
    if up is not None and st.sidebar.button("Index tài liệu", use_container_width=True):
        with st.spinner("Đang đọc, chia chunk và vector hóa..."):
            res = _call("POST", "/upload", files={"file": (up.name, up.getvalue(), "application/pdf")})
        if res:
            st.sidebar.success(f"{res['filename']}: {res['chunks_indexed']} chunks")

    docs = _call("GET", "/documents") or []
    names = [d["filename"] for d in docs]
    for d in docs:
        st.sidebar.caption(f"• {d['filename']} — {d['pages']} trang, {d['chunks']} chunks")

    chosen = st.sidebar.multiselect("Phạm vi (để trống = tất cả)", names)
    page = st.sidebar.number_input("Trang (0 = tất cả, chỉ khi chọn 1 file)", min_value=0, value=0, step=1)
    return chosen, int(page)


def _tab_chat(filenames, page):
    rerank = st.toggle("Bật Cross-Encoder reranking", value=False)
    if "history" not in st.session_state:
        st.session_state.history = []
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            _sources(item["citations"])

    question = st.chat_input("Đặt câu hỏi về tài liệu...")
    if question:
        payload = {"question": question, "rerank": rerank, **_scope_payload(filenames, page)}
        with st.spinner("Đang truy xuất và trả lời..."):
            res = _call("POST", "/ask", json=payload)
        if res:
            st.session_state.history.append(res)
            st.rerun()
    if st.session_state.history:
        last = RagAnswer.model_validate(st.session_state.history[-1])
        with st.expander("Ngữ cảnh đã dùng"):
            for i, c in enumerate(last.chunks, start=1):
                st.markdown(f"**[S{i}]** {c.metadata.filename}, trang {c.metadata.page}")
                st.caption(c.text)


def _tab_summary(filenames, page):
    query = st.text_input("Chủ đề cần tóm tắt (tùy chọn)", key="sum-q")
    if st.button("Tóm tắt", key="sum-btn"):
        payload = {"query": query or None, **_scope_payload(filenames, page)}
        with st.spinner("Đang tóm tắt..."):
            res = _call("POST", "/summarize", json=payload)
        if res:
            st.session_state.summary = res
    if "summary" in st.session_state:
        s = st.session_state.summary
        st.markdown(f'<div class="card">{s["summary"]}</div>', unsafe_allow_html=True)
        if s["key_points"]:
            st.subheader("Ý chính")
            for p in s["key_points"]:
                st.markdown(f"- {p}")
        _sources(s["citations"])
        _downloads(Summary.model_validate(s), "summary")


def _tab_quiz(filenames, page):
    c1, c2 = st.columns([1, 2])
    count = c1.slider("Số câu", 1, 20, settings.quiz_default_count, key="quiz-n")
    query = c2.text_input("Chủ đề (tùy chọn)", key="quiz-q")
    if st.button("Tạo quiz", key="quiz-btn"):
        payload = {"query": query or None, "count": count, **_scope_payload(filenames, page)}
        with st.spinner("Đang tạo câu hỏi..."):
            res = _call("POST", "/quiz", json=payload)
        if res:
            st.session_state.quiz = res
    if "quiz" in st.session_state:
        quiz = QuizSet.model_validate(st.session_state.quiz)
        answers = {}
        for i, item in enumerate(quiz.items):
            answers[i] = st.radio(
                f"**Câu {i + 1}.** {item.question}",
                options=list(range(4)),
                format_func=lambda j, it=item: f"{_LETTERS[j]}. {it.options[j]}",
                index=None,
                key=f"quiz-ans-{i}",
            )
        if st.button("Chấm điểm", key="quiz-grade"):
            score = 0
            for i, item in enumerate(quiz.items):
                ok = answers[i] == item.correct_index
                score += ok
                mark = '<span class="ok">Đúng</span>' if ok else '<span class="bad">Sai</span>'
                st.markdown(
                    f"Câu {i + 1}: {mark} — đáp án {_LETTERS[item.correct_index]}. {item.explanation}",
                    unsafe_allow_html=True,
                )
            st.success(f"Điểm: {score}/{len(quiz.items)}")
        _downloads(quiz, "quiz")


def _tab_flashcards(filenames, page):
    c1, c2 = st.columns([1, 2])
    count = c1.slider("Số thẻ", 1, 30, settings.flashcards_default_count, key="fc-n")
    query = c2.text_input("Chủ đề (tùy chọn)", key="fc-q")
    if st.button("Tạo flashcards", key="fc-btn"):
        payload = {"query": query or None, "count": count, **_scope_payload(filenames, page)}
        with st.spinner("Đang tạo flashcards..."):
            res = _call("POST", "/flashcards", json=payload)
        if res:
            st.session_state.flashcards = res
    if "flashcards" in st.session_state:
        deck = FlashcardSet.model_validate(st.session_state.flashcards)
        for i, card in enumerate(deck.cards):
            with st.expander(f"Thẻ {i + 1}: {card.front}"):
                if card.hint:
                    st.caption(f"Gợi ý: {card.hint}")
                st.markdown(f"**{card.back}**")
        _downloads(deck, "flashcards")


def run():
    st.set_page_config(page_title="RAG Learning System", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.title("Simple NotebookLM")

    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)


if __name__ == "__main__":
    run()
