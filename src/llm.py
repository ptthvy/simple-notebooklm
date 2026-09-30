"""Lớp trung gian gọi LLM: hf_local | gemini | vllm. Các module khác chỉ gọi invoke_llm()."""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage

from .config import settings

log = logging.getLogger(__name__)


def _build_hf_local():
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    use_cuda = settings.hf_device >= 0 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, dtype=dtype)

    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device if use_cuda else -1,
        return_full_text=False,
    )
    text_gen.generation_config.max_new_tokens = settings.hf_max_new_tokens
    text_gen.generation_config.do_sample = settings.llm_temperature > 0
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError("Thiếu GOOGLE_API_KEY (đặt trong .env) khi llm_provider='gemini'.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.vllm_model or settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    provider = provider or settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    if provider == "vllm":
        return _build_vllm()
    raise ValueError(f"Unknown llm_provider '{provider}'")


def _content_to_text(content) -> str:
    """Một số model (vd. Gemini) trả content là list các part."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(str(p.get("text", "")))
        return "".join(parts)
    return str(content)


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    """Gọi LLM, tự retry với backoff khi lỗi tạm thời (rate limit, mạng...)."""
    llm = get_llm(provider=provider)
    attempts = settings.llm_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return _content_to_text(response.content)
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            wait = 2**attempt
            log.warning("LLM lỗi (%s), thử lại sau %ss [%d/%d]", exc, wait, attempt, attempts - 1)
            time.sleep(wait)
    raise RuntimeError("unreachable")
