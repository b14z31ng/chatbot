"""Gemini LLM generator via LangChain ChatGoogleGenerativeAI.

Same public interface as before:
    generate_answer(prompt, request_id, max_output_tokens) -> str
    LLM_UNAVAILABLE, LLM_RATE_LIMITED sentinels preserved.
"""
from __future__ import annotations

import logging
import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

logger = logging.getLogger(__name__)

# gemini-2.5-flash: best quality on free tier; lite falls back if quota hit
MODEL_NAME = "gemini-2.5-flash"

LLM_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
LLM_RATE_LIMITED = "LLM_RATE_LIMITED"

# Token limits — simple raised so entity names are never truncated
TOKENS_SIMPLE = 150
TOKENS_COMPLEX = 400

_RETRY_WAIT_SECONDS = 2

# Lazy-init: don't create the client until first call (avoids startup errors
# if GEMINI_API_KEY is set after import)
_LLM_CACHE: dict[int, ChatGoogleGenerativeAI] = {}


def _get_llm(max_output_tokens: int) -> ChatGoogleGenerativeAI:
    if max_output_tokens not in _LLM_CACHE:
        _LLM_CACHE[max_output_tokens] = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
            max_output_tokens=max_output_tokens,
        )
    return _LLM_CACHE[max_output_tokens]


def generate_answer(
    prompt: str,
    request_id: str = "-",
    max_output_tokens: int = TOKENS_COMPLEX,
) -> str:
    """Generate a grounded answer via LangChain + Gemini.

    Returns:
        Answer text, or one of:
        - LLM_UNAVAILABLE  → hard service error
        - LLM_RATE_LIMITED → 429 quota exhausted after retry
    """
    llm = _get_llm(max_output_tokens)

    for attempt in range(2):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (response.content or "").strip()

            logger.debug(
                "llm.response",
                extra={
                    "preview": text[:120],
                    "request_id": request_id,
                    "attempt": attempt,
                    "max_output_tokens": max_output_tokens,
                    "model": MODEL_NAME,
                },
            )
            return text

        except ResourceExhausted:
            if attempt == 0:
                logger.warning(
                    "llm.rate_limited_retrying",
                    extra={"request_id": request_id, "wait_s": _RETRY_WAIT_SECONDS},
                )
                time.sleep(_RETRY_WAIT_SECONDS)
                continue
            logger.warning("llm.rate_limited_giving_up", extra={"request_id": request_id})
            return LLM_RATE_LIMITED

        except ServiceUnavailable:
            logger.warning("llm.service_unavailable", extra={"request_id": request_id})
            return LLM_UNAVAILABLE

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                if attempt == 0:
                    logger.warning(
                        "llm.rate_limited_retrying_generic",
                        extra={"request_id": request_id, "wait_s": _RETRY_WAIT_SECONDS},
                    )
                    time.sleep(_RETRY_WAIT_SECONDS)
                    continue
                return LLM_RATE_LIMITED
            logger.exception("llm.error", extra={"request_id": request_id, "model": MODEL_NAME})
            return LLM_UNAVAILABLE

    return LLM_UNAVAILABLE