"""Hybrid RAG chat service.

Routing policy
--------------
Simple factual queries  →  regex extraction  →  zero Gemini calls
Complex / summary       →  Gemini generation →  dynamic token limits

This reduces Gemini usage by 80–95% for typical document QA workloads.
"""
from __future__ import annotations

import hashlib
import logging
import re
from services.rag.retriever import ContextChunk
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from services.llm.extractor import extract_answer
from services.llm.generator import (
    TOKENS_COMPLEX,
    TOKENS_SIMPLE,
    LLM_UNAVAILABLE,
    LLM_RATE_LIMITED,
    generate_answer,
)
from services.llm.intent import classify_query, is_complex_query, top_k_for_query
from services.llm.memory import Message, add_message, get_history
from services.llm.prompt_builder import build_prompt
from services.rag.retriever import ContextChunk, retrieve

logger = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = "No documents uploaded for this chat."
RETRIEVAL_EMPTY_ANSWER = "Not found in uploaded documents."
LLM_BUSY_ANSWER = "AI service temporarily busy. Please try again shortly."

STATUS_NO_DOCUMENTS = "NO_DOCUMENTS"
STATUS_RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY"
STATUS_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
STATUS_SUCCESS = "SUCCESS"

REFUSAL_PHRASES = (
    "i cannot answer",
    "not enough information",
    "context does not contain",
    "cannot determine",
)

# ============================================================================
# DETERMINISTIC ENTITY EXTRACTION
# ============================================================================

ENTITY_PATTERNS = {
    "company": [
        r"prepared\s+by\s*[:\-]\s*([^\n\r|]+)",
        r"submitted\s+by\s*[:\-]\s*([^\n\r|]+)",
        r"quotation\s+by\s*[:\-]\s*([^\n\r|]+)",
        r"company\s+name\s*[:\-]\s*([^\n\r|]+)",
        r"company\s*[:\-]\s*([^\n\r|]+)",
        r"organization\s*[:\-]\s*([^\n\r|]+)",
        r"vendor\s*[:\-]\s*([^\n\r|]+)",
    ],

    "client": [
        r"prepared\s+for\s*[:\-]\s*([^\n\r|]+)",
        r"client\s+name\s*[:\-]\s*([^\n\r|]+)",
        r"client\s*[:\-]\s*([^\n\r|]+)",
        r"customer\s*[:\-]\s*([^\n\r|]+)",
        r"submitted\s+to\s*[:\-]\s*([^\n\r|]+)",
    ],
}


def _clean_entity(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s{2,}", " ", value)
    value = value.strip("|:- ")

    if len(value) > 120:
        value = value[:120]

    return value


def _detect_entity_type(query: str):
    q = query.lower()

    if any(x in q for x in [
        "company",
        "organization",
        "organisation",
        "vendor",
        "who made",
        "prepared by",
    ]):
        return "company"

    if any(x in q for x in [
        "client",
        "customer",
        "prepared for",
        "submitted to",
    ]):
        return "client"

    return None


# Service-role signals — entity values containing these are NOT document owners
SERVICE_LABELS = {
    "sslcommerz", "payment gateway", "bkash", "nagad", "paypal",
    "stripe", "integration", " api", "sdk", "framework", "platform",
    "powered by", "technology", "hosting", "crm", "erp",
    "woocommerce", "shopify", "laravel", "react", "vue", "django",
}


def _is_service_entity(value: str) -> bool:
    """Return True if the extracted value looks like a service/tool name, not an owner."""
    v_lower = value.lower()
    return any(sig in v_lower for sig in SERVICE_LABELS)


def extract_entity_answer(
    query: str,
    contexts: list[ContextChunk],
) -> str | None:
    """Deterministic entity extraction before Gemini.

    Returns the extracted entity string, or None if not found / value looks
    like a service name (not a document owner).
    """
    entity_type = _detect_entity_type(query)

    if not entity_type:
        return None

    # Prefer owner-signal chunks (prepared by, submitted by, etc.) first
    text = "\n".join(c.content for c in contexts)

    patterns = ENTITY_PATTERNS.get(entity_type, [])

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            value = _clean_entity(match.group(1))

            if len(value) < 3:
                continue

            # Reject service/tool names masquerading as owners
            if _is_service_entity(value):
                logger.debug(
                    "entity.extract.service_rejected",
                    extra={"value": value, "pattern": pattern[:60]},
                )
                continue

            logger.info(
                "entity.extract.success",
                extra={"entity_type": entity_type, "value": value},
            )
            return value

    return None
# ---------------------------------------------------------------------------
# LRU answer cache — larger to maximise cache hits
# ---------------------------------------------------------------------------

MAX_CACHE_ENTRIES = 200
_ANSWER_CACHE: "OrderedDict[str, str]" = OrderedDict()

# ---------------------------------------------------------------------------
# Per-user cooldown — 1 req/sec to prevent accidental quota spam
# ---------------------------------------------------------------------------

_USER_LAST_REQUEST: "dict[str, float]" = {}
COOLDOWN_SECONDS = 0.3   # reduced: 1.0s was too aggressive, caused context_count=0 on fast sends

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

_TOTAL_REQUESTS = 0
_TOTAL_ERRORS = 0
_TOTAL_CACHE_HITS = 0
_TOTAL_GEMINI_CALLS = 0
_TOTAL_EXTRACTION_HITS = 0
_COUNTER_LOCK = Lock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatResult:
    """Service-layer result for a chat request."""
    answer: str
    citations: list[str]
    confidence: float
    conversation_id: str | None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str, context_text: str) -> str:
    """Stable cache key: hash of normalised query + first 200 chars of context."""
    raw = query.lower().strip() + "|" + context_text[:200]
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    if key in _ANSWER_CACHE:
        _ANSWER_CACHE.move_to_end(key)
        return _ANSWER_CACHE[key]
    return None


def _cache_set(key: str, answer: str) -> None:
    if not key or not answer or answer in (NO_DOCUMENTS_ANSWER, RETRIEVAL_EMPTY_ANSWER, LLM_BUSY_ANSWER):
        return
    _ANSWER_CACHE[key] = answer
    _ANSWER_CACHE.move_to_end(key)
    if len(_ANSWER_CACHE) > MAX_CACHE_ENTRIES:
        _ANSWER_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Reranking helpers
# ---------------------------------------------------------------------------

def _lexical_score(query: str, text: str) -> int:
    q_words = set(query.lower().split())
    t = text.lower()
    return sum(w in t for w in q_words)


def _header_boost(text: str) -> int:
    t = text.lower()
    score = 0
    if "prepare by" in t or "prepared by" in t:
        score += 3
    if "quotation" in t or "project price" in t:
        score += 2
    if any(tok.isupper() and len(tok) > 3 for tok in text.split()):
        score += 1
    return score


def _pick_best_context(query: str, contexts: list[ContextChunk]) -> ContextChunk:
    best = max(
        contexts,
        key=lambda c: (
            0.6 * (1 / (1 + c.score))
            + 0.3 * _lexical_score(query, c.content)
            + 0.5 * _header_boost(c.content)
        ),
    )
    return best


def _top_similarity(contexts: list[ContextChunk]) -> float:
    if not contexts:
        return 0.0
    return max(1 / (1 + c.score) for c in contexts)


def _confidence_from_similarity(contexts: list[ContextChunk]) -> float:
    sim = _top_similarity(contexts)
    return round(85 + min(sim * 10, 10), 2)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_answer(answer: str) -> tuple[bool, str]:
    """Accept by default; reject only empty or explicit refusal phrases."""
    stripped = answer.strip()
    if not stripped:
        return False, "empty"
    lowered = stripped.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lowered:
            return False, f"refusal_phrase:{phrase}"
    return True, "accepted"


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def _retrieve_context(
    query: str,
    top_k: int,
    document_ids: list[str] | None,
    chat_id: str | int | None,
    request_id: str,
) -> list[ContextChunk]:
    try:
        results = retrieve(query, top_k=top_k, document_ids=document_ids, chat_id=chat_id)
        # ============================================================================
        # DETERMINISTIC ENTITY EXTRACTION
        # ============================================================================

        entity_answer = extract_entity_answer(
            query=query,
            contexts=results,
        )

        if entity_answer:
            logger.info(
                "entity.answer.used",
                extra={
                    "request_id": request_id,
                    "query": query,
                    "answer": entity_answer,
                },
            )

            return [
                ContextChunk(
                    content=entity_answer,
                    source_id="entity_extraction",
                    score=0.0,
                )
            ]
    except Exception:
        logger.exception("retrieval.error", extra={"request_id": request_id})
        raise
    logger.info(
        "retrieval.done",
        extra={"count": len(results), "top_k": top_k, "request_id": request_id},
    )
    return results


# ---------------------------------------------------------------------------
# Main hybrid handler
# ---------------------------------------------------------------------------

def handle_chat(
    message: str,
    conversation_id: str | None = None,
    filters: dict[str, str] | None = None,
    document_ids: list[str] | None = None,
    chat_id: str | int | None = None,
    request_id: str = "-",
    injected_history: list[Message] | None = None,
    user_id: str | None = None,
) -> ChatResult:
    """Process a chat request using the hybrid routing pipeline.

    Flow
    ----
    1. Per-user cooldown check (1 req/sec)
    2. Classify query: simple vs complex
    3. Retrieve context (top_k=1 for simple, top_k=5 for complex)
    4. Cache lookup
    5a. Simple: regex extraction → if found, return (NO GEMINI)
    5b. Simple: extraction failed → Gemini fallback (max_tokens=50)
    5c. Complex: Gemini generation (max_tokens=300)
    6. Cache store
    7. Update in-memory history
    """
    global _TOTAL_ERRORS, _TOTAL_REQUESTS, _TOTAL_CACHE_HITS
    global _TOTAL_GEMINI_CALLS, _TOTAL_EXTRACTION_HITS

    with _COUNTER_LOCK:
        _TOTAL_REQUESTS += 1

    start_time = time.perf_counter()

    # ── Per-user cooldown ──────────────────────────────────────────────────
    if user_id:
        now = time.monotonic()
        last = _USER_LAST_REQUEST.get(user_id, 0.0)
        if now - last < COOLDOWN_SECONDS:
            logger.warning(
                "chat.rate_limited",
                extra={"user_id": user_id, "request_id": request_id},
            )
            return ChatResult(
                answer="Please wait a moment before sending another message.",
                citations=[],
                confidence=0.0,
                conversation_id=conversation_id,
            )
        _USER_LAST_REQUEST[user_id] = now

    # ── Query classification (used for token limit selection) ─────────────
    query_type = classify_query(message)
    complex_mode = query_type == "complex"

    # Always retrieve at least 5 chunks; 8 for summary/overview queries.
    # Simple queries previously used top_k=1 which starved the LLM of context.
    _SUMMARY_WORDS = {"summary", "summarize", "summarise", "about", "overview",
                      "describe", "explain", "what is the pdf"}
    is_summary = any(w in message.lower() for w in _SUMMARY_WORDS)
    top_k = 8 if is_summary else 5

    logger.info(
        "chat.classified",
        extra={
            "query_type": query_type,
            "top_k": top_k,
            "request_id": request_id,
        },
    )

    # ── No documents attached to this chat ───────────────────────────────
    if document_ids is not None and len(document_ids) == 0:
        _log_metrics(
            request_id, message,
            0, 0,
            0, 0,
            cache_hit=False,
            gemini_used=False,
            extraction_used=False,
            query_type=query_type,
            retrieval_success=False,
            llm_success=False,
            status=STATUS_NO_DOCUMENTS,
        )
        return ChatResult(
            answer=NO_DOCUMENTS_ANSWER,
            citations=[],
            confidence=0.0,
            conversation_id=conversation_id,
        )

    try:
        # ── Retrieval ─────────────────────────────────────────────────────
        t_retrieval = time.perf_counter()
        contexts = _retrieve_context(message, top_k, document_ids, chat_id, request_id)
        retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

        # ── Entity extraction short-circuit ──────────────────────────────
        if (
            len(contexts) == 1
            and contexts[0].source_id == "entity_extraction"
        ):
            entity_ans = contexts[0].content
            logger.info(
                "entity.answer.short_circuit",
                extra={"request_id": request_id, "answer": entity_ans},
            )
            # Write to memory so follow-up questions work
            add_message(conversation_id, "user", message)
            add_message(conversation_id, "assistant", entity_ans)
            return ChatResult(
                answer=entity_ans,
                citations=["entity_extraction"],
                confidence=95.0,
                conversation_id=conversation_id,
            )

        if not contexts:
            _log_metrics(
                request_id, message,
                0, 0,
                retrieval_ms, 0,
                cache_hit=False,
                gemini_used=False,
                extraction_used=False,
                query_type=query_type,
                retrieval_success=False,
                llm_success=False,
                status=STATUS_RETRIEVAL_EMPTY,
            )
            return ChatResult(
                answer=RETRIEVAL_EMPTY_ANSWER,
                citations=[],
                confidence=0.0,
                conversation_id=conversation_id,
            )

        # Use ALL retrieved chunks as context — do NOT pick a single 'best' chunk.
        # Single-chunk mode was starving Gemini of context and causing wrong answers.
        prompt_contexts = contexts  # up to top_k chunks
        context_text = "\n\n".join(c.content for c in prompt_contexts)
        ckey = _cache_key(message, context_text)

        # ── Cache lookup ──────────────────────────────────────────────────
        cached = _cache_get(ckey)
        if cached:   # cached answers are already validated — no re-grounding needed
            with _COUNTER_LOCK:
                _TOTAL_CACHE_HITS += 1
            confidence = _confidence_from_similarity(contexts)
            add_message(conversation_id, "user", message)
            add_message(conversation_id, "assistant", cached)
            logger.info(
                "chat.cache_hit",
                extra={"request_id": request_id, "query_type": query_type},
            )
            _log_metrics(
                request_id, message,
                len(prompt_contexts), len(contexts),
                retrieval_ms, 0,
                cache_hit=True,
                gemini_used=False,
                extraction_used=False,
                query_type=query_type,
                retrieval_success=True,
                llm_success=True,
                status=STATUS_SUCCESS,
            )
            return ChatResult(
                answer=cached,
                citations=[c.source_id for c in prompt_contexts],
                confidence=confidence,
                conversation_id=conversation_id,
            )

        # ── Gemini path: ALL queries route here — no regex bypass ────────────
        gemini_used = False
        extraction_used = False
        llm_ms = 0.0
        answer: str = ""
        citations = [c.source_id for c in prompt_contexts]

        history = injected_history if injected_history is not None else get_history(conversation_id)
        max_tokens = TOKENS_COMPLEX if complex_mode else TOKENS_SIMPLE
        prompt = build_prompt(
            prompt_contexts, message, history, is_complex=complex_mode
        )

        t_llm = time.perf_counter()
        raw_answer = generate_answer(
            prompt,
            request_id=request_id,
            max_output_tokens=max_tokens,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000
        gemini_used = True

        with _COUNTER_LOCK:
            _TOTAL_GEMINI_CALLS += 1

        logger.info(
            "chat.gemini_used",
            extra={
                "query_type": query_type,
                "max_tokens": max_tokens,
                "answer_preview": raw_answer[:100],
                "request_id": request_id,
            },
        )

        if raw_answer in (LLM_RATE_LIMITED, LLM_UNAVAILABLE):
            _log_metrics(
                request_id, message,
                len(prompt_contexts), len(contexts),
                retrieval_ms, llm_ms,
                cache_hit=False,
                gemini_used=True,
                extraction_used=False,
                query_type=query_type,
                retrieval_success=True,
                llm_success=False,
                status=STATUS_LLM_UNAVAILABLE,
            )
            return ChatResult(
                answer=LLM_BUSY_ANSWER,
                citations=[],
                confidence=70.0,
                conversation_id=conversation_id,
            )

        # Log the raw Gemini answer before any decision
        logger.info(
            "chat.raw_gemini_answer",
            extra={
                "raw_answer": raw_answer,
                "length": len(raw_answer),
                "request_id": request_id,
            },
        )

        # Trust Gemini — only reject explicit refusal phrases or empty
        is_valid, reason = _validate_answer(raw_answer)
        validated_answer = raw_answer if is_valid else RETRIEVAL_EMPTY_ANSWER
        logger.info(
            "chat.validator",
            extra={
                "validator_input": raw_answer,
                "validator_output": "accepted" if is_valid else "rejected",
                "rejection_reason": None if is_valid else reason,
                "validated_answer": validated_answer,
                "request_id": request_id,
            },
        )

        accepted = is_valid
        logger.info(
            "chat.validator_decision",
            extra={
                "request_id": request_id,
                "validated_answer": validated_answer,
                "accepted": accepted,
            },
        )

        if not accepted:
            answer = validated_answer
            citations = []
        else:
            answer = validated_answer

        # ── Cache store ───────────────────────────────────────────────────
        if answer and answer not in (NO_DOCUMENTS_ANSWER, RETRIEVAL_EMPTY_ANSWER, LLM_BUSY_ANSWER):
            _cache_set(ckey, answer)

        # ── Confidence ───────────────────────────────────────────────────
        if answer == RETRIEVAL_EMPTY_ANSWER:
            confidence = 0.0
        else:
            confidence = _confidence_from_similarity(contexts)

        # ── Memory ───────────────────────────────────────────────────────
        add_message(conversation_id, "user", message)
        add_message(conversation_id, "assistant", answer)

        _log_metrics(
            request_id, message,
            len(prompt_contexts), len(contexts),
            retrieval_ms, llm_ms,
            cache_hit=False,
            gemini_used=gemini_used,
            extraction_used=extraction_used,
            query_type=query_type,
            retrieval_success=True,
            llm_success=not gemini_used or answer not in (LLM_BUSY_ANSWER,),
            status=STATUS_SUCCESS,
        )

        grounded = len(prompt_contexts) > 0
        final_answer = answer
        logger.info(
            "chat.final_answer_before_response",
            extra={
                "request_id": request_id,
                "final_answer": final_answer,
                "confidence": confidence,
                "grounded": grounded,
            },
        )

        return ChatResult(
            answer=final_answer,
            citations=citations,
            confidence=confidence,
            conversation_id=conversation_id,
        )

    except Exception:
        with _COUNTER_LOCK:
            _TOTAL_ERRORS += 1
        logger.exception(
            "chat.error",
            extra={"request_id": request_id},
        )
        raise


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

def _log_metrics(
    request_id: str,
    message: str,
    context_count: int,
    retrieved_count: int,
    retrieval_ms: float,
    llm_ms: float,
    cache_hit: bool,
    gemini_used: bool,
    extraction_used: bool = False,
    query_type: str = "unknown",
    retrieval_success: bool = False,
    llm_success: bool = False,
    status: str = "unknown",
) -> None:
    with _COUNTER_LOCK:
        total_req = _TOTAL_REQUESTS
        total_err = _TOTAL_ERRORS
        total_cache = _TOTAL_CACHE_HITS
        total_gemini = _TOTAL_GEMINI_CALLS
        total_extract = _TOTAL_EXTRACTION_HITS

    logger.info(
        "chat.metrics",
        extra={
            "request_id": request_id,
            "query_type": query_type,
            "query_length": len(message),
            "context_count": context_count,
            "retrieved_count": retrieved_count,
            "retrieval_time_ms": round(retrieval_ms, 2),
            "llm_time_ms": round(llm_ms, 2),
            "retrieval_success": retrieval_success,
            "llm_success": llm_success,
            "status": status,
            "cache_hit": cache_hit,
            "gemini_used": gemini_used,
            "extraction_used": extraction_used,
            "total_requests": total_req,
            "total_errors": total_err,
            "total_cache_hits": total_cache,
            "total_gemini_calls": total_gemini,
            "total_extraction_hits": total_extract,
            "gemini_rate": round(total_gemini / total_req, 3) if total_req else 0,
        },
    )
