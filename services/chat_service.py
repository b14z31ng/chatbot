from collections import OrderedDict
from dataclasses import dataclass
import logging
import re
from threading import Lock
import time

from services.llm.generator import generate_answer, LLM_UNAVAILABLE
from services.llm.memory import Message, add_message, get_history
from services.llm.prompt_builder import build_prompt
from services.rag.retriever import ContextChunk, retrieve

UNKNOWN_ANSWER = "Not in knowledge base."
LLM_DOWN_ANSWER = "AI service temporarily unavailable."
INVALID_PHRASES = (
    "i don't know",
    "i do not know",
    "not sure",
    "cannot find",
    "no information",
)

logger = logging.getLogger(__name__)

MAX_CACHE_ENTRIES = 20
_ANSWER_CACHE: "OrderedDict[str, str]" = OrderedDict()
_TOTAL_REQUESTS = 0
_TOTAL_ERRORS = 0
_TOTAL_CACHE_HITS = 0
_COUNTER_LOCK = Lock()


# ---------------------------------------------------------------------------
# Reranking helpers
# ---------------------------------------------------------------------------

def lexical_score(query: str, text: str) -> int:
    """Count how many query words appear in the text."""
    q_words = set(query.lower().split())
    t = text.lower()
    return sum(w in t for w in q_words)


def header_boost(text: str) -> int:
    """Boost chunks that look like document headers."""
    t = text.lower()
    score = 0
    if "prepare by" in t or "prepared by" in t:
        score += 3
    if "quotation" in t or "project price" in t:
        score += 2
    if any(tok.isupper() and len(tok) > 3 for tok in text.split()):
        score += 1
    return score


def pick_best_context(query: str, contexts: list[ContextChunk]) -> ContextChunk:
    """Hybrid reranker: vector similarity + lexical + header boost."""
    best: ContextChunk | None = None
    best_score = -1e9

    for c in contexts:
        # Convert L2 distance → similarity (lower distance = higher similarity)
        v = 1 / (1 + c.score)
        l = lexical_score(query, c.content)
        h = header_boost(c.content)

        combined = (0.6 * v) + (0.3 * l) + (0.5 * h)

        if combined > best_score:
            best_score = combined
            best = c

    return best if best else contexts[0]


def top_similarity(contexts: list[ContextChunk]) -> float:
    """Return the highest vector similarity from the retrieved chunks."""
    if not contexts:
        return 0.0
    return max(1 / (1 + c.score) for c in contexts)


# ---------------------------------------------------------------------------
# Service dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatResult:
    """Service-layer result for a chat request."""

    answer: str
    citations: list[str]
    confidence: float
    conversation_id: str | None


# ---------------------------------------------------------------------------
# Context retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str, filters: dict[str, str] | None = None, request_id: str = "-"
) -> list[ContextChunk]:
    """Retrieve relevant context for a query."""

    logger.info(
        "retrieval.start",
        extra={"query_length": len(query), "request_id": request_id},
    )
    try:
        results = retrieve(query, top_k=10)
    except Exception:
        logger.exception(
            "retrieval.error",
            extra={"query_length": len(query), "request_id": request_id},
        )
        raise
    if not results:
        logger.warning(
            "retrieval.empty", extra={"query": query, "request_id": request_id}
        )
    logger.info(
        "retrieval.end",
        extra={"retrieved_count": len(results), "request_id": request_id},
    )
    return results


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_get(cache_key: str) -> str | None:
    """Return cached answer for key if present."""
    if cache_key in _ANSWER_CACHE:
        _ANSWER_CACHE.move_to_end(cache_key)
        return _ANSWER_CACHE[cache_key]
    return None


def _cache_set(cache_key: str, answer: str) -> None:
    """Store answer in cache with LRU eviction."""
    if not cache_key or not answer:
        return
    _ANSWER_CACHE[cache_key] = answer
    _ANSWER_CACHE.move_to_end(cache_key)
    if len(_ANSWER_CACHE) > MAX_CACHE_ENTRIES:
        _ANSWER_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_invalid_answer(answer: str) -> bool:
    """Return True when answer is empty or non-committal."""
    stripped = answer.strip().lower()
    if not stripped:
        return True
    return any(phrase in stripped for phrase in INVALID_PHRASES)


def _is_answer_grounded(answer: str, contexts: list[ContextChunk]) -> bool:
    """Basic grounding check: at least one answer token found in context."""
    if not answer or not contexts:
        return False
    context_text = " ".join(context.content for context in contexts).lower()
    tokens = re.findall(r"[a-z0-9]{3,}", answer.lower())
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in context_text)
    logger.debug(
        "grounding.check",
        extra={
            "answer_preview": answer[:100],
            "token_count": len(tokens),
            "match_count": matches,
            "grounded": matches >= 1,
        },
    )
    return matches >= 1


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def build_grounded_answer(
    contexts: list[ContextChunk],
    user_query: str,
    chat_history: list[Message],
    request_id: str,
) -> tuple[str, list[str], float]:
    """Generate a grounded answer from context and history."""

    prompt = build_prompt(contexts, user_query, chat_history)
    answer = generate_answer(prompt, request_id=request_id)

    citations = [context.source_id for context in contexts]

    # Propagate LLM outage sentinel upstream
    if answer == LLM_UNAVAILABLE:
        return LLM_UNAVAILABLE, citations, 0.0

    answer = answer.strip().split("\n")[0]

    # Grounding validation
    context_text = contexts[0].content
    if not answer or not any(
        word.lower() in context_text.lower() for word in answer.split()
    ):
        return UNKNOWN_ANSWER, citations, 0.0

    # Confidence derived from top vector similarity (0-95 scale)
    sim = top_similarity(contexts)
    confidence = round(min(sim * 100, 95.0), 2)

    return answer, citations, confidence


# ---------------------------------------------------------------------------
# Main chat handler
# ---------------------------------------------------------------------------

def handle_chat(
    message: str,
    conversation_id: str | None = None,
    filters: dict[str, str] | None = None,
    request_id: str = "-",
) -> ChatResult:
    """Process a chat request using retrieval-first policy."""

    global _TOTAL_ERRORS
    global _TOTAL_REQUESTS
    global _TOTAL_CACHE_HITS

    with _COUNTER_LOCK:
        _TOTAL_REQUESTS += 1
    start_time = time.perf_counter()
    retrieval_start = time.perf_counter()
    retrieval_time_ms = 0.0
    llm_time_ms = 0.0
    try:
        contexts = retrieve_context(message, filters=filters, request_id=request_id)
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        if not contexts:
            duration_ms = (time.perf_counter() - start_time) * 1000
            with _COUNTER_LOCK:
                total_requests = _TOTAL_REQUESTS
                total_errors = _TOTAL_ERRORS
                total_cache_hits = _TOTAL_CACHE_HITS
            cache_hit_rate = (
                total_cache_hits / total_requests if total_requests else 0.0
            )
            logger.info(
                "chat.metrics",
                extra={
                    "query_length": len(message),
                    "context_count": 0,
                    "retrieved_count": 0,
                    "retrieval_time_ms": round(retrieval_time_ms, 2),
                    "llm_time_ms": 0,
                    "total_time_ms": round(duration_ms, 2),
                    "cache_hit": False,
                    "cache_hit_rate": round(cache_hit_rate, 4),
                    "total_requests": total_requests,
                    "total_errors": total_errors,
                    "request_id": request_id,
                },
            )
            return ChatResult(
                answer=UNKNOWN_ANSWER,
                citations=[],
                confidence=0.0,
                conversation_id=conversation_id,
            )

        # Rerank: pick the best context chunk
        best_chunk = pick_best_context(message, contexts)
        sim = top_similarity(contexts)
        logger.debug(
            "rerank.result",
            extra={
                "top_similarity": round(sim, 4),
                "candidate_count": len(contexts),
                "request_id": request_id,
            },
        )
        contexts = [best_chunk]

        history = get_history(conversation_id)
        context_ids = ",".join([context.source_id for context in contexts])
        cache_key = f"{message}:{context_ids}"
        cached_answer = _cache_get(cache_key)
        cache_hit = False

        if cached_answer and _is_answer_grounded(cached_answer, contexts):
            answer = cached_answer
            citations = [context.source_id for context in contexts]
            confidence = round(min(sim * 100, 95.0), 2)
            cache_hit = True
            with _COUNTER_LOCK:
                _TOTAL_CACHE_HITS += 1
        else:
            llm_start = time.perf_counter()
            answer, citations, confidence = build_grounded_answer(
                contexts,
                message,
                history,
                request_id,
            )
            llm_time_ms = (time.perf_counter() - llm_start) * 1000

            logger.info(
                "chat.llm_answer",
                extra={
                    "answer_preview": answer[:200] if answer else "<empty>",
                    "confidence": confidence,
                    "request_id": request_id,
                },
            )

            # LLM outage — return service-unavailable response
            if answer == LLM_UNAVAILABLE:
                logger.warning(
                    "chat.llm_unavailable",
                    extra={"request_id": request_id},
                )
                return ChatResult(
                    answer=LLM_DOWN_ANSWER,
                    citations=[],
                    confidence=0.0,
                    conversation_id=conversation_id,
                )

            if _is_invalid_answer(answer) or not _is_answer_grounded(
                answer, contexts
            ):
                logger.warning(
                    "chat.grounding_failed",
                    extra={
                        "invalid": _is_invalid_answer(answer),
                        "grounded": _is_answer_grounded(answer, contexts),
                        "request_id": request_id,
                    },
                )
                answer = UNKNOWN_ANSWER
                citations = []
                confidence = 0.0
            else:
                _cache_set(cache_key, answer)

        add_message(conversation_id, "user", message)
        add_message(conversation_id, "assistant", answer)

        duration_ms = (time.perf_counter() - start_time) * 1000
        with _COUNTER_LOCK:
            total_requests = _TOTAL_REQUESTS
            total_errors = _TOTAL_ERRORS
            total_cache_hits = _TOTAL_CACHE_HITS
        cache_hit_rate = total_cache_hits / total_requests if total_requests else 0.0
        logger.info(
            "chat.metrics",
            extra={
                "query_length": len(message),
                "context_count": len(contexts),
                "retrieved_count": len(contexts),
                "retrieval_time_ms": round(retrieval_time_ms, 2),
                "llm_time_ms": round(llm_time_ms, 2),
                "total_time_ms": round(duration_ms, 2),
                "cache_hit": cache_hit,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "request_id": request_id,
            },
        )
        logger.info(
            "chat.complete",
            extra={
                "duration_ms": round(duration_ms, 2),
                "cached": cache_hit,
                "request_id": request_id,
            },
        )
        return ChatResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            conversation_id=conversation_id,
        )
    except Exception:
        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
        duration_ms = (time.perf_counter() - start_time) * 1000
        with _COUNTER_LOCK:
            _TOTAL_ERRORS += 1
        logger.exception(
            "chat.error",
            extra={
                "request_id": request_id,
                "retrieval_time_ms": round(retrieval_time_ms, 2),
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise
