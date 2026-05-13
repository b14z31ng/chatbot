"""LangChain FAISS retriever with MMR, query expansion, and per-chunk debug logging.

Public API is IDENTICAL to the old retriever:
    retrieve(query, top_k, document_ids, chat_id) -> list[ContextChunk]

Improvements over the previous version
---------------------------------------
- MMR retrieval (k=8, fetch_k=20, lambda_mult=0.5) for diversity
- Query expansion for entity/label queries (company, client, vendor, etc.)
- Increased k: factual=8, summary/overview=12
- Chunk-level debug logs: preview, score, source_id
- Handles BOTH store formats:
    index.faiss + index.pkl    → LangChain native format
    index.faiss + metadata.pkl → legacy InMemoryVectorStore format (fallback)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

STORE_DIR = Path("data/vector_store")
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# k tuning
# ---------------------------------------------------------------------------

_SUMMARY_WORDS = {
    "summary", "summarize", "summarise", "about", "overview",
    "describe", "explain", "what is the pdf", "tell me about",
    "give me a", "provide a",
}
_K_FACTUAL = 8
_K_SUMMARY = 12
_FETCH_K = 20        # MMR candidate pool
_LAMBDA_MULT = 0.5   # MMR diversity (0 = max diversity, 1 = max relevance)

# ---------------------------------------------------------------------------
# Query expansion map — entity/label queries → richer semantic queries
# ---------------------------------------------------------------------------

_EXPANSION_MAP: list[tuple[set[str], list[str]]] = [
    (
        {"company", "company name", "organization", "organisation",
         "vendor", "supplier", "quotation owner", "who prepared", "prepared by"},
        ["prepared by company organization quotation header",
         "prepared by submitted by from",
         "company name organization"],
    ),
    (
        {"client", "client name", "customer", "client company",
         "for whom", "prepared for"},
        ["client customer prepared for",
         "to whom quotation submitted client name"],
    ),
    (
        {"project", "project name", "project type", "project title"},
        ["project name type title description",
         "project quotation custom"],
    ),
    (
        {"price", "cost", "amount", "total", "quotation amount",
         "how much", "fee", "rate"},
        ["total project cost price quotation amount fee",
         "BDT USD price total amount"],
    ),
    (
        {"date", "validity", "issued", "when"},
        ["date validity issued quotation date",
         "valid until date"],
    ),
]


def _expand_query(query: str) -> list[str]:
    """
    Universal semantic query expansion.

    Converts casual/natural user questions into
    retrieval-friendly variants without hardcoding
    document-specific knowledge.
    """

    q = query.lower().strip()

    expansions = [query]

    # temporal questions
    if any(x in q for x in ["when", "date", "year", "time"]):
        expansions.extend([
            q.replace("when", "date"),
            q.replace("when", "year"),
            f"{q} historical event timeline",
        ])

    # explanation / impact
    if any(x in q for x in ["impact", "effect", "result", "importance"]):
        expansions.extend([
            f"{q} consequences",
            f"{q} significance",
            f"{q} outcome",
        ])

    # people / founders
    if any(x in q for x in ["who", "founder", "leader", "started"]):
        expansions.extend([
            f"{q} founded by",
            f"{q} leader",
            f"{q} person",
        ])

    # summaries
    if any(x in q for x in ["about", "summary", "explain", "describe"]):
        expansions.extend([
            f"{q} overview",
            f"{q} details",
            f"{q} background",
        ])

    # entity lookup
    if any(x in q for x in ["company", "client", "organization", "name"]):
        expansions.extend([
            f"{q} prepared by",
            f"{q} organization",
            f"{q} header",
        ])

    # deduplicate while preserving order
    seen = set()
    final_queries = []

    for item in expansions:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            final_queries.append(cleaned)

    return final_queries


# ---------------------------------------------------------------------------
# Embedding singleton
# ---------------------------------------------------------------------------

_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


# ---------------------------------------------------------------------------
# ContextChunk dataclass — identical shape to callers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextChunk:
    """Retrieved context chunk."""
    content: str
    source_id: str
    score: float


# ---------------------------------------------------------------------------
# Per-chat FAISS store cache
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[str, FAISS] = {}


def _load_store(chat_id: str | int) -> FAISS | None:
    """Load (or return cached) the LangChain FAISS store for a chat.

    Tries LangChain native format (index.pkl) first, falls back gracefully.
    """
    key = str(chat_id)
    if key in _STORE_CACHE:
        return _STORE_CACHE[key]

    store_path = STORE_DIR / f"chat_{key}"
    index_file = store_path / "index.faiss"
    lc_pkl = store_path / "index.pkl"       # LangChain native

    if not index_file.exists() or not lc_pkl.exists():
        return None  # legacy store — will be handled by _retrieve_legacy

    try:
        store = FAISS.load_local(
            str(store_path),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
            index_name="index",
        )
        _STORE_CACHE[key] = store
        logger.info(
            "retriever.store_loaded",
            extra={"chat_id": key, "path": str(store_path), "format": "langchain"},
        )
        return store
    except Exception:
        logger.exception("retriever.store_load_failed", extra={"chat_id": key})
        return None


def invalidate_cache(chat_id: str | int) -> None:
    """Call this after a new upload to force the store to reload from disk."""
    _STORE_CACHE.pop(str(chat_id), None)


# ---------------------------------------------------------------------------
# Public retrieve()
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = _K_FACTUAL,
    document_ids: list[str] | None = None,
    chat_id: str | int | None = None,
) -> list[ContextChunk]:
    """Retrieve context chunks with MMR + query expansion.

    Args:
        query:        User query string.
        top_k:        Max chunks to return (auto-adjusted for summary queries).
        document_ids: UUIDs this chat owns (isolation). [] → return empty.
        chat_id:      Identifies which per-chat FAISS store to use.
    """
    if document_ids is not None and len(document_ids) == 0:
        logger.info("retriever.no_documents", extra={"chat_id": chat_id})
        return []

    # Adjust k based on query type
    q_lower = query.lower()
    if any(w in q_lower for w in _SUMMARY_WORDS):
        top_k = max(top_k, _K_SUMMARY)
    else:
        top_k = max(top_k, _K_FACTUAL)

    t0 = time.perf_counter()

    # ── Try LangChain store (index.pkl format) first ─────────────────────
    if chat_id is not None:
        lc_store = _load_store(chat_id)
        if lc_store is not None:
            return _retrieve_mmr(lc_store, query, top_k, document_ids, chat_id, t0)

    # ── Fallback: legacy InMemoryVectorStore (metadata.pkl format) ───────
    return _retrieve_legacy(query, top_k, document_ids, chat_id, t0)


# ---------------------------------------------------------------------------
# Header/metadata priority helper
# ---------------------------------------------------------------------------

# Labels that signal a chunk is about document ownership/metadata
_OWNER_SIGNALS = {
    "prepared by", "submitted by", "quotation by", "organization",
    "organisation", "company name", "vendor", "about us", "quotation header",
}
# Labels that signal a chunk is about a service/tool, NOT the document owner
_SERVICE_SIGNALS = {
    "payment gateway", "ssl", "integration", "api", "platform",
    "powered by", "technology stack", "framework",
}


def _header_priority(text: str) -> int:
    """Score a chunk by how likely it is to contain document ownership metadata.

    Higher score = should rank earlier.
    Owner-signal keywords give +3; service-signal keywords give -2.
    """
    t = text.lower()
    score = 0
    for sig in _OWNER_SIGNALS:
        if sig in t:
            score += 3
    for sig in _SERVICE_SIGNALS:
        if sig in t:
            score -= 2
    return score


def _apply_header_priority(chunks: list[ContextChunk]) -> list[ContextChunk]:
    """Stable-sort chunks so owner/header chunks surface before service chunks."""
    return sorted(chunks, key=lambda c: _header_priority(c.content), reverse=True)


# ---------------------------------------------------------------------------
# MMR retrieval from LangChain FAISS
# ---------------------------------------------------------------------------

def _retrieve_mmr(
    store: FAISS,
    query: str,
    top_k: int,
    document_ids: list[str] | None,
    chat_id: str | int | None,
    t0: float,
) -> list[ContextChunk]:
    """MMR retrieval with query expansion and UUID-level document filtering."""

    allowed_set = {str(d) for d in document_ids} if document_ids is not None else None
    queries = _expand_query(query)

    logger.debug(
        "retriever.query_expanded",
        extra={
            "original": query,
            "expanded_count": len(queries),
            "chat_id": str(chat_id),
        },
    )

    seen_content: set[str] = set()
    chunks: list[ContextChunk] = []

    for q in queries:
        if len(chunks) >= top_k:
            break
        try:
            # MMR: fetch_k candidates → re-rank for diversity → return k
            raw_docs = store.max_marginal_relevance_search(
                q,
                k=min(top_k * 2, _FETCH_K),
                fetch_k=_FETCH_K,
                lambda_mult=_LAMBDA_MULT,
            )
        except Exception:
            logger.exception("retriever.mmr_failed", extra={"chat_id": chat_id, "query": q[:60]})
            # Graceful degradation: fall back to similarity for this query
            try:
                raw_docs_scored = store.similarity_search_with_score(q, k=_FETCH_K)
                raw_docs = [doc for doc, _ in raw_docs_scored]
            except Exception:
                continue

        for doc in raw_docs:
            if len(chunks) >= top_k:
                break

            meta = doc.metadata or {}
            source_id = meta.get("source_id", "")
            doc_uuid = source_id.split(":")[0] if ":" in source_id else meta.get("document_id", source_id)
            doc_uuid = str(doc_uuid)

            # UUID isolation filter
            if allowed_set is not None and doc_uuid not in allowed_set:
                continue

            # Deduplicate across multi-query expansion
            content_key = doc.page_content[:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            chunks.append(ContextChunk(
                content=doc.page_content,
                source_id=source_id or doc_uuid,
                score=0.0,   # assigned properly after sort below
            ))

    # ── Header/metadata priority re-sort ─────────────────────────────────
    chunks = _apply_header_priority(chunks)

    # Assign rank-based approximate scores after sort (lower = better, like L2)
    chunks = [
        ContextChunk(content=c.content, source_id=c.source_id,
                     score=round(0.05 * rank, 4))
        for rank, c in enumerate(chunks)
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Per-chunk debug log ───────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        logger.debug(
            "retriever.chunk_debug",
            extra={
                "rank": i + 1,
                "source_id": chunk.source_id,
                "preview": chunk.content[:120].replace("\n", " "),
                "chat_id": str(chat_id),
            },
        )

    logger.info(
        "retrieval.done",
        extra={
            "chat_id": str(chat_id),
            "retrieved_count": len(chunks),
            "top_k": top_k,
            "queries_used": len(queries),
            "elapsed_ms": round(elapsed_ms, 1),
            "backend": "langchain_mmr",
        },
    )
    return chunks


# ---------------------------------------------------------------------------
# Legacy fallback — similarity search via InMemoryVectorStore
# ---------------------------------------------------------------------------

def _retrieve_legacy(
    query: str,
    top_k: int,
    document_ids: list[str] | None,
    chat_id: str | int | None,
    t0: float,
) -> list[ContextChunk]:
    """Fallback for stores saved in legacy metadata.pkl format."""
    try:
        from services.rag.embeddings import get_embedding
        from services.rag.vector_store import similarity_search
        import numpy as np

        # Apply query expansion — try expanded queries and merge results
        queries = _expand_query(query)
        all_results: list[dict] = []
        seen: set[str] = set()

        for q in queries:
            q_emb = np.array(get_embedding(q)).astype("float32").reshape(1, -1)
            results, _ = similarity_search(
                q_emb,
                top_k=top_k,
                allowed_doc_ids=document_ids,
                chat_id=chat_id,
                return_metrics=True,
            )
            for r in results:
                key = r.get("source_id", r["text"][:80])
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # Sort by score (L2 distance — lower is better) and take top_k
        all_results.sort(key=lambda r: r.get("score", 9999))
        all_results = all_results[:top_k]

    except Exception:
        logger.exception("retriever.legacy_failed", extra={"chat_id": chat_id})
        return []

    elapsed_ms = (time.perf_counter() - t0) * 1000
    chunks = [
        ContextChunk(
            content=r["text"],
            source_id=r.get("source_id", ""),
            score=float(r.get("score", 0.0)),
        )
        for r in all_results
    ]

    for i, chunk in enumerate(chunks):
        logger.debug(
            "retriever.chunk_debug",
            extra={
                "rank": i + 1,
                "source_id": chunk.source_id,
                "score": round(chunk.score, 4),
                "preview": chunk.content[:120].replace("\n", " "),
                "chat_id": str(chat_id),
            },
        )

    logger.info(
        "retrieval.done",
        extra={
            "chat_id": str(chat_id),
            "retrieved_count": len(chunks),
            "top_k": top_k,
            "elapsed_ms": round(elapsed_ms, 1),
            "backend": "legacy_inmemory",
        },
    )
    return chunks
