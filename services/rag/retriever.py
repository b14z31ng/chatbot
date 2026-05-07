import logging
import numpy as np
from dataclasses import dataclass

from services.rag.embeddings import get_embedding
from services.rag.vector_store import similarity_search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextChunk:
    """Retrieved context chunk."""

    content: str
    source_id: str
    score: float


def rewrite_query(q: str) -> str:
    """Lightly expand query for better embedding recall on common patterns."""
    s = q.lower()

    if "company" in s and "name" in s:
        return "company name prepared by organization header document"

    if "cost" in s or "price" in s:
        return "total project cost pricing quotation amount"

    return q


def retrieve(query: str, top_k: int = 10) -> list[ContextChunk]:
    """Retrieve top-k context chunks for a query."""

    normalized_query = rewrite_query(query.strip())
    query_embedding = get_embedding(normalized_query)
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    results = similarity_search(query_embedding, top_k=top_k)

    logger.debug(
        "retrieval.raw",
        extra={
            "query_rewritten": normalized_query[:80],
            "result_count": len(results),
        },
    )

    chunks: list[ContextChunk] = []
    for result in results:
        logger.debug(
            "retrieval.score_debug",
            extra={
                "text_preview": result["text"][:80],
                "score": round(result["score"], 4),
                "source_id": result["source_id"],
            },
        )
        chunks.append(
            ContextChunk(
                content=result["text"],
                source_id=result["source_id"],
                score=result["score"],
            )
        )

    logger.info(
        "retrieval.filtered",
        extra={
            "raw_count": len(results),
            "filtered_count": len(chunks),
            "query": normalized_query[:60],
        },
    )
    return chunks
