from typing import Any
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """In-memory FAISS vector store with text and metadata."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

    def add_documents(self, chunks: list[dict[str, str]], embeddings: np.ndarray) -> None:
        """Add documents and embeddings to the store."""

        if not chunks:
            return
        vectors = np.array(embeddings).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.embeddings.extend(vectors.tolist())
        for chunk in chunks:
            self.texts.append(chunk["text"])
            self.metadata.append({"source_id": chunk["source_id"]})

        logger.debug(
            "vector_store.added",
            extra={"stored_texts": len(self.texts), "stored_metadata": len(self.metadata)},
        )

    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for nearest neighbors by embedding similarity."""

        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.asarray(query_embedding).astype("float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        logger.debug(
            "vector_store.search",
            extra={
                "index_size": self.index.ntotal,
                "query_shape": list(query_vector.shape),
                "k": k,
            },
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.texts):
                logger.debug("vector_store.invalid_index", extra={"idx": int(idx)})
                continue

            results.append({
                "text": self.texts[idx],
                "source_id": self.metadata[idx]["source_id"],
                "score": float(distances[0][i]),
            })
        return results


_STORE = InMemoryVectorStore()
# TODO: Persist FAISS index to disk for production use


def add_documents(chunks: list[dict[str, str]], embeddings: np.ndarray) -> None:
    """Add documents to the global vector store."""
    _STORE.add_documents(chunks, embeddings)


def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the global vector store."""
    return _STORE.similarity_search(query_embedding, top_k=top_k)
