"""FAISS vector store with persistence to disk."""
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

STORE_DIR = Path("data/vector_store")


class InMemoryVectorStore:
    """FAISS vector store with text, metadata, and disk persistence."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.index_file = store_dir / "index.faiss"
        self.meta_file = store_dir / "metadata.pkl"
        self.index: faiss.Index | None = None
        self.texts: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

    def add_documents(self, chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
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
            text = chunk.get("text", "")
            source_id = chunk.get("source_id")
            meta = dict(chunk.get("metadata") or {})
            if source_id:
                meta.setdefault("source_id", source_id)
                meta.setdefault("document_id", source_id.split(":")[0])
            self.texts.append(text)
            self.metadata.append(meta)

        logger.debug(
            "vector_store.added",
            extra={"stored_texts": len(self.texts), "stored_metadata": len(self.metadata)},
        )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        allowed_doc_ids: list[str] | None = None,
        chat_id: str | int | None = None,
        return_metrics: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search for nearest neighbors, filtered by document ID + chat ID.

        Args:
            allowed_doc_ids:
                - None  → unscoped search (legacy /chat endpoint)
                - []    → chat has no documents; return empty immediately
                - [ids] → scoped: only return chunks whose source_id starts
                          with one of the given document UUIDs
        """
        index_size = self.index.ntotal if self.index is not None else 0

        # Chat with no documents attached — return nothing immediately
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            metrics = {"raw_count": 0, "filtered_count": 0, "index_size": index_size}
            return ([], metrics) if return_metrics else []

        if self.index is None or self.index.ntotal == 0:
            metrics = {"raw_count": 0, "filtered_count": 0, "index_size": index_size}
            return ([], metrics) if return_metrics else []

        query_vector = np.asarray(query_embedding).astype("float32")
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Fetch more candidates when filtering so we still return top_k after filter
        scoped = allowed_doc_ids is not None
        fetch_k = min(top_k * 5 if scoped else top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, fetch_k)

        logger.debug(
            "vector_store.search",
            extra={
                "index_size": self.index.ntotal,
                "fetch_k": fetch_k,
                "scoped": scoped,
                "allowed_doc_count": len(allowed_doc_ids) if scoped else -1,
            },
        )

        allowed_set = {str(doc_id) for doc_id in allowed_doc_ids} if scoped else None
        active_chat_id = str(chat_id) if chat_id is not None else None
        results: list[dict[str, Any]] = []
        retrieved_doc_ids: list[str] = []
        raw_count = 0

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue

            raw_count += 1
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            source_id = meta.get("source_id")
            doc_uuid = str(meta.get("document_id") or (source_id.split(":")[0] if source_id else ""))
            doc_chat_id = meta.get("chat_id")
            doc_chat_id = str(doc_chat_id) if doc_chat_id is not None else None

            # Strict chat-level filtering
            if active_chat_id is not None and doc_chat_id != active_chat_id:
                continue

            # Strict UUID-level filtering — only chunks from this chat's documents
            if allowed_set is not None and doc_uuid not in allowed_set:
                continue

            retrieved_doc_ids.append(doc_uuid)
            results.append({
                "text": self.texts[idx],
                "source_id": source_id,
                "score": float(distances[0][i]),
                "metadata": meta,
            })

            if len(results) >= top_k:
                break

        logger.debug(
            "vector_store.retrieval_scope",
            extra={
                "scoped": scoped,
                "chat_id": active_chat_id,
                "raw_count": raw_count,
                "results_count": len(results),
                "retrieved_doc_ids": list(set(retrieved_doc_ids)),
            },
        )

        metrics = {
            "raw_count": raw_count,
            "filtered_count": len(results),
            "index_size": self.index.ntotal,
        }
        return (results, metrics) if return_metrics else results

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None or self.index.ntotal == 0:
            return
        if (self.store_dir / "index.pkl").exists():
            logger.info(
                "vector_store.save_skipped_langchain",
                extra={"path": str(self.store_dir)},
            )
            return
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
        logger.info(
            "vector_store.saved",
            extra={"path": str(self.store_dir), "vectors": self.index.ntotal},
        )

    def load(self) -> None:
        """Reload FAISS index and metadata from disk if available."""
        if (self.store_dir / "index.pkl").exists():
            return
        if not self.index_file.exists() or not self.meta_file.exists():
            return
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            logger.info(
                "vector_store.loaded",
                extra={"vectors": self.index.ntotal, "texts": len(self.texts), "path": str(self.store_dir)},
            )
        except Exception:
            logger.exception("vector_store.load_failed")


_GLOBAL_STORE = InMemoryVectorStore(STORE_DIR)
# Auto-load persisted index on module import (startup)
_GLOBAL_STORE.load()
_CHAT_STORES: dict[str, InMemoryVectorStore] = {}
_STORE = _GLOBAL_STORE


def _get_store(chat_id: str | int | None) -> InMemoryVectorStore:
    if chat_id is None:
        return _GLOBAL_STORE
    key = str(chat_id)
    store = _CHAT_STORES.get(key)
    if store is None:
        store = InMemoryVectorStore(STORE_DIR / f"chat_{key}")
        store.load()
        _CHAT_STORES[key] = store
    return store


def add_documents(chunks: list[dict[str, Any]], embeddings: np.ndarray, chat_id: str | int | None = None) -> None:
    """Add documents to the vector store."""
    store = _get_store(chat_id)
    store.add_documents(chunks, embeddings)


def save_store(chat_id: str | int | None = None) -> None:
    """Persist the vector store to disk."""
    store = _get_store(chat_id)
    store.save()


def similarity_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
    allowed_doc_ids: list[str] | None = None,
    chat_id: str | int | None = None,
    return_metrics: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search the vector store, optionally filtered by document IDs + chat ID."""
    store = _get_store(chat_id)
    return store.similarity_search(
        query_embedding,
        top_k=top_k,
        allowed_doc_ids=allowed_doc_ids,
        chat_id=chat_id,
        return_metrics=return_metrics,
    )
