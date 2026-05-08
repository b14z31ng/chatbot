"""PDF ingestion using LangChain's RecursiveCharacterTextSplitter.

Keeps the same return format as before:
    [{"text": str, "source_id": str}, ...]

so all callers (chat_upload.py, etc.) need zero changes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
STORE_DIR = Path("data/vector_store")
EMBED_MODEL = "all-MiniLM-L6-v2"

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


def ingest_pdf(
    file_path: str,
    document_id: str,
    chat_id: str | int | None = None,
    filename: str | None = None,
) -> list[dict[str, Any]]:
    """Load and chunk a PDF, returning chunks with source IDs.

    Returns the same format as before so callers require no changes:
        [{"text": "...", "source_id": "<uuid>:<index>", "metadata": {...}}, ...]
    """
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception:
        logger.exception("ingestion.pdf_load_failed", extra={"file_path": file_path})
        return []

    if not pages:
        logger.warning("ingestion.empty", extra={"document_id": document_id})
        return []

    docs = _SPLITTER.split_documents(pages)

    if not docs:
        logger.warning("ingestion.no_chunks", extra={"document_id": document_id})
        return []

    logger.info(
        "ingestion.chunked",
        extra={
            "document_id": document_id,
            "page_count": len(pages),
            "chunk_count": len(docs),
            "chunk_size": CHUNK_SIZE,
        },
    )

    safe_filename = filename or Path(file_path).name
    vector_path = STORE_DIR / f"chat_{chat_id}" if chat_id is not None else None

    for i, doc in enumerate(docs):
        doc.metadata = {
            **(doc.metadata or {}),
            "chat_id": str(chat_id) if chat_id is not None else "None",
            "document_id": str(document_id),
            "filename": safe_filename,
            "source_id": f"{document_id}:{i}",
        }

    if vector_path is not None:
        try:
            store = FAISS.from_documents(docs, _get_embeddings())
            store.save_local(str(vector_path))
        except Exception:
            logger.exception(
                "ingestion.save_failed",
                extra={"document_id": document_id, "vector_path": str(vector_path)},
            )

    return [
        {
            "text": doc.page_content,
            "source_id": doc.metadata.get("source_id", ""),
            "metadata": doc.metadata,
        }
        for doc in docs
        if doc.page_content.strip()
    ]
