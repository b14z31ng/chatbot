import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 100


def load_pdf(file_path: str) -> str:
    """Load text content from a PDF file."""

    reader = PdfReader(file_path)
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages_text.append(text)
    return "\n".join(pages_text).strip()


def chunk_text(
    text: str,
    chunk_size_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks using character counts."""

    if not text:
        return []
    step = max(chunk_size_chars - overlap_chars, 1)
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    logger.info(
        "ingestion.chunked",
        extra={"total_chars": length, "chunk_count": len(chunks), "chunk_size": chunk_size_chars},
    )
    return chunks


def ingest_pdf(file_path: str, document_id: str) -> list[dict[str, str]]:
    """Load and chunk a PDF, returning chunks with source ids."""

    text = load_pdf(file_path)
    chunks = chunk_text(text)
    if not chunks:
        logger.warning("ingestion.empty", extra={"document_id": document_id})
        return []
    return [
        {"text": chunk, "source_id": f"{document_id}:{index}"}
        for index, chunk in enumerate(chunks)
    ]
