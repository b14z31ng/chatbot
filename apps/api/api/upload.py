from pathlib import Path
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from services.auth.dependencies import require_admin
from services.rag.embeddings import get_embeddings
from services.rag.ingestion import ingest_pdf
from services.rag.vector_store import add_documents

router = APIRouter(tags=["upload"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")


@router.post(
    "/upload",
    summary="Upload PDF for ingestion",
    description=(
        "Admin-only endpoint. Upload a PDF and ingest it into the vector store."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {"document_id": "doc-123", "chunks": 12}
                }
            }
        },
        401: {
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid token"}
                }
            }
        },
        403: {
            "content": {
                "application/json": {
                    "example": {"detail": "Admin privileges required"}
                }
            }
        },
    },
)
async def upload_document(
    file: UploadFile = File(...),
    _admin: dict[str, str] = Depends(require_admin),
) -> dict[str, object]:
    """Upload PDF and ingest; admin-only."""

    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    document_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{document_id}.pdf"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(content) > 10_000_000:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    file_path.write_bytes(content)

    chunks = ingest_pdf(str(file_path), document_id=document_id)
    if not chunks:
        logger.warning("upload.no_chunks", extra={"document_id": document_id})
        return {"document_id": document_id, "chunks": 0}

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    add_documents(chunks, embeddings)
    logger.info("upload.ingested", extra={"document_id": document_id, "chunks": len(chunks)})
    return {"document_id": document_id, "chunks": len(chunks)}
