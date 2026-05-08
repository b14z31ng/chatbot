"""Chat-scoped document upload and management endpoints."""
import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from apps.api.schemas import DocumentRead
from services.auth.dependencies import get_current_user
from services.db.database import get_db
from services.db.models import Chat, Document, User
from services.rag.embeddings import get_embeddings
from services.rag.ingestion import ingest_pdf
from services.rag.vector_store import add_documents, save_store

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10_000_000  # 10 MB


def _get_db_user(user: dict, db: Session) -> User:
    db_user = db.query(User).filter(User.username == user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return db_user


def _get_owned_chat(chat_id: int, db_user: User, db: Session) -> Chat:
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    if chat.user_id != db_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return chat


@router.post("/chats/{chat_id}/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_to_chat(
    chat_id: int,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentRead:
    """Upload a PDF and link it to a specific chat."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)

    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Sanitize filename
    safe_name = Path(filename).name.replace(" ", "_")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    # SHA256 dedup: check if this EXACT file is already linked to THIS chat
    # (not globally — that would leak doc UUIDs across chats)
    sha256 = hashlib.sha256(content).hexdigest()
    already_in_chat = (
        db.query(Document)
        .join(Document.chats)
        .filter(
            Document.sha256 == sha256,
            Document.user_id == db_user.id,
            Chat.id == chat_id,
        )
        .first()
    )

    if already_in_chat:
        # Same file already indexed for this specific chat — skip re-ingestion
        return DocumentRead(
            id=already_in_chat.id,
            filename=already_in_chat.filename,
            document_uuid=already_in_chat.document_uuid,
            uploaded_at=already_in_chat.uploaded_at,
        )

    # Save file — each chat gets its own document UUID (ensures FAISS isolation)
    document_uuid = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{document_uuid}.pdf"
    file_path.write_bytes(content)

    # Ingest into FAISS under this chat's document UUID
    chunks = ingest_pdf(
        str(file_path),
        document_id=document_uuid,
        chat_id=chat_id,
        filename=safe_name,
    )
    if not chunks:
        logger.warning("chat_upload.no_chunks", extra={"document_uuid": document_uuid})
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    add_documents(chunks, embeddings, chat_id=chat_id)
    save_store(chat_id=chat_id)

    logger.info(
        "chat_upload.ingested",
        extra={
            "chat_id": chat_id,
            "document_uuid": document_uuid,
            "chunks": len(chunks),
        },
    )

    # Save document record linked to this chat
    doc = Document(
        user_id=db_user.id,
        filename=safe_name,
        file_path=str(file_path),
        document_uuid=document_uuid,
        sha256=sha256,
    )
    db.add(doc)
    db.flush()
    chat.documents.append(doc)
    db.commit()
    db.refresh(doc)

    return DocumentRead(
        id=doc.id,
        filename=doc.filename,
        document_uuid=doc.document_uuid,
        uploaded_at=doc.uploaded_at,
    )


@router.get("/chats/{chat_id}/documents", response_model=list[DocumentRead])
def list_chat_documents(
    chat_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DocumentRead]:
    """List all documents linked to a chat."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)
    return [
        DocumentRead(
            id=d.id,
            filename=d.filename,
            document_uuid=d.document_uuid,
            uploaded_at=d.uploaded_at,
        )
        for d in chat.documents
    ]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Remove a document from the user's library (unlinks from all chats)."""
    db_user = _get_db_user(user, db)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.user_id != db_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Optionally delete the physical file
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(doc)
    db.commit()
