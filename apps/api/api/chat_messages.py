"""Message endpoints: load history and send a new message inside a chat."""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session, joinedload

from apps.api.schemas import MessageRead, MessageSend
from services.auth.dependencies import get_current_user
from services.chat_service import handle_chat
from services.db.database import get_db
from services.db.models import Chat, Message, User
from services.llm.memory import Message as MemMessage

router = APIRouter(prefix="/chats", tags=["messages"])
logger = logging.getLogger(__name__)


def _get_db_user(user: dict, db: Session) -> User:
    db_user = db.query(User).filter(User.username == user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return db_user


def _get_owned_chat(chat_id: int, db_user: User, db: Session) -> Chat:
    # Eagerly load documents so relationship is available outside the ORM session
    chat = (
        db.query(Chat)
        .options(joinedload(Chat.documents))
        .filter(Chat.id == chat_id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    if chat.user_id != db_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return chat


def _auto_title(first_message: str) -> str:
    """Generate a short title from the first message (<=40 chars)."""
    words = first_message.strip().split()
    title = " ".join(words[:6])
    return title[:40] if len(title) <= 40 else title[:37] + "..."


@router.get("/{chat_id}/messages", response_model=list[MessageRead])
def get_messages(
    chat_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MessageRead]:
    """Return all messages for a chat in chronological order."""
    db_user = _get_db_user(user, db)
    _get_owned_chat(chat_id, db_user, db)
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [
        MessageRead(
            id=m.id,
            chat_id=m.chat_id,
            role=m.role,
            content=m.content,
            confidence=m.confidence,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.post("/{chat_id}/messages", response_model=MessageRead)
def send_message(
    chat_id: int,
    payload: MessageSend,
    request: Request,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageRead:
    """Send a message, run RAG, save both user+assistant messages, return assistant reply."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)

    request_id = getattr(request.state, "request_id", "-")

    # Auto-title from first user message
    if chat.title == "New Chat" and not chat.messages:
        chat.title = _auto_title(payload.message)

    # ── Chat-scoped document context ────────────────────────────────────────
    # Collect ALL document UUIDs permanently linked to this chat.
    # These are set at upload time and persist for the entire conversation.
    # This is what gives the chatbot ChatGPT-style document memory.
    document_uuids = [doc.document_uuid for doc in chat.documents]

    logger.info(
        "chat.context",
        extra={
            "chat_id": chat_id,
            "document_count": len(document_uuids),
            "document_ids": document_uuids,
            "request_id": request_id,
        },
    )

    # Save user message first so history is accurate
    user_msg = Message(
        chat_id=chat_id,
        role="user",
        content=payload.message,
    )
    db.add(user_msg)
    db.flush()

    conversation_id = f"chat-{chat_id}"

    # Load last 6 turns from DB for conversational continuity
    # This gives the LLM context for pronoun resolution (they/it/this)
    recent_msgs = (
        db.query(Message)
        .filter(Message.chat_id == chat_id, Message.id != user_msg.id)
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )
    recent_msgs.reverse()
    mem_history = [MemMessage(role=m.role, content=m.content) for m in recent_msgs]

    # ── RAG pipeline with strict chat-scoped retrieval ────────────────────────
    # IMPORTANT: always pass document_uuids as-is, even if empty.
    # - Non-empty list -> retrieval scoped to this chat's documents only
    # - Empty list     -> vector store returns [] immediately (no global fallback)
    # Never pass None here — that would trigger unscoped global retrieval.
    result = handle_chat(
        message=payload.message,
        conversation_id=conversation_id,
        document_ids=document_uuids,
        chat_id=chat_id,
        request_id=request_id,
        injected_history=mem_history,
        user_id=str(db_user.id),
    )

    # Save assistant response
    assistant_msg = Message(
        chat_id=chat_id,
        role="assistant",
        content=result.answer,
        confidence=result.confidence,
    )
    db.add(assistant_msg)
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(assistant_msg)

    response = MessageRead(
        id=assistant_msg.id,
        chat_id=assistant_msg.chat_id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        confidence=assistant_msg.confidence,
        created_at=assistant_msg.created_at,
    )

    logger.info(
        "chat.response_payload",
        extra={"request_id": request_id, "payload": response.model_dump()},
    )

    return response
