"""Chat CRUD endpoints: list, create, get, delete, rename."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.schemas import ChatCreate, ChatRead, ChatUpdate
from services.auth.dependencies import get_current_user
from services.db.database import get_db
from services.db.models import Chat, User

router = APIRouter(prefix="/chats", tags=["chats"])


def _get_db_user(user: dict, db: Session) -> User:
    """Resolve the DB User record from the JWT-authenticated user dict."""
    db_user = db.query(User).filter(User.username == user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return db_user


def _get_owned_chat(chat_id: int, db_user: User, db: Session) -> Chat:
    """Return chat if it belongs to the authenticated user, else 404/403."""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    if chat.user_id != db_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return chat


@router.get("", response_model=list[ChatRead])
def list_chats(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatRead]:
    """Return all chats for the authenticated user, newest first."""
    db_user = _get_db_user(user, db)
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == db_user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )
    results = []
    for c in chats:
        results.append(
            ChatRead(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                document_count=len(c.documents),
                message_count=len(c.messages),
            )
        )
    return results


@router.post("", response_model=ChatRead, status_code=status.HTTP_201_CREATED)
def create_chat(
    payload: ChatCreate = ChatCreate(),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatRead:
    """Create a new empty chat session."""
    db_user = _get_db_user(user, db)
    chat = Chat(
        user_id=db_user.id,
        title=payload.title or "New Chat",
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return ChatRead(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        document_count=0,
        message_count=0,
    )


@router.get("/{chat_id}", response_model=ChatRead)
def get_chat(
    chat_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatRead:
    """Get a single chat by ID."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)
    return ChatRead(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        document_count=len(chat.documents),
        message_count=len(chat.messages),
    )


@router.patch("/{chat_id}", response_model=ChatRead)
def rename_chat(
    chat_id: int,
    payload: ChatUpdate,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatRead:
    """Rename a chat."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)
    chat.title = payload.title
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(chat)
    return ChatRead(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        document_count=len(chat.documents),
        message_count=len(chat.messages),
    )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(
    chat_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a chat and all its messages."""
    db_user = _get_db_user(user, db)
    chat = _get_owned_chat(chat_id, db_user, db)
    db.delete(chat)
    db.commit()
