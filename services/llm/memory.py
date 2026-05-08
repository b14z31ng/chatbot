"""In-memory short-term conversation memory with DB history loading support."""
from dataclasses import dataclass

MAX_MESSAGES = 10


@dataclass(frozen=True)
class Message:
    """Single conversation message."""

    role: str
    content: str


_STORE: dict[str, list[Message]] = {}


def add_message(conversation_id: str | None, role: str, content: str) -> None:
    """Add a message to short-term memory."""

    if not conversation_id:
        return
    history = _STORE.setdefault(conversation_id, [])
    history.append(Message(role=role, content=content))
    if len(history) > MAX_MESSAGES:
        _STORE[conversation_id] = history[-MAX_MESSAGES:]


def get_history(conversation_id: str | None) -> list[Message]:
    """Get recent message history for a conversation."""

    if not conversation_id:
        return []
    return list(_STORE.get(conversation_id, []))


def load_history_from_db(db_messages: list) -> list[Message]:
    """Convert a list of DB Message ORM objects into in-memory Message objects.

    Used by the chat_messages endpoint to pass DB-backed history into the
    prompt builder without coupling the LLM layer to SQLAlchemy.
    """
    return [Message(role=m.role, content=m.content) for m in db_messages]


def seed_from_db(conversation_id: str, db_messages: list) -> None:
    """Pre-warm in-memory store from DB messages (called on chat load)."""
    if not conversation_id:
        return
    _STORE[conversation_id] = [
        Message(role=m.role, content=m.content)
        for m in db_messages[-MAX_MESSAGES:]
    ]
