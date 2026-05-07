from dataclasses import dataclass

MAX_MESSAGES = 5


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
