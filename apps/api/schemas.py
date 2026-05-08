"""Pydantic schemas for the RAG chatbot API."""
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Login request payload."""

    username: str = Field(..., description="Username", example="admin")
    password: str = Field(..., description="Password", example="admin")
    expires_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Token lifetime in minutes",
        example=60,
    )


class TokenResponse(BaseModel):
    """JWT token response payload."""

    access_token: str = Field(
        ..., description="JWT access token", example="eyJhbGciOiJI..."
    )
    token_type: str = Field(..., description="Token type", example="bearer")


# ---------------------------------------------------------------------------
# Legacy chat (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Chat request payload (legacy /chat endpoint)."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query (max 2000 chars)",
        example="What is this document about?",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id for short-term memory",
        example="conv-1",
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Trim leading/trailing whitespace from message."""
        return v.strip() if isinstance(v, str) else v


class ChatResponse(BaseModel):
    """Chat response payload."""

    answer: str = Field(
        ..., description="Grounded answer from retrieved context", example="Not in knowledge base."
    )
    citations: list[str] = Field(
        ..., description="Source ids used for grounding", example=["doc-1:0", "doc-1:1"]
    )
    confidence: float = Field(
        ..., description="Confidence score between 0 and 95", example=72.5
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id associated with response",
        example="conv-1",
    )


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

class ChatCreate(BaseModel):
    """Create a new chat session."""
    title: str | None = Field(default=None, max_length=200, description="Optional initial title")


class ChatUpdate(BaseModel):
    """Update chat title."""
    title: str = Field(..., min_length=1, max_length=200, description="New title")


class ChatRead(BaseModel):
    """Chat record returned from API."""
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    message_count: int = 0

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class MessageSend(BaseModel):
    """Send a message inside a specific chat."""
    message: str = Field(
        ..., min_length=1, max_length=2000,
        description="User message",
        example="Summarize this document.",
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class MessageRead(BaseModel):
    """Message record returned from API."""
    id: int
    chat_id: int
    role: str
    content: str
    confidence: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentRead(BaseModel):
    """Document record returned from API."""
    id: int
    filename: str
    document_uuid: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}
