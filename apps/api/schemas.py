from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query (max 1000 chars)",
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
        ..., description="Confidence score between 0 and 1", example=0.6
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id associated with response",
        example="conv-1",
    )


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
