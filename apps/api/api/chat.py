from fastapi import APIRouter, Depends, Request

from apps.api.schemas import ChatRequest, ChatResponse
from services.auth.dependencies import get_current_user
from services.chat_service import handle_chat

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Grounded chat response",
    description=(
        "Return answer grounded only in retrieved context. Requires Bearer token."
    ),
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "answer": "Not in knowledge base.",
                        "citations": [],
                        "confidence": 0.0,
                        "conversation_id": "conv-1",
                    }
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
                    "example": {"detail": "Forbidden"}
                }
            }
        },
    },
)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    _user: dict[str, str] = Depends(get_current_user),
) -> ChatResponse:
    """Return grounded response based on retrieved context."""

    request_id = getattr(request.state, "request_id", "-")
    result = handle_chat(
        message=payload.message,
        conversation_id=payload.conversation_id,
        request_id=request_id,
    )
    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        conversation_id=result.conversation_id,
    )
