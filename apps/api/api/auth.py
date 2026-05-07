from fastapi import APIRouter, HTTPException, status

from apps.api.schemas import LoginRequest, TokenResponse
from services.auth import auth_service

router = APIRouter(tags=["auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT",
    description="Authenticate user and return a Bearer token.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJI...",
                        "token_type": "bearer",
                    }
                }
            }
        },
        401: {
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid username or password"}
                }
            }
        },
    },
)
async def login(payload: LoginRequest) -> TokenResponse:
    """Authenticate user and return JWT access token."""

    user = auth_service.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = auth_service.issue_token(payload.username, payload.expires_minutes)
    return TokenResponse(access_token=token, token_type="bearer")
