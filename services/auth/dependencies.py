import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.auth import auth_service

security = HTTPBearer()
logger = logging.getLogger(__name__)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> dict[str, str]:
    """Return current user based on the Authorization header."""

    token = credentials.credentials
    try:
        return auth_service.get_current_user(token)
    except ValueError:
        request_id = getattr(request.state, "request_id", "-") if request else "-"
        logger.warning("auth.invalid_token", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(
    user: dict[str, str] = Depends(get_current_user),
    request: Request = None,
) -> dict[str, str]:
    """Ensure the current user has admin role."""

    if not auth_service.check_admin(user):
        request_id = getattr(request.state, "request_id", "-") if request else "-"
        logger.error(
            "auth.error",
            extra={"error": "admin_required", "request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
