import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

ALGORITHM = "HS256"
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set")


def create_access_token(data: dict[str, object], expires_minutes: int) -> str:
    """Create a signed JWT access token."""

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict[str, object]:
    """Verify and decode a JWT token."""

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc
