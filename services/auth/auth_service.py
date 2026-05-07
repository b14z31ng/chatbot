import os

from passlib.context import CryptContext

from services.auth.jwt_handler import create_access_token, verify_token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
DEFAULT_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ADMIN_PASSWORD_HASH = pwd_context.hash(ADMIN_PASSWORD)

USERS: dict[str, dict[str, str]] = {
    "admin": {
        "username": "admin",
        "password": ADMIN_PASSWORD_HASH,
        "role": "admin",
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""

    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    """Authenticate user credentials against the in-memory store."""

    user = USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["password"]):
        return None
    return user


def get_current_user(token: str) -> dict[str, str]:
    """Get current user from a JWT token."""

    payload = verify_token(token)
    if "sub" not in payload or "role" not in payload:
        raise ValueError("Invalid token payload")
    username = payload.get("sub")
    if not isinstance(username, str):
        raise ValueError("Invalid token")
    user = USERS.get(username)
    if not user:
        raise ValueError("Invalid token")
    return user


def check_admin(user: dict[str, str]) -> bool:
    """Return True if user has admin role."""

    return user.get("role") == "admin"


def issue_token(username: str, expires_minutes: int | None = None) -> str:
    """Create access token for a user."""

    user = USERS.get(username)
    if not user:
        raise ValueError("Unknown user")
    minutes = expires_minutes or DEFAULT_EXPIRE_MINUTES
    return create_access_token({"sub": username, "role": user["role"]}, minutes)
