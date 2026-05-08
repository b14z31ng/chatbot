"""SQLAlchemy database engine and session factory for rag_chat.db."""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DB_PATH = Path("rag_chat.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and seed initial data."""
    # Import models so their tables are registered on Base.metadata
    from services.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _seed_admin_user()


def _seed_admin_user() -> None:
    """Ensure the admin user exists in the DB (seeded from env/in-memory store)."""
    import os
    from passlib.context import CryptContext
    from services.db.models import User

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin")
            user = User(
                username="admin",
                password_hash=pwd_context.hash(admin_password),
                role="admin",
            )
            db.add(user)
            db.commit()
    finally:
        db.close()
