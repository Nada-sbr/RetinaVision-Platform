from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# Database URL configuration
# For local development, default to SQLite if PostgreSQL is not specified.
# This makes it easy to run tests and develop without PostgreSQL installed,
# but we configure it to easily swap to PostgreSQL in production via environment variables.
from app.core import config

DATABASE_URL = config.DATABASE_URL

# If using SQLite for testing or fallback
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW,
        pool_pre_ping=config.DB_POOL_PRE_PING
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency generator for FastAPI endpoints to yield a database session
    and close it after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
