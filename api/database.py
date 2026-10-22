import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from sqlalchemy.pool import NullPool

load_dotenv()

# Fallback to local SQLite if DATABASE_URL is not set (useful for local testing)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/codepilot.db")

# SQLite requires specific connect args to prevent thread issues. Postgres does not.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool, # Disables SQLAlchemy's internal pooling
    connect_args={
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0,
    }
)

SessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with SessionLocal() as session:
        yield session
