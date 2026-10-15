from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

# SQLite connection string. To switch to PostgreSQL later, just change this URL.
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./codepilot.db"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # Needed only for SQLite
)

SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()

async def get_db():
    """Dependency to provide a database session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()