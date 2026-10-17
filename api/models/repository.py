from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from datetime import datetime, timezone
from api.database import Base

class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)
    github_url = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class RepositoryFile(Base):
    __tablename__ = "repository_files"

    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"))
    path = Column(String, nullable=False)
    language = Column(String)

class Symbol(Base):
    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("repository_files.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    start_line = Column(Integer)
    end_line = Column(Integer)