from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text
from api.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(BigInteger, unique=True, index=True, nullable=False)
    username = Column(String, index=True, nullable=False)
    avatar_url = Column(String, nullable=True)
    email = Column(String, nullable=True)
    encrypted_github_token = Column(Text, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )