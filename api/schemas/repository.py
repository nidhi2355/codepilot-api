from pydantic import BaseModel, HttpUrl
from datetime import datetime

class RepositoryCreate(BaseModel):
    github_url: HttpUrl

class RepositoryResponse(BaseModel):
    id: int
    github_url: str
    name: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True