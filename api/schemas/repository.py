from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime

class RepositoryCreate(BaseModel):
    github_url: HttpUrl = Field(
        ..., 
        description="The full public GitHub URL of the repository to ingest.", 
        json_schema_extra={"example": "https://github.com/nidhi2355/codepilot-api"}
    )

class RepositoryResponse(BaseModel):
    id: int = Field(..., description="Internal database ID.")
    github_url: str = Field(..., description="The parsed GitHub URL.")
    name: str = Field(..., description="The extracted repository name.")
    status: str = Field(..., description="Current state: pending, cloned, indexing, embedding, ready, or error.")
    created_at: datetime

    class Config:
        from_attributes = True