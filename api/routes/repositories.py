from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.schemas.repository import RepositoryCreate, RepositoryResponse
from services.repository_service import RepositoryService

router = APIRouter(prefix="/repositories", tags=["Repositories"])

@router.post("/", response_model=RepositoryResponse)
async def add_repository(repo_in: RepositoryCreate, db: AsyncSession = Depends(get_db)):
    """Accepts a GitHub URL and initializes it in the database."""
    try:
        repo = await RepositoryService.create_repository(db, repo_in)
        return repo
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add repository")