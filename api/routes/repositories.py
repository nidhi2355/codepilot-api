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
    

@router.get("/{repo_id}", response_model=RepositoryResponse)
async def get_repository(repo_id: int, db: AsyncSession = Depends(get_db)):
    """Fetches a specific repository by its ID to check its cloning status."""
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.get("/", response_model=list[RepositoryResponse])
async def list_repositories(db: AsyncSession = Depends(get_db)):
    """Lists all ingested repositories in the database."""
    try:
        repos = await RepositoryService.get_repositories(db)
        return repos
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve repositories")