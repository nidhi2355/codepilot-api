from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.schemas.repository import RepositoryCreate, RepositoryResponse
from services.repository_service import RepositoryService
from services.git_service import GitService

router = APIRouter(prefix="/repositories", tags=["Repositories"])

@router.post("/", response_model=RepositoryResponse)
async def add_repository(
    repo_in: RepositoryCreate, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """Accepts a GitHub URL, saves to DB, and triggers a background clone."""
    try:
        repo = await RepositoryService.create_repository(db, repo_in)
        
        # Only trigger clone if it is newly created or previously failed
        if repo.status in ["pending", "error"]:
            background_tasks.add_task(
                GitService.clone_repository_task, 
                repo_id=repo.id, 
                github_url=repo.github_url
            )
            
        return repo
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to add repository")