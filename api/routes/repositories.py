from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.schemas.repository import RepositoryCreate, RepositoryResponse
from services.repository_service import RepositoryService
from services.git_service import GitService
from services.parser_service import ParserService
from services.embedding_service import EmbeddingService

router = APIRouter(prefix="/repositories", tags=["Repositories"])

@router.post("/", response_model=RepositoryResponse)
async def add_repository(
    repo_in: RepositoryCreate, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """Accepts a GitHub URL and initializes it in the database."""
    try:
        repo = await RepositoryService.create_repository(db, repo_in)
        
        if repo.status in ["pending", "error"]:
            print(f"DEBUG: Triggering background task for {repo.github_url}")
            background_tasks.add_task(
                GitService.clone_repository_task, 
                repo_id=repo.id, 
                github_url=repo.github_url
            )
            
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


@router.post("/{repo_id}/index")
async def index_repository(
    repo_id: int, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """Triggers the background parsing and symbol extraction process."""
    repo = await RepositoryService.get_repository(db, repo_id)
    
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "cloned":
        raise HTTPException(status_code=400, detail=f"Cannot index repository in status: {repo.status}")

    print(f"DEBUG: Triggering indexing task for repo ID {repo.id}")
    background_tasks.add_task(ParserService.index_repository_task, repo_id=repo.id)
    
    return {"message": "Indexing started in the background", "status": "indexing"}


@router.post("/{repo_id}/embed")
async def embed_repository(
    repo_id: int, 
    background_tasks: BackgroundTasks, 
    db: AsyncSession = Depends(get_db)
):
    """Triggers the background generation of vector embeddings for Qdrant."""
    repo = await RepositoryService.get_repository(db, repo_id)
    
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "indexed":
        raise HTTPException(status_code=400, detail=f"Cannot embed repository in status: {repo.status}. Must be 'indexed'.")
    
    print(f"DEBUG: Triggering embedding task for repo ID {repo.id}")
    background_tasks.add_task(EmbeddingService.embed_repository_task, repo_id=repo.id)
    
    return {"message": "Embedding started in the background", "status": "embedding"}