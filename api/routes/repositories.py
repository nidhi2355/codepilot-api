from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.schemas.repository import RepositoryCreate, RepositoryResponse
from services.repository_service import RepositoryService
from services.git_service import GitService
from services.parser_service import ParserService
from services.embedding_service import EmbeddingService
from api.schemas.search import SearchQuery
from services.search_service import SearchService
from api.schemas.chat import ChatRequest
from services.llm_service import LLMService

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
        
        # FIX: If the repo is already processed, reset it to trigger a fresh sync
        if repo.status == "ready":
            repo.status = "pending"
            await db.commit()
            
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


@router.post("/{repo_id}/search")
async def search_repository(
    repo_id: int, 
    search_req: SearchQuery, 
    db: AsyncSession = Depends(get_db)
):
    """Searches the repository's embedded code chunks based on semantic similarity."""
    repo = await RepositoryService.get_repository(db, repo_id)
    
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready for search. Current status: {repo.status}")

    # Execute vector search
    try:
        results = SearchService.search_codebase(repo_id, search_req.query, search_req.limit)
        return {"query": search_req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/{repo_id}/ask")
async def ask_repository(
    repo_id: int, 
    chat_req: ChatRequest, 
    db: AsyncSession = Depends(get_db)
):
    """Retrieves code context and generates an LLM answer."""
    repo = await RepositoryService.get_repository(db, repo_id)
    
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail="Repository not ready. Embeddings must be generated first.")

    try:
        # 1. Retrieve relevant chunks using existing SearchService
        search_results = SearchService.search_codebase(repo_id, chat_req.query, chat_req.limit)
        
        # 2. Pass chunks and query to the LLM
        llm_response = LLMService.generate_answer(chat_req.query, search_results)
        
        # 3. Deduplicate file paths to show which files were referenced
        referenced_files = list(set([res["file_path"] for res in search_results]))
        
        return {
            "query": chat_req.query,
            "answer": llm_response,
            "context_files": referenced_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")