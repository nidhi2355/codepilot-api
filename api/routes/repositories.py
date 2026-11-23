import inspect
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db, SessionLocal
from api.schemas.repository import RepositoryCreate, RepositoryResponse
from services.repository_service import RepositoryService
from services.git_service import GitService
from services.parser_service import ParserService
from services.embedding_service import EmbeddingService
from api.schemas.search import SearchQuery
from services.search_service import SearchService
from api.schemas.chat import ChatRequest
from services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/repositories", tags=["Repositories"])


async def _run_step(fn, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return fn(**kwargs)


async def ingest_repository_pipeline(repo_id: int, github_url: str):
    """Runs clone -> index -> embed end-to-end in the background."""
    await _run_step(GitService.clone_repository_task, repo_id=repo_id, github_url=github_url)

    async with SessionLocal() as db:
        repo = await RepositoryService.get_repository(db, repo_id)
        if not repo or repo.status != "cloned":
            logger.error(f"Stopping pipeline for repo {repo_id}: status is {repo.status if repo else 'missing'}")
            return

    await _run_step(ParserService.index_repository_task, repo_id=repo_id)

    async with SessionLocal() as db:
        repo = await RepositoryService.get_repository(db, repo_id)
        if not repo or repo.status != "indexed":
            logger.error(f"Stopping pipeline for repo {repo_id}: status is {repo.status if repo else 'missing'}")
            return

    await _run_step(EmbeddingService.embed_repository_task, repo_id=repo_id)


@router.post("/", response_model=RepositoryResponse)
async def add_repository(
    repo_in: RepositoryCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Accepts a GitHub URL, saves it, and runs clone -> index -> embed in the background."""
    try:
        repo = await RepositoryService.create_repository(db, repo_in)

        if repo.status == "ready":
            repo.status = "pending"
            await db.commit()

        if repo.status in ["pending", "error"]:
            background_tasks.add_task(
                ingest_repository_pipeline,
                repo_id=repo.id,
                github_url=repo.github_url,
            )

        return repo
    except Exception as e:
        logger.error(f"Failed to add repository: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add repository")


@router.get("/{repo_id}", response_model=RepositoryResponse)
async def get_repository(repo_id: int, db: AsyncSession = Depends(get_db)):
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    return repo


@router.get("/", response_model=list[RepositoryResponse])
async def list_repositories(db: AsyncSession = Depends(get_db)):
    try:
        return await RepositoryService.get_repositories(db)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve repositories")


@router.post("/{repo_id}/index")
async def index_repository(
    repo_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "cloned":
        raise HTTPException(status_code=400, detail=f"Cannot index repository in status: {repo.status}")

    background_tasks.add_task(ParserService.index_repository_task, repo_id=repo.id)
    return {"message": "Indexing started in the background", "status": "indexing"}


@router.post("/{repo_id}/embed")
async def embed_repository(
    repo_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "indexed":
        raise HTTPException(status_code=400, detail=f"Cannot embed repository in status: {repo.status}. Must be 'indexed'.")

    background_tasks.add_task(EmbeddingService.embed_repository_task, repo_id=repo.id)
    return {"message": "Embedding started in the background", "status": "embedding"}


@router.post("/{repo_id}/search")
async def search_repository(
    repo_id: int,
    search_req: SearchQuery,
    db: AsyncSession = Depends(get_db),
):
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail=f"Repository not ready for search. Current status: {repo.status}")

    try:
        results = SearchService.search_codebase(repo_id, search_req.query, search_req.limit)
        return {"query": search_req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/{repo_id}/ask")
async def ask_repository(
    repo_id: int,
    chat_req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    repo = await RepositoryService.get_repository(db, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")
    if repo.status != "ready":
        raise HTTPException(status_code=400, detail="Repository not ready. Embeddings must be generated first.")

    try:
        search_results = SearchService.search_codebase(repo_id, chat_req.query, chat_req.limit)
        llm_response = LLMService.generate_answer(chat_req.query, search_results)
        referenced_files = list(set([res["file_path"] for res in search_results]))
        return {
            "query": chat_req.query,
            "answer": llm_response,
            "context_files": referenced_files,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")