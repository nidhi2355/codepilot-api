import inspect
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from agents.supervisor import create_agent_workflow
from agents.state import AgentState
from services.git_service import GitService
from services.parser_service import ParserService
from services.embedding_service import EmbeddingService
from services.repository_service import RepositoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Autonomous Agent"])


class TaskRequest(BaseModel):
    repo_id: int
    task: str
    repo_path: Optional[str] = None


async def _run_step(fn, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return fn(**kwargs)


async def stream_agent_execution(repo_id: int, repo_path: str, task: str):
    """Generator function that yields LangGraph state updates as Server-Sent Events (SSE)."""
    initial_state = AgentState(
        repo_id=repo_id,
        repo_path=repo_path,
        task=task,
        plan=[],
        retrieved_context=[],
        modified_files=[],
        test_results={},
        retry_count=0,
        status="starting",
        errors=[],
    )

    workflow = create_agent_workflow()
    app = workflow.compile()

    try:
        async for output in app.astream(initial_state):
            for node_name, state_update in output.items():
                logger.info(f"Completed node: {node_name}")
                payload = {
                    "node": node_name,
                    "state_update": state_update,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'node': 'end', 'status': 'completed'})}\n\n"

    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/run")
async def run_autonomous_agent(
    request: TaskRequest,
    db: AsyncSession = Depends(get_db),
):
    """Triggers the end-to-end LangGraph autonomous workflow and streams progress."""
    resolved_path = request.repo_path

    if not resolved_path:
        repo = await RepositoryService.get_repository(db, request.repo_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")

        resolved_path, needs_reindex, current_sha = await GitService.ensure_fresh_repo(
            repo.id, repo.github_url
        )

        if needs_reindex:
            logger.info(f"Auto-syncing index and embeddings for repo {repo.id} at {current_sha}")
            await _run_step(ParserService.index_repository_task, repo_id=repo.id)
            await _run_step(EmbeddingService.embed_repository_task, repo_id=repo.id)

            await db.refresh(repo)
            if repo.status != "ready":
                raise HTTPException(
                    status_code=500,
                    detail=f"Repository re-indexing failed (status: {repo.status})",
                )

            await GitService.save_indexed_sha(repo.id, current_sha)

    logger.info(f"Starting SSE agent workflow for repo_id {request.repo_id} at {resolved_path}")
    return StreamingResponse(
        stream_agent_execution(request.repo_id, resolved_path, request.task),
        media_type="text/event-stream",
    )