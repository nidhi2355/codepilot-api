import inspect
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from api.dependencies.auth import get_current_user
from api.models.user import User
from agents.supervisor import create_agent_workflow
from agents.state import AgentState
from services.crypto_service import decrypt_token
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
    issue_number: Optional[int] = None


async def _run_step(fn, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return fn(**kwargs)


async def stream_agent_execution(
    repo_id: int,
    repo_path: str,
    task: str,
    github_token: str,
    issue_number: Optional[int] = None,
):
    """Generator function that yields LangGraph state updates as Server-Sent Events (SSE)."""
    initial_state = AgentState(
        repo_id=repo_id,
        repo_path=repo_path,
        task=task,
        issue_number=issue_number,
        github_token=github_token,
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
    final_status = "completed"
    pr_created = False

    try:
        async for output in app.astream(initial_state):
            for node_name, state_update in output.items():
                logger.info(f"Completed node: {node_name}")
                if isinstance(state_update, dict):
                    if state_update.get("status") == "error":
                        final_status = "error"
                    elif state_update.get("status") == "pr_created":
                        final_status = "pr_created"
                        pr_created = True
                payload = {
                    "node": node_name,
                    "state_update": state_update,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        if not pr_created:
            final_status = "error"

        yield f"data: {json.dumps({'node': 'end', 'status': final_status})}\n\n"

    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/run")
async def run_autonomous_agent(
    request: TaskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Triggers the end-to-end LangGraph autonomous workflow for the authenticated user."""
    repo = await RepositoryService.get_repository(db, request.repo_id, current_user.id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    github_token = decrypt_token(current_user.encrypted_github_token)

    resolved_path, needs_reindex, current_sha = await GitService.ensure_fresh_repo(
        repo.id, repo.github_url, github_token=github_token
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

    logger.info(
        f"Starting SSE agent workflow for user {current_user.username}, repo_id {request.repo_id}"
    )
    return StreamingResponse(
        stream_agent_execution(
            request.repo_id,
            resolved_path,
            request.task,
            github_token,
            request.issue_number,
        ),
        media_type="text/event-stream",
    )