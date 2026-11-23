import asyncio
import json
import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from git import Repo
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import get_db
from agents.supervisor import create_agent_workflow
from agents.state import AgentState
from services.git_service import REPOS_DIR
from services.repository_service import RepositoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Autonomous Agent"])


class TaskRequest(BaseModel):
    repo_id: int
    task: str
    repo_path: Optional[str] = None


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

        repo_name = repo.github_url.rstrip("/").split("/")[-1]
        resolved_path = os.path.join(REPOS_DIR, repo_name)

        if not os.path.exists(resolved_path):
            logger.info(f"Re-cloning {repo.github_url} into {resolved_path}")
            await asyncio.to_thread(Repo.clone_from, repo.github_url, resolved_path)

    logger.info(f"Starting SSE agent workflow for repo_id {request.repo_id} at {resolved_path}")
    return StreamingResponse(
        stream_agent_execution(request.repo_id, resolved_path, request.task),
        media_type="text/event-stream",
    )