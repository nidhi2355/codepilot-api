import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.supervisor import create_agent_workflow
from agents.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Autonomous Agent"])

class TaskRequest(BaseModel):
    repo_id: int
    repo_path: str
    task: str

async def stream_agent_execution(request: TaskRequest):
    """Generator function that yields LangGraph state updates as Server-Sent Events (SSE)."""
    initial_state = AgentState(
        repo_id=request.repo_id,
        repo_path=request.repo_path,
        task=request.task,
        plan=[],
        retrieved_context=[],
        modified_files=[],
        test_results={},
        retry_count=0,
        status="starting",
        errors=[]
    )
    
    workflow = create_agent_workflow()
    app = workflow.compile()
    
    try:
        # .astream() yields a dict of the state update every time a node finishes
        async for output in app.astream(initial_state):
            for node_name, state_update in output.items():
                logger.info(f"Completed node: {node_name}")
                
                # Format the payload for SSE ("data: <json>\n\n")
                payload = {
                    "node": node_name,
                    "state_update": state_update
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
        # Send a final termination event
        yield f"data: {json.dumps({'node': 'end', 'status': 'completed'})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/run")
async def run_autonomous_agent(request: TaskRequest):
    """
    Triggers the end-to-end LangGraph autonomous workflow and streams progress.
    """
    logger.info(f"Starting SSE agent workflow for repo_id {request.repo_id}")
    return StreamingResponse(stream_agent_execution(request), media_type="text/event-stream")