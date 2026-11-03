import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.supervisor import create_agent_workflow
from agents.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Autonomous Agent"])

class TaskRequest(BaseModel):
    repo_id: int
    repo_path: str
    task: str

@router.post("/run")
async def run_autonomous_agent(request: TaskRequest):
    """
    Triggers the end-to-end LangGraph autonomous coding workflow.
    """
    logger.info(f"Starting agent workflow for repo_id {request.repo_id}")
    
    try:
        # 1. Initialize the LangGraph state
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
        
        # 2. Compile and execute the workflow
        workflow = create_agent_workflow()
        app = workflow.compile()
        
        # This will block until the graph reaches the END node
        final_state = app.invoke(initial_state)
        
        # 3. Return a clean summary of the execution
        return {
            "message": "Autonomous workflow completed.",
            "modified_files": final_state.get("modified_files", []),
            "test_results": final_state.get("test_results", {}),
            "retry_count": final_state.get("retry_count", 0),
            "errors": final_state.get("errors", [])
        }
        
    except Exception as e:
        logger.error(f"Agent workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")