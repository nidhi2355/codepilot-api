import logging
from agents.state import AgentState
from services.search_service import SearchService

logger = logging.getLogger(__name__)

def retrieve_node(state: AgentState) -> dict:
    """
    LangGraph node: Retrieves relevant codebase context based on the user task and plan.
    """
    logger.info("Retriever Agent: Fetching codebase context...")
    
    # Safely get repo_id, defaulting to 1 for testing purposes
    repo_id = state.get("repo_id", 1) 
    task = state["task"]
    plan = state.get("plan", [])
    
    # Combine the main task and the first few plan steps to form a rich search query
    search_query = task + " " + " ".join(plan[:3])
    
    try:
        # Call your static, synchronous search method
        results = SearchService.search_codebase(repo_id=repo_id, query=search_query, limit=5)
        
        logger.info(f"Retrieved {len(results)} context chunks from Qdrant.")
        return {"retrieved_context": results}
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return {"retrieved_context": [], "errors": [f"Retriever Error: {str(e)}"]}