import logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import plan_node
from agents.retriever import retrieve_node
from agents.coding import coding_node
from agents.testing import execute_tests_node
from agents.reflection import reflection_node
from agents.pr_creator import pr_creation_node # Added import

logger = logging.getLogger(__name__)

def route_after_testing(state: AgentState) -> str:
    """Determine the next node based on test results and retry count."""
    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = 3
    
    if passed:
        logger.info("Tests passed. Routing to PR creation.")
        return "pr_creator" # Updated destination
    elif retry_count >= MAX_RETRIES:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached. Routing to END with failure.")
        return END
    else:
        logger.info(f"Tests failed (Retry {retry_count}). Routing to reflection.")
        return "reflection"

def create_agent_workflow() -> StateGraph:
    logger.info("Initializing LangGraph Supervisor Workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", plan_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("testing", execute_tests_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("pr_creator", pr_creation_node) # Added node
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "coding")
    workflow.add_edge("coding", "testing")
    
    workflow.add_conditional_edges("testing", route_after_testing)
    workflow.add_edge("reflection", "coding")
    
    # Route PR creator to END
    workflow.add_edge("pr_creator", END) 
    
    return workflow