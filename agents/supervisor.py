import logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import plan_node
from agents.retriever import retrieve_node
from agents.coding import coding_node
from agents.testing import execute_tests_node # Updated import

logger = logging.getLogger(__name__)

def create_agent_workflow() -> StateGraph:
    logger.info("Initializing LangGraph Supervisor Workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", plan_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("testing", execute_tests_node) # Updated node name
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "coding")
    workflow.add_edge("coding", "testing")
    workflow.add_edge("testing", END)
    
    return workflow