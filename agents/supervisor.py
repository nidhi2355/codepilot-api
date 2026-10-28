import logging
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import plan_node

logger = logging.getLogger(__name__)

def create_agent_workflow() -> StateGraph:
    logger.info("Initializing LangGraph Supervisor Workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", plan_node)
    
    workflow.set_entry_point("planner")
    # Add an edge from planner to END so it's not a dead-end
    workflow.add_edge("planner", END)
    
    return workflow