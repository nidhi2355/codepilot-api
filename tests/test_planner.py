import os
import pytest
from agents.state import AgentState
from agents.planner import plan_node
from agents.supervisor import create_agent_workflow

# Ensure you have your GROQ_API_KEY in your .env file
# and loaded in your environment before running this.

def test_supervisor_compiles():
    """Test that the LangGraph workflow compiles without errors."""
    workflow = create_agent_workflow()
    app = workflow.compile()
    assert app is not None

def test_planner_node():
    """Test that the Planner agent generates a structured step-by-step plan."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Skipping test: GROQ_API_KEY is not set.")

    # Mock starting state
    initial_state = AgentState(
        task="The /login endpoint is returning a 500 error. Fix it.",
        plan=[],
        retrieved_context=[],
        modified_files=[],
        test_results={},
        retry_count=0,
        status="planning",
        errors=[]
    )
    
    # Run just the planner node
    result = plan_node(initial_state)
    
    # Validate output
    assert "plan" in result
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0
    print("\nGenerated Plan Steps:")
    for step in result["plan"]:
        print(f"- {step}")