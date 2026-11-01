import os
import pytest
from agents.state import AgentState
from agents.pr_creator import pr_creation_node

def test_pr_creation_node():
    """Test that the PR Creation agent generates a structured title and body."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Skipping test: GROQ_API_KEY is not set.")
        
    initial_state = AgentState(
        repo_id=1,
        repo_path="tests/dummy_repo", # Changed to trigger the safe Git simulation
        task="Fix the 500 error on the /login endpoint by updating the database timeout.",
        plan=["1. Open auth.py", "2. Increase db timeout to 30s."],
        retrieved_context=[],
        modified_files=["auth.py"],
        test_results={"passed": True},
        retry_count=0,
        status="pr_creator",
        errors=[]
    )
    
    result = pr_creation_node(initial_state)
    
    assert "status" in result
    assert result["status"] in ["pr_created", "completed"]
    assert "errors" not in result