import os
import pytest
from langgraph.graph import END
from agents.state import AgentState
from agents.reflection import reflection_node
from agents.supervisor import route_after_testing

def test_reflection_node():
    """Test that the Reflection agent analyzes a failed test and generates feedback."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Skipping test: GROQ_API_KEY is not set.")
        
    # Mock a state where a test just failed
    initial_state = AgentState(
        repo_id=1,
        repo_path=".",
        task="Write an add function",
        plan=[],
        retrieved_context=[],
        modified_files=["math_ops.py"],
        test_results={
            "passed": False,
            "output": "AssertionError: expected 5, got 4. test_math.py:line 12 assert add(2, 3) == 5"
        },
        retry_count=0,
        status="testing",
        errors=[]
    )
    
    # Run the reflection node
    result = reflection_node(initial_state)
    
    # Assert it incremented the retry count and appended feedback
    assert "retry_count" in result
    assert result["retry_count"] == 1
    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Retry 1 Feedback:" in result["errors"][0]
    
    print(f"\nGenerated Feedback:\n{result['errors'][0]}")

def test_routing_logic():
    """Test that the supervisor correctly routes based on state."""
    
    # Scenario 1: Tests Passed -> Should go to END
    state_pass = AgentState(test_results={"passed": True}, retry_count=0)
    assert route_after_testing(state_pass) == END
    
    # Scenario 2: Tests Failed, under retry limit -> Should loop to reflection
    state_fail_retry = AgentState(test_results={"passed": False}, retry_count=1)
    assert route_after_testing(state_fail_retry) == "reflection"
    
    # Scenario 3: Tests Failed, hit retry limit (3) -> Should go to END
    state_fail_max = AgentState(test_results={"passed": False}, retry_count=3)
    assert route_after_testing(state_fail_max) == END