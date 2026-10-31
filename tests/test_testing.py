import os
import shutil
import pytest
from agents.state import AgentState
from agents.testing import execute_tests_node # Updated import

def test_testing_node():
    """Test that the Testing agent executes pytest and captures results."""
    
    # Create a temporary sandboxed directory for testing
    sandbox_dir = "tests/temp_sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # 1. Create a dummy python file
    with open(f"{sandbox_dir}/math_ops.py", "w") as f:
        f.write("def add(a, b):\n    return a + b\n")
        
    # 2. Create a dummy pytest file that tests the math_ops.py file
    with open(f"{sandbox_dir}/test_math.py", "w") as f:
        f.write("from math_ops import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
        
    # Mock starting state mimicking the output of the Coding agent
    initial_state = AgentState(
        repo_id=1,
        repo_path=sandbox_dir,
        task="Test the math operations.",
        plan=[],
        retrieved_context=[],
        modified_files=[f"{sandbox_dir}/math_ops.py"],
        test_results={},
        retry_count=0,
        status="testing",
        errors=[]
    )
    
    # Run the newly renamed testing node
    result = execute_tests_node(initial_state)
    
    # Clean up the sandbox directory
    shutil.rmtree(sandbox_dir, ignore_errors=True)
    
    # Assert state updates
    assert "test_results" in result
    assert result["test_results"]["passed"] is True
    assert "passed in" in result["test_results"]["output"] or "1 passed" in result["test_results"]["output"]
    
    print(f"\nTest Execution Output:\n{result['test_results']['output']}")