import os
import pytest
from agents.state import AgentState
from agents.coding import coding_node

def test_coding_node():
    """Test that the Coding agent generates code and writes it to a file."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Skipping test: GROQ_API_KEY is not set.")
        
    # Define a temporary test file path
    test_file_path = "tests/temp_test_output.py"
        
    initial_state = AgentState(
        repo_id=1,
        task="Create a simple Python function that adds two numbers together.",
        plan=["1. Create a file named temp_test_output.py", "2. Write an add_numbers function."],
        retrieved_context=[{
            "file_path": test_file_path, 
            "content": "# Placeholder"
        }],
        modified_files=[],
        test_results={},
        retry_count=0,
        status="coding",
        errors=[]
    )
    
    # Run the coding node
    result = coding_node(initial_state)
    
    # Assert state updates
    assert "modified_files" in result
    assert len(result["modified_files"]) > 0
    
    # Verify the file was actually written to disk
    written_file = result["modified_files"][0]
    assert os.path.exists(written_file)
    
    with open(written_file, "r") as f:
        content = f.read()
        print(f"\nWritten Content:\n{content}")
        assert "def" in content
        
    # Clean up the test file
    if os.path.exists(written_file):
        os.remove(written_file)