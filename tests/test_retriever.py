import pytest
from agents.state import AgentState
from agents.retriever import retrieve_node

def test_retrieve_node():
    """Test that the Retriever agent fetches context from the vector store."""
    
    # Mock starting state mimicking the output of the Planner
    initial_state = AgentState(
        repo_id=1,  # Provide a valid repo_id that exists in your local Qdrant
        task="The /login endpoint is returning a 500 error. Fix it.",
        plan=["1. Inspect auth.py", "2. Check database connection in login route."],
        retrieved_context=[],
        modified_files=[],
        test_results={},
        retry_count=0,
        status="retrieving",
        errors=[]
    )
    
    # Execute the synchronous node
    result = retrieve_node(initial_state)
    
    # Validate the output
    assert "retrieved_context" in result
    assert isinstance(result["retrieved_context"], list)
    
    print(f"\nRetrieved {len(result['retrieved_context'])} chunks.")
    if len(result["retrieved_context"]) > 0:
        print("First chunk snippet:", str(result["retrieved_context"][0])[:150] + "...")