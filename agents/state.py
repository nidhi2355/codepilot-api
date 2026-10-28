from typing import List, Dict, Any, TypedDict

class AgentState(TypedDict):
    """
    Maintains the state of the LangGraph execution.
    """
    task: str
    plan: List[str]
    retrieved_context: List[Dict[str, Any]]
    modified_files: List[str]
    test_results: Dict[str, Any]
    retry_count: int
    status: str
    errors: List[str]