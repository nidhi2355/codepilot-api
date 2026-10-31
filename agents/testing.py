import logging
import subprocess
from agents.state import AgentState

logger = logging.getLogger(__name__)

# Renamed from testing_node to avoid Pytest confusion
def execute_tests_node(state: AgentState) -> dict:
    """
    LangGraph node: Executes the test suite to verify generated code.
    """
    logger.info("Testing Agent: Running test suite...")
    
    # Target the specific repository directory, defaulting to current dir
    target_dir = state.get("repo_path", ".")
    
    try:
        # Run pytest in the target directory with a strict timeout to prevent infinite loops
        result = subprocess.run(
            ["pytest"], 
            cwd=target_dir,
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        passed = result.returncode == 0
        # Combine stdout and stderr for full context
        output = result.stdout + "\n" + result.stderr
        
        logger.info(f"Testing completed. Passed: {passed}")
        
        return {
            "test_results": {
                "passed": passed,
                # Keep the last 2000 characters to prevent state bloat in the LangGraph memory
                "output": output[-2000:] 
            }
        }
        
    except subprocess.TimeoutExpired:
        logger.error("Testing failed: Execution timed out.")
        return {"test_results": {"passed": False, "output": "Execution timed out after 30 seconds."}}
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        return {"errors": [f"Testing Error: {str(e)}"]}