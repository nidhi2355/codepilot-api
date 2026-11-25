import logging
import os
import re
from langgraph.graph import StateGraph, END
from agents.ast_guard import audit_diff_regressions
from agents.state import AgentState
from agents.planner import plan_node
from agents.retriever import retrieve_node
from agents.coding import coding_node
from agents.testing import execute_tests_node
from agents.reflection import reflection_node
from agents.pr_creator import pr_creation_node

logger = logging.getLogger(__name__)
MAX_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "5"))


def _is_test_file(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("./")
    base = os.path.basename(norm)
    return (
        norm.startswith("tests/")
        or "/tests/" in norm
        or base.startswith("test_")
        or base.endswith("_test.py")
    )


def route_after_coding(state: AgentState) -> str:
    """Routes to testing only when valid file modifications exist on disk."""
    modified_files = state.get("modified_files", [])
    retry_count = state.get("retry_count", 0)

    if modified_files:
        return "testing"
    if retry_count >= MAX_RETRIES:
        logger.warning(f"Coding produced no modifications after {MAX_RETRIES} retries. Ending run.")
        return END
    logger.info(f"Coding produced no modifications (Retry {retry_count}). Routing to reflection.")
    return "reflection"


def route_after_testing(state: AgentState) -> str:
    """Determines the next node based on test results, diff audit, test coverage, and retry count."""
    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    retry_count = state.get("retry_count", 0)
    modified_files = state.get("modified_files", [])

    if not modified_files:
        logger.warning("No modified files present after testing. Ending run without PR creation.")
        return END

    if passed:
        has_test_file = any(_is_test_file(f) for f in modified_files)
        has_source_change = any(not _is_test_file(f) for f in modified_files)
        task_mentions_test = bool(
            re.search(r"\btests?\b|\bpytest\b|\bunit\s*test", state.get("task", ""), re.IGNORECASE)
        )
        if not has_test_file and (has_source_change or task_mentions_test):
            if retry_count >= MAX_RETRIES:
                logger.warning("Max retries reached with missing test file coverage. Ending run.")
                return END
            logger.info("Tests passed, but no test files were modified. Routing to reflection.")
            return "reflection"

        violations = audit_diff_regressions(
            state.get("repo_path", "."), modified_files, state.get("task", "")
        )
        if violations:
            if retry_count >= MAX_RETRIES:
                logger.warning(f"Max retries reached with unresolved diff regressions: {violations}")
                return END
            logger.warning(f"Diff audit detected regressions: {violations}. Routing to reflection.")
            return "reflection"

        logger.info("Tests and diff audit passed. Routing to PR creation.")
        return "pr_creator"

    if retry_count >= MAX_RETRIES:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached with failing tests. Routing to END.")
        return END

    logger.info(f"Tests failed (Retry {retry_count}). Routing to reflection.")
    return "reflection"


def create_agent_workflow() -> StateGraph:
    logger.info("Initializing LangGraph Supervisor Workflow...")
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("planner", plan_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("testing", execute_tests_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("pr_creator", pr_creation_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "planner")
    workflow.add_edge("planner", "coding")
    workflow.add_conditional_edges("coding", route_after_coding)
    workflow.add_conditional_edges("testing", route_after_testing)
    workflow.add_edge("reflection", "coding")
    workflow.add_edge("pr_creator", END)

    return workflow