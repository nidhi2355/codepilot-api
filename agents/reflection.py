import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from agents.ast_guard import audit_diff_regressions
from agents.llm_utils import invoke_structured_groq
from agents.state import AgentState

logger = logging.getLogger(__name__)


class ReflectionOutput(BaseModel):
    feedback: str = Field(
        description="Root-cause diagnosis with exact file/line numbers and behavioral requirements, without code blocks."
    )


def _summarize_coding_guidance(retry_num: int, latest_err: str) -> str:
    """Produces a non-repetitive, actionable retry hint for pre-test coding validation errors."""
    hints: list[str] = []
    if "SyntaxError" in latest_err:
        hints.append("fix the Python SyntaxError and check indentation/colons")
    if "Missing unit tests" in latest_err:
        hints.append("add unit tests in tests/test_<module>.py alongside your source edits")
    if "Missing source code" in latest_err:
        hints.append("include implementation edits in src/ alongside your unit tests")
    if "target_code" in latest_err or "did not match" in latest_err:
        hints.append("copy exact lines from Complete Target Files From Disk into target_code")
    if "Deleted " in latest_err or "Changed default value" in latest_err:
        hints.append("preserve all existing methods, functions, and default parameter values")

    if not hints:
        hints.append("address the coding validation issue above and regenerate valid edits")

    return f"Retry {retry_num} Guidance: {', '.join(hints).capitalize()}."


def reflection_node(state: AgentState) -> dict:
    """
    LangGraph node: Diagnoses coding/test/diff failures and returns ONLY the new
    guidance item so LangGraph's operator.add reducer appends it once without duplication.
    """
    logger.info("Reflection Agent: Reviewing execution and diff integrity...")

    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    output = test_results.get("output", "")
    retry_count = state.get("retry_count", 0)
    modified_files = state.get("modified_files", [])
    next_retry = retry_count + 1

    if not modified_files:
        latest_err = (state.get("errors") or ["Coding produced no valid file modifications."])[-1]
        return {
            "retry_count": next_retry,
            "errors": [_summarize_coding_guidance(next_retry, latest_err)],
        }

    if passed:
        violations = audit_diff_regressions(
            state.get("repo_path", "."), modified_files, state.get("task", "")
        )
        if violations:
            violation_lines = "; ".join(violations)
            feedback = (
                f"Retry {next_retry} Guidance: Pytest passed, but diff audit flagged regressions "
                f"({violation_lines}). Restore those exact symbols/defaults while keeping your changes."
            )
            return {"retry_count": next_retry, "errors": [feedback]}

        source_files = ", ".join(modified_files)
        feedback = (
            f"Retry {next_retry} Guidance: Tests passed, but no test files were modified alongside "
            f"{source_files}. Add unit tests in tests/test_<module>.py."
        )
        return {"retry_count": next_retry, "errors": [feedback]}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior software engineer diagnosing a failing test run.\n"
                "Rules you MUST follow:\n"
                "1. State ONLY the root-cause diagnosis, the exact file path and line numbers from the traceback, "
                "and the behavioral requirement that failed.\n"
                "2. NEVER write replacement code, code blocks, or speculative implementations.\n\n"
                "{format_instructions}"
            ),
        ),
        ("human", "Task: {task}\n\nTest Output:\n{test_output}"),
    ])

    try:
        result_obj = invoke_structured_groq(
            prompt=prompt,
            payload={
                "task": state["task"],
                "test_output": output[-2000:],
            },
            schema=ReflectionOutput,
            temperature=0.1,
        )
        return {
            "retry_count": next_retry,
            "errors": [f"Retry {next_retry} Diagnosis: {result_obj.feedback}"],
        }
    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        return {
            "retry_count": next_retry,
            "errors": [f"Reflection Error: {str(e)}"],
        }