import os
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.ast_guard import audit_diff_regressions
from agents.state import AgentState

logger = logging.getLogger(__name__)


class ReflectionOutput(BaseModel):
    feedback: str = Field(
        description="Root-cause diagnosis with exact file/line numbers and behavioral requirements, without code blocks."
    )


def reflection_node(state: AgentState) -> dict:
    """
    LangGraph node: Diagnoses coding/test/diff failures and returns ONLY the new
    feedback item so LangGraph's operator.add reducer appends it once.
    """
    logger.info("Reflection Agent: Reviewing execution and diff integrity...")

    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    output = test_results.get("output", "")
    retry_count = state.get("retry_count", 0)
    modified_files = state.get("modified_files", [])

    # Case 1: Coding failed before running tests (empty modified_files)
    if not modified_files:
        latest_err = (state.get("errors") or ["Coding produced no valid file modifications."])[-1]
        new_feedback = f"Retry {retry_count + 1} Feedback: {latest_err}"
        return {"retry_count": retry_count + 1, "errors": [new_feedback]}

    # Case 2: Pytest passed, but diff audit or test coverage gate rejected the patch
    if passed:
        violations = audit_diff_regressions(
            state.get("repo_path", "."), modified_files, state.get("task", "")
        )
        if violations:
            violation_lines = "\n".join(f"- {v}" for v in violations)
            feedback = (
                "Pytest passed, but the diff audit detected unintended deletions or altered defaults:\n"
                f"{violation_lines}\n"
                "Restore these exact attributes, default values, and functions while keeping your new changes."
            )
            logger.info(f"Reflection flagged diff regressions: {violations}")
            return {
                "retry_count": retry_count + 1,
                "errors": [f"Retry {retry_count + 1} Feedback: {feedback}"],
            }

        source_files = ", ".join(modified_files)
        feedback = (
            f"Tests passed, but no test files were modified alongside {source_files}. "
            "Add unit tests covering your changes in the matching existing test file "
            "(e.g., `tests/test_<module>.py`)."
        )
        return {
            "retry_count": retry_count + 1,
            "errors": [f"Retry {retry_count + 1} Feedback: {feedback}"],
        }

    # Case 3: Pytest failed -> diagnose root cause without writing speculative code
    base_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )
    llm = base_llm.bind(tool_choice="none")
    parser = JsonOutputParser(pydantic_object=ReflectionOutput)

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

    chain = prompt | llm | parser

    try:
        result_dict = chain.invoke({
            "task": state["task"],
            "test_output": output[-2000:],
            "format_instructions": parser.get_format_instructions(),
        })

        feedback = result_dict.get("feedback", "No specific feedback generated.")
        logger.info(f"Reflection complete. Feedback: {feedback}")
        return {
            "retry_count": retry_count + 1,
            "errors": [f"Retry {retry_count + 1} Feedback: {feedback}"],
        }
    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        return {
            "retry_count": retry_count + 1,
            "errors": [f"Reflection Error: {str(e)}"],
        }