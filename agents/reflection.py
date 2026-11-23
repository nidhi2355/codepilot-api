import os
import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState
from agents.supervisor import audit_diff_regressions

logger = logging.getLogger(__name__)


class ReflectionOutput(BaseModel):
    feedback: str = Field(
        description="Root-cause diagnosis with exact file/line numbers and behavioral requirements, without code blocks."
    )


def reflection_node(state: AgentState) -> dict:
    """
    LangGraph node: Diagnoses test failures, missing test coverage, or unintended
    diff regressions without generating speculative code snippets.
    """
    logger.info("Reflection Agent: Reviewing test execution and diff integrity...")

    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    output = test_results.get("output", "")
    retry_count = state.get("retry_count", 0)
    modified_files = state.get("modified_files", [])
    new_errors = list(state.get("errors", []))

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
            new_errors.append(f"Retry {retry_count + 1} Feedback: {feedback}")
            return {"retry_count": retry_count + 1, "errors": new_errors}

        source_files = ", ".join(modified_files) or "the modified modules"
        feedback = (
            f"Existing tests passed, but no test files were added or updated for {source_files}. "
            "Add unit tests covering your changes in the matching existing test file "
            "(e.g., `tests/test_<module>.py`) and include that test file in `modifications`."
        )
        logger.info(f"Reflection enforced test coverage: {feedback}")
        new_errors.append(f"Retry {retry_count + 1} Feedback: {feedback}")
        return {"retry_count": retry_count + 1, "errors": new_errors}

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )

    parser = JsonOutputParser(pydantic_object=ReflectionOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior software engineer diagnosing a failing test run.\n"
                "Rules you MUST follow:\n"
                "1. State ONLY the root-cause diagnosis, the exact file path and line numbers from the traceback, "
                "and the behavioral requirement that failed.\n"
                "2. NEVER write replacement code, code blocks, or speculative implementations—you do not have the "
                "full source files and must not guess internal data structures.\n\n"
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
        new_errors.append(f"Retry {retry_count + 1} Feedback: {feedback}")

        return {"retry_count": retry_count + 1, "errors": new_errors}
    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        new_errors.append(f"Reflection Error: {str(e)}")
        return {"retry_count": retry_count + 1, "errors": new_errors}