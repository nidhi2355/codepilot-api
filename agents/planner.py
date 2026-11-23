import os
import re
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from agents.state import AgentState

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".ruff_cache",
}


class PlanOutput(BaseModel):
    steps: List[str] = Field(
        description="A flat, strictly sequential list of concrete code-edit and pytest steps."
    )


def _get_repo_files(repo_path: str, limit: int = 200) -> List[str]:
    files: List[str] = []
    if not repo_path or not os.path.exists(repo_path):
        return files
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(root, fname), repo_path).replace("\\", "/")
            files.append(rel)
            if len(files) >= limit:
                return sorted(files)
    return sorted(files)


def plan_node(state: AgentState) -> dict:
    """
    LangGraph node: Converts a user request into a grounded, flat, actionable engineering plan.
    """
    logger.info("Planner Agent: Generating grounded execution plan...")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    repo_files = _get_repo_files(state.get("repo_path", "."))
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    context_str = "\n\n".join([
        f"--- File: {chunk.get('file_path')} ---\n{chunk.get('content')}"
        for chunk in state.get("retrieved_context", [])
    ]) or "(no retrieved chunks)"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert software architect creating an execution plan for an automated coding agent.\n"
                "Rules you MUST follow:\n"
                "1. GROUNDING: Cite ONLY exact file paths, class names, function names, and constants present in the "
                "Repository File Tree and Retrieved Codebase Context. Never guess paths or use hedging phrases like "
                "'typically in...' or 'functions like...'.\n"
                "2. ACTIONABLE STEPS ONLY: Include ONLY concrete code edits to specific files and automated `pytest` runs. "
                "NEVER include human/manual steps (e.g., 'manual sanity check', 'code review', 'inspect') and NEVER include "
                "Git/PR steps (e.g., 'commit', 'push', 'create branch', 'open PR')—those are automated.\n"
                "3. TEST COVERAGE: Always include a step to add or update unit tests in the matching existing test file "
                "(e.g., `tests/test_<module>.py`) if one exists in the file tree.\n"
                "4. FLAT LIST: Return a flat list of single-line steps with NO nested sub-bullets, NO sub-lists, and NO actual code."
            ),
        ),
        (
            "human",
            "Task: {task}\n\nRepository File Tree:\n{file_tree}\n\nRetrieved Codebase Context:\n{context}",
        ),
    ])

    structured_llm = llm.with_structured_output(PlanOutput)
    chain = prompt | structured_llm

    try:
        result = chain.invoke({
            "task": state["task"],
            "file_tree": file_tree_str,
            "context": context_str,
        })

        # Normalize numbering to guarantee strictly sequential 1..N formatting
        cleaned_steps = []
        for idx, raw_step in enumerate(result.steps, start=1):
            single_line = " ".join(raw_step.strip().splitlines())
            stripped = re.sub(r"^\d+[\.\)]\s*", "", single_line).strip()
            if stripped:
                cleaned_steps.append(f"{idx}. {stripped}")

        logger.info(f"Generated plan with {len(cleaned_steps)} steps.")
        return {"plan": cleaned_steps}
    except Exception as e:
        logger.error(f"Planning failed: {str(e)}")
        return {"plan": [], "errors": [f"Planner Error: {str(e)}"]}