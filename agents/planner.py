import os
import re
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from agents.llm_utils import invoke_structured_groq
from agents.state import AgentState

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
}


class PlanOutput(BaseModel):
    steps: List[str] = Field(
        description="Ordered, atomic engineering steps grounded in exact repository file paths and symbols."
    )


def _get_repo_files(repo_path: str, limit: int = 150) -> List[str]:
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


def _normalize_flat_steps(raw_steps: List[str]) -> List[str]:
    """Flattens any nested bullets into clean sequential steps (1., 2., 3., ...)."""
    flat: List[str] = []
    for item in raw_steps:
        for line in item.splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)", "", line).strip()
            if cleaned:
                flat.append(cleaned)
    return [f"{idx}. {text}" for idx, text in enumerate(flat, start=1)]


def plan_node(state: AgentState) -> dict:
    """LangGraph node: Generates a codebase-grounded, flat numbered execution plan."""
    logger.info("Planner Agent: Generating grounded execution plan...")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior software architect creating a concise execution plan for an autonomous coding agent.\n"
                "Rules you MUST follow:\n"
                "1. CODEBASE GROUNDING: Reference ONLY exact file paths, function names, class names, and constants "
                "present in the Repository File Tree and Retrieved Context. Never guess or invent symbol names.\n"
                "2. ATOMIC AGENT ACTIONS ONLY: Include only concrete code edits and `pytest` verification. "
                "NEVER include manual human sanity checks, manual app runs, or git commit/push steps.\n"
                "3. FLAT NUMBERED STEPS: Return 3 to 6 concise, single-sentence steps with NO nested sub-bullets.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task:\n{task}\n\nRepository File Tree:\n{file_tree}\n\nRetrieved Context:\n{context}",
        ),
    ])

    repo_path = state.get("repo_path", ".")
    repo_files = _get_repo_files(repo_path)
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    retrieved = state.get("retrieved_context", [])
    context_str = "\n\n".join([
        f"--- {c.get('file_path')} ---\n{c.get('content', '')[:1200]}"
        for c in retrieved[:4]
    ]) or "(none)"

    try:
        plan_obj = invoke_structured_groq(
            prompt=prompt,
            payload={
                "task": state["task"],
                "file_tree": file_tree_str,
                "context": context_str,
            },
            schema=PlanOutput,
            temperature=0.1,
        )
        steps = _normalize_flat_steps(plan_obj.steps)
        return {"plan": steps, "status": "planned"}
    except Exception as e:
        logger.error(f"Planning failed: {str(e)}")
        return {"status": "error", "errors": [f"Planner Error: {str(e)}"]}