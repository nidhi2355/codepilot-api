import os
import re
import time
import logging
import subprocess
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    steps: List[str] = Field(
        description="A flat, strictly sequential list of concrete code-edit and pytest steps."
    )


def _get_repo_files(repo_path: str, limit: int = 200) -> List[str]:
    if not repo_path or not os.path.exists(repo_path):
        return []
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return sorted(proc.stdout.strip().splitlines())[:limit]

    files: List[str] = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}]
        for fname in filenames:
            files.append(os.path.relpath(os.path.join(root, fname), repo_path).replace("\\", "/"))
            if len(files) >= limit:
                return sorted(files)
    return sorted(files)


def plan_node(state: AgentState) -> dict:
    """
    LangGraph node: Converts a user request into a grounded, flat, actionable engineering plan
    with tool calls disabled and exponential backoff retries.
    """
    logger.info("Planner Agent: Generating grounded execution plan...")

    base_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
    # Explicitly disable tool calling so Groq never fails with 400 tool_use_failed
    llm = base_llm.bind(tool_choice="none")

    parser = JsonOutputParser(pydantic_object=PlanOutput)

    repo_files = _get_repo_files(state.get("repo_path", "."))
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    context_chunks = state.get("retrieved_context", [])[:6]
    context_str = "\n\n".join([
        f"--- File: {chunk.get('file_path')} ---\n{str(chunk.get('content', ''))[:1000]}"
        for chunk in context_chunks
    ]) or "(no retrieved chunks)"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert software architect creating an execution plan for an automated coding agent.\n"
                "Rules you MUST follow:\n"
                "1. NO TOOL CALLS: Do not invoke or emit any tool/function calls. Respond ONLY with the JSON object.\n"
                "2. GROUNDING: Cite ONLY exact file paths, class names, function names, and constants present in the "
                "Repository File Tree and Retrieved Codebase Context. Never guess paths or use hedging phrases.\n"
                "3. ACTIONABLE STEPS ONLY: Include ONLY concrete code edits to specific files and automated `pytest` runs. "
                "NEVER include human/manual steps or Git/PR steps.\n"
                "4. TEST COVERAGE: Always include a step to add or update unit tests in the matching existing test file "
                "(e.g., `tests/test_<module>.py`) if one exists in the file tree.\n"
                "5. FLAT LIST: Return a flat JSON list of single-line strings under `steps` with NO nested sub-bullets "
                "and NO actual code.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task: {task}\n\nRepository File Tree:\n{file_tree}\n\nRetrieved Codebase Context:\n{context}",
        ),
    ])

    chain = prompt | llm | parser

    last_error = None
    for attempt in range(3):
        try:
            result_dict = chain.invoke({
                "task": state["task"],
                "file_tree": file_tree_str,
                "context": context_str,
                "format_instructions": parser.get_format_instructions(),
            })

            raw_steps = result_dict.get("steps", []) if isinstance(result_dict, dict) else []
            if isinstance(raw_steps, str):
                raw_steps = [line for line in raw_steps.splitlines() if line.strip()]

            cleaned_steps = []
            for idx, raw_step in enumerate(raw_steps, start=1):
                single_line = " ".join(str(raw_step).strip().splitlines())
                stripped = re.sub(r"^\d+[\.\)]\s*", "", single_line).strip()
                if stripped:
                    cleaned_steps.append(f"{idx}. {stripped}")

            logger.info(f"Generated plan with {len(cleaned_steps)} steps.")
            return {"plan": cleaned_steps}
        except Exception as e:
            last_error = e
            logger.warning(f"Planner attempt {attempt + 1}/3 failed: {str(e)}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error(f"Planning failed after 3 attempts: {str(last_error)}")
    return {"plan": [], "errors": [f"Planner Error: {str(last_error)}"]}