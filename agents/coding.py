import os
import re
import logging
from typing import List, Set
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
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


class FileModification(BaseModel):
    file_path: str = Field(description="Relative path to the file being modified.")
    new_content: str = Field(description="The exact new content to write to the file.")


class CodingOutput(BaseModel):
    explanation: str = Field(
        default="",
        description="Brief explanation of the changes made.",
    )
    modifications: List[FileModification] = Field(
        description="List of files to modify with their new code."
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


def _discover_matching_test_files(
    repo_path: str,
    repo_files: List[str],
    retrieved_chunks: list,
    plan: List[str],
    task: str,
) -> List[tuple[str, str]]:
    """
    Finds existing test files (e.g. tests/test_<module>.py) matching referenced source
    modules even if Qdrant did not rank them in top-K.
    """
    already_loaded: Set[str] = {
        chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        for chunk in retrieved_chunks
    }

    candidate_stems: Set[str] = set()
    for path in already_loaded:
        base = os.path.basename(path)
        if base.endswith(".py") and not base.startswith("test_") and base != "__init__.py":
            candidate_stems.add(base[:-3])

    for match in re.findall(r"([a-zA-Z0-9_]+)\.py", "\n".join(plan) + "\n" + task):
        if not match.startswith("test_") and match != "__init__":
            candidate_stems.add(match)

    discovered: List[tuple[str, str]] = []
    for rel_path in repo_files:
        if rel_path in already_loaded:
            continue
        base = os.path.basename(rel_path)
        if any(base in (f"test_{stem}.py", f"{stem}_test.py") for stem in candidate_stems):
            full_path = os.path.join(repo_path, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    discovered.append((rel_path, f.read()))
                already_loaded.add(rel_path)
            except Exception as e:
                logger.warning(f"Could not read test file {rel_path}: {e}")
        if len(discovered) >= 4:
            break

    return discovered


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates code patches based on the plan and retrieved context,
    and writes them to the filesystem.
    """
    logger.info("Coding Agent: Generating code modifications...")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    parser = JsonOutputParser(pydantic_object=CodingOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert autonomous software engineer. "
                "Follow the provided plan to fix the user's issue. "
                "Use the Repository File Tree and Retrieved Context to ensure your changes match exact existing file paths, "
                "imports, and function signatures.\n"
                "Test Hygiene & Coverage Rules:\n"
                "- Prefer adding new tests to existing test files matching the modified module (e.g., `tests/test_<module>.py`). "
                "Only create a new test file if no corresponding module test file exists in the Repository File Tree.\n"
                "- Whenever you modify or add production code, always include corresponding unit tests in `modifications`.\n"
                "Output the exact relative file paths and the FULL updated code for those files.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task: {task}\n\nRepository File Tree:\n{file_tree}\n\nPlan: {plan}\n\n"
            "Retrieved Context:\n{context}\n\nPrevious Errors/Feedback:\n{errors}",
        ),
    ])

    chain = prompt | llm | parser

    repo_path = state.get("repo_path", ".")
    repo_files = _get_repo_files(repo_path)
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    retrieved_chunks = state.get("retrieved_context", [])
    context_parts = [
        f"--- File: {chunk.get('file_path')} ---\n{chunk.get('content')}"
        for chunk in retrieved_chunks
    ]

    for test_path, test_content in _discover_matching_test_files(
        repo_path, repo_files, retrieved_chunks, state.get("plan", []), state["task"]
    ):
        logger.info(f"Auto-included matching existing test file in coding context: {test_path}")
        context_parts.append(f"--- Existing Test File: {test_path} ---\n{test_content}")

    context_str = "\n\n".join(context_parts)
    errors_list = list(state.get("errors", []))

    result = None
    last_error = None

    for attempt in range(2):
        try:
            result_dict = chain.invoke({
                "task": state["task"],
                "file_tree": file_tree_str,
                "plan": "\n".join(state.get("plan", [])),
                "context": context_str,
                "errors": "\n".join(errors_list),
                "format_instructions": parser.get_format_instructions(),
            })
            result = CodingOutput(**result_dict)
            break
        except (ValidationError, ValueError, TypeError) as e:
            last_error = e
            logger.warning(f"CodingOutput validation failed on attempt {attempt + 1}: {e}")
            errors_list.append(
                f"Previous JSON output failed schema validation: {e}. "
                "Return valid JSON matching the exact schema with 'explanation' and 'modifications'."
            )
        except Exception as e:
            logger.error(f"Coding failed: {str(e)}")
            return {"errors": [f"Coding Error: {str(e)}"]}

    if result is None:
        logger.error(f"Coding failed after retries: {last_error}")
        return {"errors": [f"Coding Error: {str(last_error)}"]}

    try:
        modified_files = []
        for mod in result.modifications:
            full_path = os.path.join(repo_path, mod.file_path)
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(mod.new_content)
            modified_files.append(mod.file_path)
            logger.info(f"Successfully modified: {full_path}")

        return {"modified_files": modified_files}

    except Exception as e:
        logger.error(f"Writing modified files failed: {str(e)}")
        return {"errors": [f"Coding Error: {str(e)}"]}