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


def _is_test_file(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("./")
    base = os.path.basename(norm)
    return (
        norm.startswith("tests/")
        or "/tests/" in norm
        or base.startswith("test_")
        or base.endswith("_test.py")
    )


def _is_test_only_task(task: str) -> bool:
    return bool(
        re.search(
            r"^\s*(?:add|write|create|update)\s+(?:unit\s+)?tests?\b|\btests?\s+only\b",
            task.strip(),
            re.IGNORECASE,
        )
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


def _collect_full_target_files(
    repo_path: str,
    repo_files: List[str],
    retrieved_chunks: list,
    plan: List[str],
    task: str,
) -> dict[str, str]:
    """
    Resolves all target source and matching test files and reads their COMPLETE
    contents directly from disk so the LLM never reconstructs missing lines.
    """
    target_paths: Set[str] = set()
    plan_and_task = "\n".join(plan) + "\n" + task

    # 1. Any file surfaced by retrieval
    for chunk in retrieved_chunks:
        fp = chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        if fp in repo_files:
            target_paths.add(fp)

    # 2. Any repo file explicitly mentioned by path or module filename in plan/task
    mentioned_names = set(re.findall(r"([a-zA-Z0-9_./-]+\.py)", plan_and_task))
    for rel_path in repo_files:
        base = os.path.basename(rel_path)
        if rel_path in plan_and_task or base in mentioned_names:
            target_paths.add(rel_path)

    # 3. Matching test files (e.g. tests/test_<module>.py) for every candidate source module
    stems: Set[str] = set()
    for path in list(target_paths):
        base = os.path.basename(path)
        if base.endswith(".py") and not base.startswith("test_") and base != "__init__.py":
            stems.add(base[:-3])

    for rel_path in repo_files:
        base = os.path.basename(rel_path)
        if any(base in (f"test_{stem}.py", f"{stem}_test.py") for stem in stems):
            target_paths.add(rel_path)

    full_files: dict[str, str] = {}
    for rel_path in sorted(target_paths):
        full_path = os.path.join(repo_path, rel_path)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    full_files[rel_path] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read full file {rel_path}: {e}")

    return full_files


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Loads complete target files from disk, generates minimal code patches,
    and writes them to the filesystem.
    """
    logger.info("Coding Agent: Loading full target files and generating modifications...")

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
                "You are an expert autonomous software engineer.\n"
                "Rules you MUST follow:\n"
                "1. COMPLETE FILE INTEGRITY: Existing files to edit are provided in full under "
                "'Complete Target Files From Disk'. Preserve every existing class attribute, dataclass field, "
                "default value, import, function, and existing test character-for-character unless the task "
                "explicitly asks to change or remove it.\n"
                "2. NEVER OVERWRITE PARTIAL FILES: Do not modify an existing file unless its full content was "
                "provided in 'Complete Target Files From Disk'.\n"
                "3. COMPLETE IMPLEMENTATION: For bug-fix and feature tasks, always output BOTH the source code "
                "changes AND the corresponding unit tests in `modifications`.\n"
                "4. TEST HYGIENE: Prefer adding tests to the matching existing module test file "
                "(e.g., `tests/test_<module>.py`) rather than creating new test files.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task: {task}\n\nRepository File Tree:\n{file_tree}\n\nPlan:\n{plan}\n\n"
            "Complete Target Files From Disk:\n{full_files}\n\n"
            "Additional Reference Snippets:\n{snippets}\n\nPrevious Errors/Feedback:\n{errors}",
        ),
    ])

    chain = prompt | llm | parser

    repo_path = state.get("repo_path", ".")
    repo_files = _get_repo_files(repo_path)
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    retrieved_chunks = state.get("retrieved_context", [])
    full_files_map = _collect_full_target_files(
        repo_path, repo_files, retrieved_chunks, state.get("plan", []), state["task"]
    )

    full_files_str = "\n\n".join([
        f"=== Complete File: {path} ===\n{content}"
        for path, content in full_files_map.items()
    ]) or "(none)"

    snippets_str = "\n\n".join([
        f"--- Snippet: {chunk.get('file_path')} ---\n{chunk.get('content')}"
        for chunk in retrieved_chunks
        if chunk.get("file_path", "").replace("\\", "/").lstrip("./") not in full_files_map
    ]) or "(none)"

    errors_list = list(state.get("errors", []))
    require_source_edit = not _is_test_only_task(state["task"])

    result = None
    last_error = None

    for attempt in range(3):
        try:
            result_dict = chain.invoke({
                "task": state["task"],
                "file_tree": file_tree_str,
                "plan": "\n".join(state.get("plan", [])),
                "full_files": full_files_str,
                "snippets": snippets_str,
                "errors": "\n".join(errors_list),
                "format_instructions": parser.get_format_instructions(),
            })
            candidate = CodingOutput(**result_dict)

            # Reject test-only first attempts on implementation tasks before wasting a pytest cycle
            if (
                require_source_edit
                and attempt < 2
                and candidate.modifications
                and all(_is_test_file(m.file_path) for m in candidate.modifications)
            ):
                logger.warning(
                    f"Coding attempt {attempt + 1} only modified test files; re-prompting for source edits."
                )
                errors_list.append(
                    "Your previous output only modified test files and omitted the required source code "
                    "implementation. Include BOTH the source file edits (e.g., under src/) and the test "
                    "file updates in `modifications`."
                )
                continue

            result = candidate
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
            norm_path = mod.file_path.replace("\\", "/").lstrip("./")
            full_path = os.path.join(repo_path, norm_path)

            # Guard against overwriting an existing file that was never loaded in full from disk
            if os.path.exists(full_path) and norm_path not in full_files_map:
                with open(full_path, "r", encoding="utf-8") as f:
                    full_files_map[norm_path] = f.read()
                logger.warning(
                    f"Blocked blind overwrite of unloaded existing file {norm_path}; loaded full file for retry."
                )
                return {
                    "errors": [
                        f"Coding Error: Attempted to overwrite existing file '{norm_path}' without reading "
                        "its full contents first. Retry with full file context."
                    ]
                }

            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(mod.new_content)
            modified_files.append(norm_path)
            logger.info(f"Successfully modified: {full_path}")

        return {"modified_files": modified_files}

    except Exception as e:
        logger.error(f"Writing modified files failed: {str(e)}")
        return {"errors": [f"Coding Error: {str(e)}"]}