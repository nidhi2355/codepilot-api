import ast
import os
import re
import logging
import subprocess
from typing import List, Set
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.ast_guard import check_source_regressions
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


class SearchReplaceBlock(BaseModel):
    target_code: str = Field(
        default="",
        description=(
            "Exact contiguous snippet from the existing file to replace. "
            "Must match character-for-character. Leave empty ONLY when creating a brand-new file."
        ),
    )
    replacement_code: str = Field(
        description="Replacement code for target_code, or the full content when creating a new file."
    )


class FileModification(BaseModel):
    file_path: str = Field(description="Relative path to the file being modified or created.")
    edits: List[SearchReplaceBlock] = Field(
        description="Surgical search-and-replace blocks for this file."
    )


class CodingOutput(BaseModel):
    explanation: str = Field(
        default="",
        description="Brief summary of the changes made.",
    )
    modifications: List[FileModification] = Field(
        description="List of files to modify or create."
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
    """Loads full contents of all candidate source files and their matching test files."""
    target_paths: Set[str] = set()
    plan_and_task = "\n".join(plan) + "\n" + task

    for chunk in retrieved_chunks:
        fp = chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        if fp in repo_files:
            target_paths.add(fp)

    mentioned_names = set(re.findall(r"([a-zA-Z0-9_./-]+\.py)", plan_and_task))
    for rel_path in repo_files:
        base = os.path.basename(rel_path)
        if rel_path in plan_and_task or base in mentioned_names:
            target_paths.add(rel_path)

    # Auto-discover matching existing test files (e.g. order_service.py -> tests/test_order_service.py)
    stems: Set[str] = set()
    for path in list(target_paths):
        base = os.path.basename(path)
        if base.endswith(".py") and not base.startswith("test_") and base != "__init__.py":
            stem = base[:-3]
            stems.add(stem)
            if "_" in stem:
                stems.add(stem.split("_")[0])

    for rel_path in repo_files:
        base = os.path.basename(rel_path)
        if any(
            base in (f"test_{stem}.py", f"test_{stem}s.py", f"{stem}_test.py")
            for stem in stems
        ):
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


def _apply_search_replace(
    rel_path: str, original_content: str, edits: List[SearchReplaceBlock]
) -> tuple[str, list[str]]:
    """Applies search-and-replace blocks in memory and returns (new_content, errors)."""
    content = original_content
    errors: list[str] = []

    for idx, block in enumerate(edits, start=1):
        target = block.target_code
        replacement = block.replacement_code

        if not target.strip():
            errors.append(
                f"Edit #{idx} for existing file '{rel_path}' had an empty 'target_code'. "
                "Provide the exact existing snippet to replace rather than rewriting the whole file."
            )
            continue

        count = content.count(target)
        if count == 1:
            content = content.replace(target, replacement, 1)
            continue
        if count > 1:
            errors.append(
                f"Edit #{idx} in '{rel_path}' matched {count} places; include more surrounding context lines."
            )
            continue

        # Fallback: match ignoring trailing whitespace on each line
        target_lines = [line.rstrip() for line in target.splitlines()]
        src_lines = content.splitlines()
        n = len(target_lines)
        matched_indices = [
            i
            for i in range(len(src_lines) - n + 1)
            if [l.rstrip() for l in src_lines[i : i + n]] == target_lines
        ]
        if len(matched_indices) == 1:
            start = matched_indices[0]
            updated_lines = src_lines[:start] + replacement.splitlines() + src_lines[start + n :]
            trailing_nl = "\n" if content.endswith("\n") else ""
            content = "\n".join(updated_lines) + trailing_nl
        else:
            errors.append(
                f"Edit #{idx} 'target_code' did not match '{rel_path}'. "
                "Copy the exact lines character-for-character from Complete Target Files From Disk."
            )

    return content, errors


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates surgical search-and-replace patches, validates syntax,
    AST symbol preservation, and test coverage in a single pass, and writes to disk.
    """
    logger.info("Coding Agent: Loading full target files and generating surgical edits...")

    base_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )
    llm = base_llm.bind(tool_choice="none")
    parser = JsonOutputParser(pydantic_object=CodingOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert autonomous software engineer.\n"
                "Rules you MUST follow:\n"
                "1. SURGICAL SEARCH-AND-REPLACE ONLY: For existing files shown under 'Complete Target Files From Disk', "
                "output ONLY small `edits` blocks with exact `target_code` (copied character-for-character from the file) "
                "and `replacement_code`. NEVER rewrite entire existing files and NEVER delete existing methods, functions, "
                "attributes, or default values.\n"
                "2. BRAND-NEW FILES ONLY: Leave `target_code` empty (\"\") ONLY when creating a new file that does not "
                "exist in the Repository File Tree.\n"
                "3. SOURCE + TESTS TOGETHER: For bug-fix and feature tasks, always modify BOTH the implementation file "
                "(e.g., `src/...`) AND its matching existing test file (e.g., `tests/test_<module>.py`) in the same response.\n\n"
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

    # Keep only the latest 3 feedback entries to prevent prompt bloat across outer retries
    local_errors = list(state.get("errors", [])[-3:])
    require_source_edit = not _is_test_only_task(state["task"])

    validated_outputs: dict[str, str] | None = None
    last_validation_errors: list[str] = []

    for attempt in range(3):
        try:
            result_dict = chain.invoke({
                "task": state["task"],
                "file_tree": file_tree_str,
                "plan": "\n".join(state.get("plan", [])),
                "full_files": full_files_str,
                "snippets": snippets_str,
                "errors": "\n".join(local_errors),
                "format_instructions": parser.get_format_instructions(),
            })
            candidate = CodingOutput(**result_dict)

            if not candidate.modifications:
                last_validation_errors = ["Returned empty modifications list; provide required file edits."]
                local_errors.append(last_validation_errors[0])
                continue

            # Group edits by normalized path
            grouped_edits: dict[str, List[SearchReplaceBlock]] = {}
            for mod in candidate.modifications:
                norm_path = mod.file_path.replace("\\", "/").lstrip("./")
                grouped_edits.setdefault(norm_path, []).extend(mod.edits)

            # Single-pass bundled static validation (patch match + syntax + AST regressions + test coverage)
            static_errors: list[str] = []
            candidate_contents: dict[str, str] = {}

            for norm_path, edits in grouped_edits.items():
                full_path = os.path.join(repo_path, norm_path)
                if os.path.exists(full_path):
                    if norm_path not in full_files_map:
                        with open(full_path, "r", encoding="utf-8") as f:
                            full_files_map[norm_path] = f.read()
                        full_files_str += f"\n\n=== Complete File: {norm_path} ===\n{full_files_map[norm_path]}"
                        static_errors.append(
                            f"Loaded '{norm_path}' into context; re-emit surgical edits using its exact lines."
                        )
                        continue

                    new_src, patch_errs = _apply_search_replace(
                        norm_path, full_files_map[norm_path], edits
                    )
                    static_errors.extend(patch_errs)
                    if not patch_errs:
                        if norm_path.endswith(".py"):
                            static_errors.extend(
                                check_source_regressions(
                                    norm_path, full_files_map[norm_path], new_src, state["task"]
                                )
                            )
                        candidate_contents[norm_path] = new_src
                else:
                    new_src = "\n".join(b.replacement_code for b in edits)
                    if norm_path.endswith(".py"):
                        try:
                            ast.parse(new_src)
                        except SyntaxError as syn_err:
                            static_errors.append(
                                f"SyntaxError in new file '{norm_path}' at line {syn_err.lineno}: {syn_err.msg}"
                            )
                    candidate_contents[norm_path] = new_src

            paths = list(grouped_edits.keys())
            has_test = any(_is_test_file(p) for p in paths)
            has_source = any(not _is_test_file(p) for p in paths)

            if require_source_edit and not has_source:
                static_errors.append(
                    "Missing source code changes: include edits to the target implementation file(s) alongside tests."
                )
            if not has_test:
                static_errors.append(
                    "Missing unit tests: add or update tests in the matching existing test file (e.g., tests/test_<module>.py)."
                )

            if static_errors:
                last_validation_errors = static_errors
                combined_msg = "Attempt validation failed—fix ALL of the following together:\n" + "\n".join(
                    f"- {err}" for err in static_errors
                )
                logger.warning(f"Coding static validation failed (internal attempt {attempt + 1}/3): {combined_msg}")
                local_errors.append(combined_msg)
                continue

            validated_outputs = candidate_contents
            break

        except (ValidationError, ValueError, TypeError) as e:
            last_validation_errors = [f"Schema validation error: {e}"]
            logger.warning(f"CodingOutput validation failed on attempt {attempt + 1}/3: {e}")
            local_errors.append(
                f"JSON output failed schema validation: {e}. Return valid JSON with 'explanation' and 'modifications'."
            )
        except Exception as e:
            logger.error(f"Coding failed: {str(e)}")
            return {"modified_files": [], "errors": [f"Coding Error: {str(e)}"]}

    if not validated_outputs:
        summary = "; ".join(last_validation_errors) or "Unable to produce valid code modifications."
        logger.error(f"Coding failed after internal retries: {summary}")
        return {"modified_files": [], "errors": [f"Coding Error: {summary}"]}

    try:
        # Reset tracked files to clean HEAD before applying verified edits so stale partial edits never linger
        if os.path.exists(os.path.join(repo_path, ".git")):
            subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)

        modified_files: list[str] = []
        for norm_path, content in validated_outputs.items():
            full_path = os.path.join(repo_path, norm_path)
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            modified_files.append(norm_path)
            logger.info(f"Successfully wrote verified edits to: {full_path}")

        return {"modified_files": modified_files}

    except Exception as e:
        logger.error(f"Writing modified files failed: {str(e)}")
        return {"modified_files": [], "errors": [f"Coding Error: {str(e)}"]}