import ast
import os
import re
import time
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

MAX_FULL_FILES_CHARS = 11000
MAX_SNIPPETS_CHARS = 2500
MAX_ERRORS_CHARS = 1500


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


def _collect_prioritized_files(
    repo_path: str,
    repo_files: List[str],
    retrieved_chunks: list,
    plan: List[str],
    task: str,
) -> dict[str, str]:
    """
    Loads full contents of directly relevant source files and their matching test file
    within a strict character budget to stay safely under Groq's 8,000 TPM limit.
    """
    ordered_candidates: List[str] = []
    seen: Set[str] = set()

    def _add(rel_path: str):
        if rel_path in repo_files and rel_path not in seen:
            seen.add(rel_path)
            ordered_candidates.append(rel_path)

    # 1. Explicit .py paths mentioned in task or plan (e.g. src/shopcore/pricing.py)
    plan_and_task = task + "\n" + "\n".join(plan)
    mentioned_py = set(re.findall(r"[\w./-]+\.py", plan_and_task))
    for rel_path in repo_files:
        if rel_path in mentioned_py or os.path.basename(rel_path) in mentioned_py:
            _add(rel_path)

    # 2. Top retrieved chunks from vector search
    for chunk in retrieved_chunks[:4]:
        fp = chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        _add(fp)

    # 3. Matching test file for non-test candidate modules (e.g. pricing.py -> tests/test_pricing.py)
    stems: List[str] = []
    for path in list(ordered_candidates):
        base = os.path.basename(path)
        if base.endswith(".py") and not _is_test_file(path) and base != "__init__.py":
            stems.append(base[:-3])

    for stem in stems:
        for rel_path in repo_files:
            base = os.path.basename(rel_path)
            if base in (f"test_{stem}.py", f"test_{stem}s.py", f"{stem}_test.py"):
                _add(rel_path)

    full_files: dict[str, str] = {}
    total_chars = 0

    for rel_path in ordered_candidates:
        full_path = os.path.join(repo_path, rel_path)
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read full file {rel_path}: {e}")
            continue

        if total_chars + len(content) > MAX_FULL_FILES_CHARS and full_files:
            continue
        full_files[rel_path] = content
        total_chars += len(content)

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


def _invoke_with_rate_limit_retry(chain, payload: dict, max_429_retries: int = 3):
    """Handles Groq 429 TPM rate limits by waiting the requested retry-after window."""
    for attempt in range(max_429_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as err:
            msg = str(err)
            is_rate_limit = "429" in msg or "rate_limit_exceeded" in msg
            if not is_rate_limit or attempt >= max_429_retries:
                raise
            wait_match = re.search(r"try again in\s+([0-9.]+)s", msg, re.IGNORECASE)
            wait_secs = min(float(wait_match.group(1)) + 1.0, 30.0) if wait_match else 12.0
            logger.warning(
                f"Groq TPM rate limit hit; waiting {wait_secs:.1f}s before retry ({attempt + 1}/{max_429_retries})..."
            )
            time.sleep(wait_secs)


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates surgical search-and-replace patches within Groq's TPM budget,
    validates syntax, AST symbol preservation, and test coverage, and writes to disk.
    """
    logger.info("Coding Agent: Loading target files and generating surgical edits...")

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
                "1. SURGICAL SEARCH-AND-REPLACE ONLY: For existing files under 'Complete Target Files From Disk', "
                "output ONLY small `edits` blocks with exact `target_code` (copied character-for-character) "
                "and `replacement_code`. NEVER rewrite entire existing files and NEVER delete existing methods, "
                "functions, attributes, or default values.\n"
                "2. PLACE LOGIC IN THE RIGHT MODULE: Put module-level constants/functions in the target module "
                "(e.g. `src/shopcore/pricing.py`) rather than inlining them into caller functions.\n"
                "3. BRAND-NEW FILES ONLY: Leave `target_code` empty (\"\") ONLY when creating a new file not in the tree.\n"
                "4. SOURCE + TESTS TOGETHER: Always modify BOTH the target implementation file(s) AND the matching "
                "test file (e.g. `tests/test_<module>.py`) in the same response.\n\n"
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
    full_files_map = _collect_prioritized_files(
        repo_path, repo_files, retrieved_chunks, state.get("plan", []), state["task"]
    )

    full_files_str = "\n\n".join([
        f"=== Complete File: {path} ===\n{content}"
        for path, content in full_files_map.items()
    ]) or "(none)"

    snippets_parts: list[str] = []
    snippets_len = 0
    for chunk in retrieved_chunks:
        fp = chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        if fp in full_files_map:
            continue
        entry = f"--- Snippet: {fp} ---\n{chunk.get('content', '')}"
        if snippets_len + len(entry) > MAX_SNIPPETS_CHARS:
            continue
        snippets_parts.append(entry)
        snippets_len += len(entry)
    snippets_str = "\n\n".join(snippets_parts) or "(none)"

    recent_errors = "\n".join(state.get("errors", [])[-2:])
    if len(recent_errors) > MAX_ERRORS_CHARS:
        recent_errors = recent_errors[-MAX_ERRORS_CHARS:]

    require_source_edit = not _is_test_only_task(state["task"])

    try:
        result_dict = _invoke_with_rate_limit_retry(
            chain,
            {
                "task": state["task"],
                "file_tree": file_tree_str,
                "plan": "\n".join(state.get("plan", [])),
                "full_files": full_files_str,
                "snippets": snippets_str,
                "errors": recent_errors or "(none)",
                "format_instructions": parser.get_format_instructions(),
            },
        )
        candidate = CodingOutput(**result_dict)

        if not candidate.modifications:
            return {
                "modified_files": [],
                "errors": ["Coding Error: Returned empty modifications list; provide required file edits."],
            }

        grouped_edits: dict[str, List[SearchReplaceBlock]] = {}
        for mod in candidate.modifications:
            norm_path = mod.file_path.replace("\\", "/").lstrip("./")
            grouped_edits.setdefault(norm_path, []).extend(mod.edits)

        static_errors: list[str] = []
        candidate_contents: dict[str, str] = {}

        for norm_path, edits in grouped_edits.items():
            full_path = os.path.join(repo_path, norm_path)
            if os.path.exists(full_path):
                if norm_path not in full_files_map:
                    with open(full_path, "r", encoding="utf-8") as f:
                        full_files_map[norm_path] = f.read()

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
            combined_msg = "Coding validation failed:\n" + "\n".join(f"- {e}" for e in static_errors)
            logger.warning(combined_msg)
            return {"modified_files": [], "errors": [f"Coding Error: {combined_msg}"]}

        if os.path.exists(os.path.join(repo_path, ".git")):
            subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)

        modified_files: list[str] = []
        for norm_path, content in candidate_contents.items():
            full_path = os.path.join(repo_path, norm_path)
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            modified_files.append(norm_path)
            logger.info(f"Successfully wrote verified edits to: {full_path}")

        return {"modified_files": modified_files}

    except (ValidationError, ValueError, TypeError) as e:
        logger.warning(f"CodingOutput validation failed: {e}")
        return {"modified_files": [], "errors": [f"Coding Error: Invalid JSON schema ({e})"]}
    except Exception as e:
        logger.error(f"Coding failed: {str(e)}")
        return {"modified_files": [], "errors": [f"Coding Error: {str(e)}"]}