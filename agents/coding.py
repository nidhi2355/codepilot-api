import ast
import os
import re
import logging
import subprocess
import textwrap
from typing import List, Set
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from agents.ast_guard import check_source_regressions
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
    ".mypy_cache",
    ".ruff_cache",
}

MAX_SOURCE_FILES_CHARS = 8500
MAX_TEST_FILES_CHARS = 4500
MAX_SNIPPETS_CHARS = 2000
MAX_ERRORS_CHARS = 1500


class SearchReplaceBlock(BaseModel):
    target_code: str = Field(
        default="",
        description=(
            "Exact contiguous snippet from the existing file to replace. "
            "Must match character-for-character. Leave empty ONLY when creating a brand-new file "
            "or appending new functions/methods/tests at the end of a file."
        ),
    )
    replacement_code: str = Field(
        description="Replacement code for target_code, or new content when creating/appending."
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


def _get_existing_dirty_sources(repo_path: str) -> list[str]:
    """Returns non-test tracked files currently modified on disk from a prior verified pass."""
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return []
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [
        line.strip().replace("\\", "/")
        for line in proc.stdout.splitlines()
        if line.strip() and not _is_test_file(line.strip())
    ]


def _collect_prioritized_files(
    repo_path: str,
    repo_files: List[str],
    retrieved_chunks: list,
    plan: List[str],
    task: str,
) -> dict[str, str]:
    """
    Loads full contents of directly relevant source files, package __init__.py files,
    and matching test files within separate character budgets.
    """
    ordered_candidates: List[str] = []
    seen: Set[str] = set()

    def _add(rel_path: str):
        if rel_path in repo_files and rel_path not in seen:
            seen.add(rel_path)
            ordered_candidates.append(rel_path)

    plan_and_task = task + "\n" + "\n".join(plan)
    mentioned_py = set(re.findall(r"[\w./-]+\.py", plan_and_task))
    for rel_path in repo_files:
        if rel_path in mentioned_py or os.path.basename(rel_path) in mentioned_py:
            _add(rel_path)

    for chunk in retrieved_chunks[:4]:
        fp = chunk.get("file_path", "").replace("\\", "/").lstrip("./")
        _add(fp)

    for path in list(ordered_candidates):
        pkg_dir = os.path.dirname(path)
        if pkg_dir and not _is_test_file(path):
            _add(f"{pkg_dir}/__init__.py")

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
    source_chars = 0
    test_chars = 0

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

        if _is_test_file(rel_path):
            if test_chars + len(content) > MAX_TEST_FILES_CHARS and any(
                _is_test_file(p) for p in full_files
            ):
                continue
            full_files[rel_path] = content
            test_chars += len(content)
        else:
            if source_chars + len(content) > MAX_SOURCE_FILES_CHARS and any(
                not _is_test_file(p) for p in full_files
            ):
                continue
            full_files[rel_path] = content
            source_chars += len(content)

    return full_files


def _try_safe_python_append_or_update(
    rel_path: str, content: str, replacement: str, task: str
) -> str | None:
    rep_clean = replacement.rstrip()
    if not rep_clean:
        return None

    try:
        ast.parse(rep_clean)
        if (
            len(rep_clean.splitlines()) >= max(1, int(len(content.splitlines()) * 0.85))
            and not check_source_regressions(rel_path, content, rep_clean, task)
        ):
            return rep_clean + "\n"
    except SyntaxError:
        pass

    if not replacement.startswith((" ", "\t")):
        try:
            ast.parse(rep_clean)
            candidate = content.rstrip() + "\n\n\n" + rep_clean + "\n"
            ast.parse(candidate)
            if not check_source_regressions(rel_path, content, candidate, task):
                return candidate
        except SyntaxError:
            pass

    dedented = textwrap.dedent(rep_clean)
    try:
        mod_ast = ast.parse(dedented)
        if mod_ast.body and all(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in mod_ast.body
        ):
            src_tree = ast.parse(content)
            classes = [n for n in src_tree.body if isinstance(n, ast.ClassDef)]
            if classes:
                last_cls = classes[-1]
                insert_at = getattr(last_cls, "end_lineno", None)
                if insert_at is not None:
                    indented_method = textwrap.indent(dedented, "    ")
                    lines = content.splitlines()
                    updated_lines = (
                        lines[:insert_at] + ["", indented_method] + lines[insert_at:]
                    )
                    candidate = "\n".join(updated_lines).rstrip() + "\n"
                    ast.parse(candidate)
                    if not check_source_regressions(rel_path, content, candidate, task):
                        return candidate
    except SyntaxError:
        pass

    return None


def _apply_search_replace(
    rel_path: str, original_content: str, edits: List[SearchReplaceBlock], task: str
) -> tuple[str, list[str]]:
    content = original_content
    errors: list[str] = []

    for idx, block in enumerate(edits, start=1):
        target = block.target_code
        replacement = block.replacement_code

        if not target.strip():
            if not content.strip():
                content = replacement.rstrip() + "\n"
                continue

            if _is_test_file(rel_path):
                if rel_path.endswith(".py"):
                    try:
                        ast.parse(replacement)
                        regressions = check_source_regressions(rel_path, content, replacement, task)
                        if not regressions and "def test_" in replacement:
                            content = replacement.rstrip() + "\n"
                            continue
                    except SyntaxError:
                        pass
                content = content.rstrip() + "\n\n\n" + replacement.lstrip().rstrip() + "\n"
                continue

            if rel_path.endswith(".py"):
                safe_updated = _try_safe_python_append_or_update(
                    rel_path, content, replacement, task
                )
                if safe_updated is not None:
                    content = safe_updated
                    continue

            eof_preview = "\n".join(content.splitlines()[-5:])
            errors.append(
                f"Edit #{idx} in '{rel_path}' had empty 'target_code' and could not be auto-appended. "
                f"Copy exact lines from '{rel_path}' (e.g. EOF lines:\n{eof_preview}\n) into 'target_code'."
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


def _generate_followup_tests(
    repo_path: str,
    task: str,
    updated_sources: dict[str, str],
    existing_tests: dict[str, str],
) -> dict[str, str]:
    """
    Focused follow-up pass when source edits are valid on disk but unit tests were omitted.
    """
    test_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert Python test engineer.\n"
                "Write unit tests covering the task using `pytest`.\n"
                "Rules:\n"
                "1. Output modifications ONLY for the target test file(s) listed below.\n"
                "2. Leave `target_code` empty (\"\") to append your new `def test_...` functions cleanly.\n"
                "3. Do NOT delete any existing tests.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Task:\n{task}\n\nUpdated Implementation Files:\n{sources}\n\nTarget Test Files:\n{tests}",
        ),
    ])

    sources_str = "\n\n".join(
        f"=== {p} ===\n{c}" for p, c in updated_sources.items()
    )
    tests_str = "\n\n".join(
        f"=== {p} ===\n{c if c.strip() else '(empty file)'}" for p, c in existing_tests.items()
    )

    test_out = invoke_structured_groq(
        prompt=test_prompt,
        payload={"task": task, "sources": sources_str, "tests": tests_str},
        schema=CodingOutput,
        temperature=0.1,
    )

    written_tests: dict[str, str] = {}
    for mod in test_out.modifications:
        norm_path = mod.file_path.replace("\\", "/").lstrip("./")
        if not _is_test_file(norm_path):
            continue
        full_path = os.path.join(repo_path, norm_path)
        orig = existing_tests.get(norm_path, "")
        if not orig and os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                orig = f.read()

        if os.path.exists(full_path):
            new_src, errs = _apply_search_replace(norm_path, orig, mod.edits, task)
            if errs:
                continue
        else:
            new_src = "\n".join(b.replacement_code for b in mod.edits)

        try:
            ast.parse(new_src)
        except SyntaxError:
            continue

        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_src)
        written_tests[norm_path] = new_src

    return written_tests


def coding_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates surgical search-and-replace patches with multi-model TPD
    failover, preserves verified source edits, auto-completes missing unit tests, and writes to disk.
    """
    logger.info("Coding Agent: Loading target files and generating surgical edits...")

    repo_path = state.get("repo_path", ".")
    repo_files = _get_repo_files(repo_path)
    file_tree_str = "\n".join(repo_files) if repo_files else "(no files found)"

    retrieved_chunks = state.get("retrieved_context", [])
    full_files_map = _collect_prioritized_files(
        repo_path, repo_files, retrieved_chunks, state.get("plan", []), state["task"]
    )
    discovered_test_files = sorted(p for p in full_files_map if _is_test_file(p))
    test_targets_hint = (
        ", ".join(discovered_test_files)
        if discovered_test_files
        else "tests/test_<module>.py"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert autonomous software engineer.\n"
                "Rules you MUST follow:\n"
                "1. SURGICAL SEARCH-AND-REPLACE ONLY: For existing files under 'Complete Target Files From Disk', "
                "output small `edits` blocks with exact `target_code` (copied character-for-character) "
                "and `replacement_code`. NEVER delete existing methods, functions, attributes, or default values.\n"
                "2. ADDING NEW METHODS OR TESTS: To add a new method/function at the end of an existing file, "
                "copy the last 3–5 lines of that file as `target_code` and include them followed by your new code in "
                "`replacement_code` (for test files, leaving `target_code` empty appends tests cleanly).\n"
                "3. PLACE LOGIC IN THE RIGHT MODULE: Put module-level constants/functions in the target module "
                "(e.g. `src/shopcore/pricing.py`) rather than inlining them into caller functions.\n"
                "4. SOURCE + TESTS TOGETHER: Always modify BOTH the target implementation file(s) AND the matching "
                f"test file ({test_targets_hint}) in the same response.\n\n"
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

    full_files_str = "\n\n".join([
        f"=== Complete File: {path} ===\n{content if content.strip() else '(empty file)'}"
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
    already_dirty_sources = _get_existing_dirty_sources(repo_path)

    try:
        candidate = invoke_structured_groq(
            prompt=prompt,
            payload={
                "task": state["task"],
                "file_tree": file_tree_str,
                "plan": "\n".join(state.get("plan", [])),
                "full_files": full_files_str,
                "snippets": snippets_str,
                "errors": recent_errors or "(none)",
            },
            schema=CodingOutput,
            temperature=0.1,
        )

        if not candidate.modifications:
            return {
                "modified_files": already_dirty_sources,
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
                # Read clean HEAD version for tracked files so patches apply against a known baseline
                base_content = full_files_map.get(norm_path)
                if base_content is None:
                    with open(full_path, "r", encoding="utf-8") as f:
                        base_content = f.read()
                    full_files_map[norm_path] = base_content

                new_src, patch_errs = _apply_search_replace(
                    norm_path, base_content, edits, state["task"]
                )
                static_errors.extend(patch_errs)
                if not patch_errs:
                    if norm_path.endswith(".py"):
                        static_errors.extend(
                            check_source_regressions(
                                norm_path, base_content, new_src, state["task"]
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
        has_source = any(not _is_test_file(p) for p in paths) or bool(already_dirty_sources)

        if require_source_edit and not has_source:
            static_errors.append(
                "Missing source code changes: include edits to the target implementation file(s) alongside tests."
            )

        # If source edits are valid and ONLY tests were omitted, write source edits and run a focused test pass
        if not static_errors and not has_test and candidate_contents:
            written_files = list(already_dirty_sources)
            for norm_path, content in candidate_contents.items():
                full_path = os.path.join(repo_path, norm_path)
                os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                if norm_path not in written_files:
                    written_files.append(norm_path)

            existing_tests_map = {
                p: full_files_map[p] for p in discovered_test_files if p in full_files_map
            }
            if not existing_tests_map:
                # Fallback target test path derived from primary edited source module
                primary_src = next((p for p in candidate_contents if not _is_test_file(p)), "module.py")
                stem = os.path.basename(primary_src)[:-3]
                existing_tests_map[f"tests/test_{stem}.py"] = ""

            try:
                written_tests = _generate_followup_tests(
                    repo_path,
                    state["task"],
                    {p: c for p, c in candidate_contents.items() if not _is_test_file(p)},
                    existing_tests_map,
                )
                for tp in written_tests:
                    if tp not in written_files:
                        written_files.append(tp)
                if written_tests:
                    return {"modified_files": written_files}
            except Exception as test_gen_err:
                logger.warning(f"Focused follow-up test generation failed: {test_gen_err}")

            return {
                "modified_files": written_files,
                "errors": [
                    f"Coding Error:\nMissing unit tests: add or update tests in {test_targets_hint}."
                ],
            }

        if not has_test:
            static_errors.append(
                f"Missing unit tests: add or update tests in {test_targets_hint}."
            )

        if static_errors:
            combined_msg = "\n".join(static_errors)
            logger.warning(f"Coding validation failed:\n{combined_msg}")
            return {
                "modified_files": already_dirty_sources,
                "errors": [f"Coding Error:\n{combined_msg}"],
            }

        # Reset only files being freshly overwritten so prior verified source edits remain intact
        if os.path.exists(os.path.join(repo_path, ".git")):
            for norm_path in candidate_contents:
                if os.path.exists(os.path.join(repo_path, norm_path)):
                    subprocess.run(
                        ["git", "checkout", "--", norm_path],
                        cwd=repo_path,
                        capture_output=True,
                    )

        modified_files: list[str] = list(already_dirty_sources)
        for norm_path, content in candidate_contents.items():
            full_path = os.path.join(repo_path, norm_path)
            os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            if norm_path not in modified_files:
                modified_files.append(norm_path)
            logger.info(f"Successfully wrote verified edits to: {full_path}")

        return {"modified_files": modified_files}

    except Exception as e:
        logger.error(f"Coding failed: {str(e)}")
        return {"modified_files": already_dirty_sources, "errors": [f"Coding Error: {str(e)}"]}