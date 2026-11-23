import ast
import logging
import os
import re
import subprocess
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.planner import plan_node
from agents.retriever import retrieve_node
from agents.coding import coding_node
from agents.testing import execute_tests_node
from agents.reflection import reflection_node
from agents.pr_creator import pr_creation_node

logger = logging.getLogger(__name__)


def _is_test_file(path: str) -> bool:
    norm = path.replace("\\", "/").lstrip("./")
    base = os.path.basename(norm)
    return (
        norm.startswith("tests/")
        or "/tests/" in norm
        or base.startswith("test_")
        or base.endswith("_test.py")
    )


def _extract_ast_contract(source: str) -> dict:
    """Extracts classes, attributes, functions, and default values from Python source."""
    tree = ast.parse(source)
    contract = {
        "functions": {},  # name -> dict(param -> default_repr)
        "classes": {},    # cls_name -> {"attrs": {attr: default_repr}, "methods": {m: {param: default_repr}}}
    }

    def _param_defaults(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
        args = fn_node.args.args
        defaults = fn_node.args.defaults
        mapping: dict[str, str] = {}
        if defaults:
            for arg, default_node in zip(args[-len(defaults):], defaults):
                mapping[arg.arg] = ast.unparse(default_node)
        for kwarg, kw_default in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
            if kw_default is not None:
                mapping[kwarg.arg] = ast.unparse(kw_default)
        return mapping

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            contract["functions"][node.name] = _param_defaults(node)
        elif isinstance(node, ast.ClassDef):
            attrs: dict[str, str | None] = {}
            methods: dict[str, dict[str, str]] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attrs[item.target.id] = ast.unparse(item.value) if item.value else None
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attrs[target.id] = ast.unparse(item.value)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = _param_defaults(item)
            contract["classes"][node.name] = {"attrs": attrs, "methods": methods}

    return contract


def audit_diff_regressions(repo_path: str, modified_files: list[str], task: str) -> list[str]:
    """
    Compares modified Python files against HEAD to catch unrequested deletions
    of attributes, functions, test_* cases, or altered default values.
    """
    violations: list[str] = []
    task_lower = task.lower()
    allows_removal = bool(re.search(r"\b(remove|delete|drop|rename|deprecate)\b", task_lower))

    for rel_path in modified_files:
        if not rel_path.endswith(".py"):
            continue
        full_path = os.path.join(repo_path, rel_path)
        if not os.path.isfile(full_path):
            continue

        head_proc = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if head_proc.returncode != 0:
            continue  # Newly created file

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                new_source = f.read()
            old_contract = _extract_ast_contract(head_proc.stdout)
            new_contract = _extract_ast_contract(new_source)
        except SyntaxError:
            continue

        # Check top-level functions (including def test_*)
        for fn_name, old_defaults in old_contract["functions"].items():
            if fn_name not in new_contract["functions"]:
                if not (allows_removal and fn_name.lower() in task_lower):
                    violations.append(f"Deleted function '{fn_name}' in {rel_path}")
                continue
            new_defaults = new_contract["functions"][fn_name]
            for param, old_val in old_defaults.items():
                if param in new_defaults and new_defaults[param] != old_val:
                    if param.lower() not in task_lower:
                        violations.append(
                            f"Changed default value for '{fn_name}({param})' in {rel_path} "
                            f"from {old_val} to {new_defaults[param]}"
                        )

        # Check classes, attributes, and methods
        for cls_name, old_cls in old_contract["classes"].items():
            if cls_name not in new_contract["classes"]:
                if not (allows_removal and cls_name.lower() in task_lower):
                    violations.append(f"Deleted class '{cls_name}' in {rel_path}")
                continue
            new_cls = new_contract["classes"][cls_name]

            for attr_name, old_val in old_cls["attrs"].items():
                if attr_name not in new_cls["attrs"]:
                    if not (allows_removal and attr_name.lower() in task_lower):
                        violations.append(
                            f"Deleted attribute '{cls_name}.{attr_name}' in {rel_path}"
                        )
                elif (
                    old_val is not None
                    and new_cls["attrs"][attr_name] != old_val
                    and attr_name.lower() not in task_lower
                ):
                    violations.append(
                        f"Changed default value of '{cls_name}.{attr_name}' in {rel_path} "
                        f"from {old_val} to {new_cls['attrs'][attr_name]}"
                    )

            for method_name, old_defaults in old_cls["methods"].items():
                if method_name not in new_cls["methods"]:
                    if not (allows_removal and method_name.lower() in task_lower):
                        violations.append(
                            f"Deleted method '{cls_name}.{method_name}' in {rel_path}"
                        )
                    continue
                new_defaults = new_cls["methods"][method_name]
                for param, old_val in old_defaults.items():
                    if param in new_defaults and new_defaults[param] != old_val:
                        if param.lower() not in task_lower:
                            violations.append(
                                f"Changed default value for '{cls_name}.{method_name}({param})' "
                                f"in {rel_path} from {old_val} to {new_defaults[param]}"
                            )

    return violations


def route_after_testing(state: AgentState) -> str:
    """Determine the next node based on test results, diff audit, test coverage, and retry count."""
    test_results = state.get("test_results", {})
    passed = test_results.get("passed", False)
    retry_count = state.get("retry_count", 0)
    modified_files = state.get("modified_files", [])
    MAX_RETRIES = 3

    if passed:
        # 1. Enforce test-file coverage
        has_test_file = any(_is_test_file(f) for f in modified_files)
        has_source_change = any(not _is_test_file(f) for f in modified_files)
        task_mentions_test = bool(
            re.search(r"\btests?\b|\bpytest\b|\bunit\s*test", state.get("task", ""), re.IGNORECASE)
        )
        if not has_test_file and (has_source_change or task_mentions_test) and retry_count < MAX_RETRIES:
            logger.info("Tests passed, but no test files were modified. Routing to reflection.")
            return "reflection"

        # 2. Audit git diff for unrequested deletions/default changes
        violations = audit_diff_regressions(
            state.get("repo_path", "."), modified_files, state.get("task", "")
        )
        if violations and retry_count < MAX_RETRIES:
            logger.warning(f"Diff audit detected regressions: {violations}. Routing to reflection.")
            return "reflection"

        logger.info("Tests and diff audit passed. Routing to PR creation.")
        return "pr_creator"
    elif retry_count >= MAX_RETRIES:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached. Routing to END with failure.")
        return END
    else:
        logger.info(f"Tests failed (Retry {retry_count}). Routing to reflection.")
        return "reflection"


def create_agent_workflow() -> StateGraph:
    logger.info("Initializing LangGraph Supervisor Workflow...")
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("planner", plan_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("testing", execute_tests_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("pr_creator", pr_creation_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "planner")
    workflow.add_edge("planner", "coding")
    workflow.add_edge("coding", "testing")

    workflow.add_conditional_edges("testing", route_after_testing)
    workflow.add_edge("reflection", "coding")
    workflow.add_edge("pr_creator", END)

    return workflow