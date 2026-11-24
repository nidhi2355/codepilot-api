import ast
import os
import re
import subprocess


def extract_ast_contract(source: str) -> dict:
    """Extracts top-level functions, classes, attributes, methods, and default values."""
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


def check_source_regressions(
    rel_path: str, old_source: str, new_source: str, task: str
) -> list[str]:
    """Compares old vs. new Python source and returns any unrequested symbol/default deletions."""
    violations: list[str] = []
    task_lower = task.lower()
    allows_removal = bool(re.search(r"\b(remove|delete|drop|rename|deprecate)\b", task_lower))

    try:
        old_contract = extract_ast_contract(old_source)
        new_contract = extract_ast_contract(new_source)
    except SyntaxError as e:
        return [f"SyntaxError in {rel_path} at line {e.lineno}: {e.msg}"]

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

    for cls_name, old_cls in old_contract["classes"].items():
        if cls_name not in new_contract["classes"]:
            if not (allows_removal and cls_name.lower() in task_lower):
                violations.append(f"Deleted class '{cls_name}' in {rel_path}")
            continue
        new_cls = new_contract["classes"][cls_name]

        for attr_name, old_val in old_cls["attrs"].items():
            if attr_name not in new_cls["attrs"]:
                if not (allows_removal and attr_name.lower() in task_lower):
                    violations.append(f"Deleted attribute '{cls_name}.{attr_name}' in {rel_path}")
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
                    violations.append(f"Deleted method '{cls_name}.{method_name}' in {rel_path}")
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


def audit_diff_regressions(repo_path: str, modified_files: list[str], task: str) -> list[str]:
    """Compares modified Python files on disk against git HEAD."""
    violations: list[str] = []
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

        with open(full_path, "r", encoding="utf-8") as f:
            new_source = f.read()

        violations.extend(check_source_regressions(rel_path, head_proc.stdout, new_source, task))

    return violations