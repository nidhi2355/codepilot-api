import hashlib
import logging
import os
import re
import subprocess
import tempfile
import venv
from agents.state import AgentState

logger = logging.getLogger(__name__)

VENVS_BASE_DIR = os.path.join(tempfile.gettempdir(), "codepilot_venvs")
os.makedirs(VENVS_BASE_DIR, exist_ok=True)

IMPORT_TO_PYPI = {
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "docx": "python-docx",
    "fitz": "PyMuPDF",
    "Crypto": "pycryptodome",
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}


def _get_venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _find_dependency_files(repo_path: str) -> list[str]:
    """Finds requirements*.txt and pyproject.toml up to 2 directory levels deep."""
    found: list[str] = []
    if not os.path.isdir(repo_path):
        return found

    base_depth = repo_path.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > 2:
            dirs[:] = []
            continue

        for fname in sorted(files):
            if fname in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
                found.append(os.path.join(root, fname))

    return sorted(found)


def _compute_deps_hash(dep_files: list[str]) -> str:
    hasher = hashlib.sha256()
    for path in dep_files:
        hasher.update(path.encode("utf-8"))
        try:
            with open(path, "rb") as f:
                hasher.update(f.read())
        except OSError:
            pass
    return hasher.hexdigest()


def _ensure_repo_test_env(target_dir: str) -> str:
    """
    Creates or reuses an isolated virtualenv for target_dir and installs
    pytest plus any repository requirements. Returns path to venv python.
    """
    abs_target = os.path.abspath(target_dir)
    repo_key = os.path.basename(abs_target) or "default_repo"
    venv_dir = os.path.join(VENVS_BASE_DIR, repo_key)
    venv_python = _get_venv_python(venv_dir)

    if not os.path.exists(venv_python):
        logger.info(f"Creating isolated test virtualenv at {venv_dir}...")
        venv.create(venv_dir, with_pip=True, symlinks=(os.name != "nt"))

    dep_files = _find_dependency_files(abs_target)
    current_hash = _compute_deps_hash(dep_files)
    marker_file = os.path.join(venv_dir, ".deps_hash")

    cached_hash = None
    if os.path.exists(marker_file):
        try:
            with open(marker_file, "r", encoding="utf-8") as f:
                cached_hash = f.read().strip()
        except OSError:
            cached_hash = None

    if cached_hash != current_hash:
        logger.info(f"Installing test dependencies for {repo_key}...")
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "pytest"],
            capture_output=True,
            text=True,
            timeout=90,
        )

        for dep_path in dep_files:
            if os.path.basename(dep_path).startswith("requirements"):
                res = subprocess.run(
                    [
                        venv_python,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--disable-pip-version-check",
                        "-r",
                        dep_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if res.returncode != 0:
                    logger.warning(f"Partial dependency install warning for {dep_path}: {res.stderr[:400]}")

        root_pyproject = os.path.join(abs_target, "pyproject.toml")
        if os.path.exists(root_pyproject):
            subprocess.run(
                [
                    venv_python,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    "--no-build-isolation",
                    "-e",
                    abs_target,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

        with open(marker_file, "w", encoding="utf-8") as f:
            f.write(current_hash)

    return venv_python


def _is_local_module(target_dir: str, top_module: str) -> bool:
    """Checks if top_module corresponds to a local file/folder inside the repo."""
    candidates = [
        os.path.join(target_dir, top_module),
        os.path.join(target_dir, f"{top_module}.py"),
        os.path.join(target_dir, "src", top_module),
        os.path.join(target_dir, "src", f"{top_module}.py"),
    ]
    return any(os.path.exists(p) for p in candidates)


def _run_pytest_subprocess(venv_python: str, target_dir: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    py_paths = [os.path.abspath(target_dir)]
    src_dir = os.path.join(os.path.abspath(target_dir), "src")
    if os.path.isdir(src_dir):
        py_paths.insert(0, src_dir)
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    return subprocess.run(
        [venv_python, "-m", "pytest"],
        cwd=target_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def execute_tests_node(state: AgentState) -> dict:
    """
    LangGraph node: Prepares an isolated repo test environment and executes pytest.
    """
    logger.info("Testing Agent: Running test suite...")
    target_dir = state.get("repo_path", ".")

    try:
        venv_python = _ensure_repo_test_env(target_dir)
        result = _run_pytest_subprocess(venv_python, target_dir)
        output = result.stdout + "\n" + result.stderr

        # Auto-install unlisted third-party package if pytest collection hits ModuleNotFoundError
        if result.returncode != 0:
            missing_match = re.search(
                r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", output
            )
            if missing_match:
                top_mod = missing_match.group(1).split(".")[0]
                if not _is_local_module(target_dir, top_mod):
                    pypi_pkg = IMPORT_TO_PYPI.get(top_mod, top_mod)
                    logger.info(f"Installing missing third-party package '{pypi_pkg}' and retrying pytest...")
                    install_res = subprocess.run(
                        [
                            venv_python,
                            "-m",
                            "pip",
                            "install",
                            "--quiet",
                            "--disable-pip-version-check",
                            pypi_pkg,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if install_res.returncode == 0:
                        result = _run_pytest_subprocess(venv_python, target_dir)
                        output = result.stdout + "\n" + result.stderr

        passed = result.returncode == 0
        logger.info(f"Testing completed. Passed: {passed}")

        return {
            "test_results": {
                "passed": passed,
                "output": output[-2000:],
            }
        }

    except subprocess.TimeoutExpired:
        logger.error("Testing failed: Execution timed out.")
        return {"test_results": {"passed": False, "output": "Execution timed out after 30 seconds."}}
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        return {"errors": [f"Testing Error: {str(e)}"]}