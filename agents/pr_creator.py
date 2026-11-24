import os
import re
import uuid
import logging
from typing import Optional
import httpx
from git import Repo, Actor
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)


class PRDetails(BaseModel):
    title: str = Field(description="A concise, descriptive title for the pull request.")
    body: str = Field(
        description="A factual markdown description of the exact changes in the diff."
    )


def _parse_github_owner_repo(remote_url: str) -> tuple[str, str]:
    """Extracts (owner, repo) from HTTPS or SSH GitHub remote URLs."""
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", remote_url.strip())
    if not match:
        raise ValueError(f"Could not parse GitHub owner/repo from remote URL: {remote_url}")
    return match.group(1), match.group(2)


def _extract_issue_number(state: AgentState) -> Optional[int]:
    """Extracts the GitHub issue number from explicit state or the task text."""
    explicit = state.get("issue_number")
    if explicit:
        return int(explicit)

    task = state.get("task", "")
    patterns = [
        r"github\.com/[^/\s]+/[^/\s]+/issues/(\d+)",
        r"\bissue\s*#?\s*(\d+)\b",
        r"(?:^|\s)#(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, task, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def pr_creation_node(state: AgentState) -> dict:
    """
    LangGraph node: Stages all changes with `git add -A`, generates a PR summary
    grounded strictly in `git diff --cached`, pushes a branch, and opens a PR.
    """
    logger.info("PR Creation Agent: Staging changes and opening pull request...")

    modified_files = state.get("modified_files", [])
    if not modified_files:
        logger.warning("No files were modified. Skipping PR creation.")
        return {"status": "error", "errors": ["PR Creation Skipped: No files were modified."]}

    existing_test_results = dict(state.get("test_results") or {})
    repo_path = state.get("repo_path", ".")

    if not os.path.exists(os.path.join(repo_path, ".git")):
        logger.warning(f"No .git directory found in {repo_path}.")
        return {"status": "error", "errors": [f"PR Creation Error: No .git repository at {repo_path}."]}

    repo = Repo(repo_path)
    base_branch = repo.active_branch.name
    branch_name = f"codepilot-fix-{uuid.uuid4().hex[:8]}"

    base_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )
    llm = base_llm.bind(tool_choice="none")
    parser = JsonOutputParser(pydantic_object=PRDetails)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior software engineer writing a pull request description from a git diff.\n"
                "Rules you MUST follow:\n"
                "1. DIFF-ONLY GROUNDING: Document ONLY files, functions, classes, and test cases that explicitly "
                "appear in the Git Diff below. Never mention any file or test function that is not in the diff, "
                "and never use hedging phrases such as 'or adjusted logic', 'likely', or 'may have'.\n"
                "2. STRUCTURE: Return a concise PR title and markdown body with sections "
                "'## Summary', '## Changes', and '## Verification'.\n"
                "3. ISSUE LINKING: If an Issue Closing Keyword (e.g., 'Closes #1') is provided, include a "
                "'## Related Issue' section containing that exact keyword.\n\n"
                "{format_instructions}"
            ),
        ),
        (
            "human",
            "Issue Closing Keyword: {issue_keyword}\n\n"
            "Staged Files (git diff --cached --name-status):\n{name_status}\n\n"
            "Git Diff:\n```diff\n{git_diff}\n```",
        ),
    ])

    chain = prompt | llm | parser

    try:
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()

        # Stage all tracked edits AND newly created untracked files
        repo.git.add("-A")
        for ignored_pattern in ("__pycache__", "*.pyc", ".pytest_cache"):
            try:
                repo.git.reset("HEAD", "--", ignored_pattern)
            except Exception:
                pass

        name_status = repo.git.diff("--cached", "--name-status").strip()
        if not name_status:
            raise RuntimeError("No staged changes found after `git add -A`.")

        staged_diff = repo.git.diff("--cached")
        if len(staged_diff) > 10000:
            staged_diff = staged_diff[:10000] + "\n... [diff truncated]"

        issue_number = _extract_issue_number(state)
        issue_keyword = f"Closes #{issue_number}" if issue_number else "None"

        pr_details = None
        last_llm_err = None
        for attempt in range(2):
            try:
                raw = chain.invoke({
                    "issue_keyword": issue_keyword,
                    "name_status": name_status,
                    "git_diff": staged_diff,
                    "format_instructions": parser.get_format_instructions(),
                })
                pr_details = PRDetails(**raw)
                break
            except (ValidationError, ValueError, TypeError) as err:
                last_llm_err = err
                logger.warning(f"PRDetails parse attempt {attempt + 1} failed: {err}")

        if pr_details is None:
            raise RuntimeError(f"Failed to generate valid PR summary JSON: {last_llm_err}")

        pr_body = pr_details.body.strip()
        if issue_number:
            has_closing_keyword = re.search(
                rf"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#{issue_number}\b",
                pr_body,
                re.IGNORECASE,
            )
            if not has_closing_keyword:
                pr_body += f"\n\n## Related Issue\nCloses #{issue_number}"

        actor = Actor("CodePilot Bot", "codepilot-bot@users.noreply.github.com")
        commit_message = f"{pr_details.title}\n\n{pr_body}"
        repo.index.commit(commit_message, author=actor, committer=actor)
        logger.info(f"Committed changes to branch: {branch_name}")

        github_token = state.get("github_token") or os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise RuntimeError(
                "No GitHub access token found for this session. Sign out and sign in with GitHub again."
            )

        origin = repo.remotes.origin
        original_url = origin.url
        owner, repo_name = _parse_github_owner_repo(original_url)
        auth_url = f"https://x-access-token:{github_token}@github.com/{owner}/{repo_name}.git"

        try:
            origin.set_url(auth_url)
            push_info = origin.push(refspec=f"{branch_name}:{branch_name}")
            for info in push_info:
                if info.flags & info.ERROR:
                    raise RuntimeError(
                        f"Git push to {owner}/{repo_name} failed ({info.summary.strip()}). "
                        "Ensure your GitHub account has write access and 'repo' OAuth scope."
                    )
        finally:
            origin.set_url(original_url)
            repo.heads[base_branch].checkout()

        logger.info(f"Pushed branch {branch_name} to {owner}/{repo_name}")

        pr_response = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "title": pr_details.title,
                "body": pr_body,
                "head": branch_name,
                "base": base_branch,
            },
            timeout=30.0,
        )
        if pr_response.status_code >= 400:
            raise RuntimeError(
                f"GitHub PR API returned {pr_response.status_code}: {pr_response.text}"
            )

        pr_url = pr_response.json()["html_url"]
        logger.info(f"Opened Pull Request: {pr_url}")

        existing_test_results["pr_url"] = pr_url
        return {"status": "pr_created", "test_results": existing_test_results}

    except Exception as e:
        logger.error(f"PR Creation failed: {str(e)}")
        try:
            if repo.active_branch.name != base_branch:
                repo.heads[base_branch].checkout()
        except Exception:
            pass
        return {"status": "error", "errors": [f"PR Creation Error: {str(e)}"]}