import os
import re
import uuid
import logging
import httpx
from git import Repo, Actor
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from agents.state import AgentState

logger = logging.getLogger(__name__)


class PRDetails(BaseModel):
    title: str = Field(description="A concise, descriptive title for the pull request.")
    body: str = Field(
        description="A detailed markdown description of the changes made, explaining the 'why' and 'how'."
    )


def _parse_github_owner_repo(remote_url: str) -> tuple[str, str]:
    """Extracts (owner, repo) from HTTPS or SSH GitHub remote URLs."""
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", remote_url.strip())
    if not match:
        raise ValueError(f"Could not parse GitHub owner/repo from remote URL: {remote_url}")
    return match.group(1), match.group(2)


def pr_creation_node(state: AgentState) -> dict:
    """
    LangGraph node: Generates PR details, commits and pushes a new branch,
    and opens a real Pull Request on GitHub.
    """
    logger.info("PR Creation Agent: Generating PR details and opening pull request...")

    modified_files = state.get("modified_files", [])
    if not modified_files:
        logger.warning("No files were modified. Skipping PR creation.")
        return {"status": "completed"}

    existing_test_results = dict(state.get("test_results") or {})

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    parser = JsonOutputParser(pydantic_object=PRDetails)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a senior software engineer opening a pull request. "
                "Based on the original task, the execution plan, and the files modified, "
                "write a clear, professional PR title and a markdown-formatted body. "
                "Mention that the changes were automatically tested and verified.\n\n"
                "{format_instructions}"
            ),
        ),
        ("human", "Task: {task}\n\nPlan: {plan}\n\nModified Files: {files}"),
    ])

    chain = prompt | llm | parser

    try:
        pr_details = chain.invoke({
            "task": state["task"],
            "plan": "\n".join(state.get("plan", [])),
            "files": ", ".join(modified_files),
            "format_instructions": parser.get_format_instructions(),
        })

        repo_path = state.get("repo_path", ".")
        if not os.path.exists(os.path.join(repo_path, ".git")):
            logger.warning(f"No .git directory found in {repo_path}. Simulating PR creation.")
            existing_test_results["pr_url"] = "simulated_pr_url"
            return {"status": "pr_created", "test_results": existing_test_results}

        repo = Repo(repo_path)
        base_branch = repo.active_branch.name
        branch_name = f"codepilot-fix-{uuid.uuid4().hex[:8]}"

        # Create and checkout feature branch
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()

        # Stage and commit with explicit author
        repo.index.add(modified_files)
        actor = Actor("CodePilot Bot", "codepilot-bot@users.noreply.github.com")
        commit_message = f"{pr_details['title']}\n\n{pr_details['body']}"
        repo.index.commit(commit_message, author=actor, committer=actor)
        logger.info(f"Committed changes to branch: {branch_name}")

        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            logger.warning("GITHUB_TOKEN is not set. Skipping push and GitHub PR creation.")
            repo.heads[base_branch].checkout()
            existing_test_results["pr_url"] = f"local-branch:{branch_name} (GITHUB_TOKEN not set)"
            return {"status": "pr_created", "test_results": existing_test_results}

        origin = repo.remotes.origin
        original_url = origin.url
        owner, repo_name = _parse_github_owner_repo(original_url)
        auth_url = f"https://x-access-token:{github_token}@github.com/{owner}/{repo_name}.git"

        # Push branch using authenticated URL, then restore clean remote URL
        try:
            origin.set_url(auth_url)
            push_info = origin.push(refspec=f"{branch_name}:{branch_name}")
            for info in push_info:
                if info.flags & info.ERROR:
                    raise RuntimeError(f"Git push failed: {info.summary}")
        finally:
            origin.set_url(original_url)
            repo.heads[base_branch].checkout()

        logger.info(f"Pushed branch {branch_name} to {owner}/{repo_name}")

        # Open Pull Request via GitHub REST API
        pr_response = httpx.post(
            f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "title": pr_details["title"],
                "body": pr_details["body"],
                "head": branch_name,
                "base": base_branch,
            },
            timeout=30.0,
        )
        pr_response.raise_for_status()
        pr_url = pr_response.json()["html_url"]
        logger.info(f"Opened Pull Request: {pr_url}")

        existing_test_results["pr_url"] = pr_url
        return {"status": "pr_created", "test_results": existing_test_results}

    except Exception as e:
        logger.error(f"PR Creation failed: {str(e)}")
        return {"errors": [f"PR Creation Error: {str(e)}"]}