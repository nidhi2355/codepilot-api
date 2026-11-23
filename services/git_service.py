import asyncio
import os
import re
import logging
import tempfile
from git import Repo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from api.database import SessionLocal
from api.models.repository import Repository, RepositoryFile
from services.vector_store import qdrant, COLLECTION_NAME
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

temp_dir = tempfile.gettempdir()
REPOS_DIR = os.path.join(temp_dir, "codepilot_repos")
os.makedirs(REPOS_DIR, exist_ok=True)


def _build_auth_url(github_url: str, github_token: str | None) -> str:
    """Injects an ephemeral x-access-token into GitHub HTTPS URLs when a token is provided."""
    if not github_token:
        return github_url
    match = re.match(r"^https://github\.com/(.+)$", github_url.strip())
    if not match:
        return github_url
    return f"https://x-access-token:{github_token}@github.com/{match.group(1)}"


class GitService:
    @staticmethod
    def get_repo_dir(repo_id: int, github_url: str) -> str:
        repo_name = github_url.rstrip("/").split("/")[-1]
        return os.path.join(REPOS_DIR, f"repo_{repo_id}_{repo_name}")

    @staticmethod
    def _clone_or_pull_clean(
        target_path: str, github_url: str, github_token: str | None = None
    ) -> Repo:
        """Clones or pulls using an ephemeral token URL and ensures .git/config stays token-free."""
        auth_url = _build_auth_url(github_url, github_token)

        if not os.path.exists(os.path.join(target_path, ".git")):
            try:
                Repo.clone_from(auth_url, target_path)
            finally:
                if os.path.exists(os.path.join(target_path, ".git")):
                    Repo(target_path).remotes.origin.set_url(github_url)
            return Repo(target_path)

        local_repo = Repo(target_path)
        origin = local_repo.remotes.origin
        try:
            origin.set_url(auth_url)
            origin.fetch()
            try:
                default_ref = origin.refs.HEAD.reference.name
            except Exception:
                default_ref = "origin/main"
            branch_name = default_ref.split("/", 1)[-1]
            local_repo.git.checkout("-B", branch_name, default_ref)
            local_repo.git.reset("--hard", default_ref)
            local_repo.git.clean("-fd")
        finally:
            origin.set_url(github_url)

        return local_repo

    @staticmethod
    async def clone_repository_task(
        repo_id: int, github_url: str, github_token: str | None = None
    ):
        target_path = GitService.get_repo_dir(repo_id, github_url)
        already_existed = os.path.exists(os.path.join(target_path, ".git"))

        async with SessionLocal() as db:
            try:
                await asyncio.to_thread(
                    GitService._clone_or_pull_clean,
                    target_path,
                    github_url,
                    github_token,
                )

                if already_existed:
                    await db.execute(
                        delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id)
                    )
                    try:
                        qdrant.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[
                                    FieldCondition(
                                        key="repo_id", match=MatchValue(value=repo_id)
                                    )
                                ]
                            ),
                        )
                    except Exception as q_err:
                        logger.warning(f"Failed to clear Qdrant vectors: {str(q_err)}")

                await GitService._update_repo_status(db, repo_id, "cloned")
            except Exception as e:
                logger.error(f"Failed to clone/sync {github_url}: {str(e)}")
                await GitService._update_repo_status(db, repo_id, "error")

    @staticmethod
    async def ensure_fresh_repo(
        repo_id: int, github_url: str, github_token: str | None = None
    ) -> tuple[str, bool, str]:
        """
        Ensures the local clone matches origin's latest commit.
        Returns (target_path, needs_reindex, current_sha).
        """
        target_path = GitService.get_repo_dir(repo_id, github_url)
        local_repo = await asyncio.to_thread(
            GitService._clone_or_pull_clean, target_path, github_url, github_token
        )
        current_sha = local_repo.head.commit.hexsha

        async with SessionLocal() as db:
            result = await db.execute(select(Repository).filter(Repository.id == repo_id))
            repo = result.scalars().first()
            if not repo:
                raise ValueError(f"Repository {repo_id} not found")

            if repo.last_indexed_sha == current_sha and repo.status == "ready":
                return target_path, False, current_sha

            logger.info(
                f"Repo {repo_id} SHA changed ({repo.last_indexed_sha} -> {current_sha}). Re-indexing..."
            )
            await db.execute(
                delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id)
            )
            try:
                qdrant.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
                    ),
                )
            except Exception as q_err:
                logger.warning(f"Failed to clear Qdrant vectors: {str(q_err)}")

            repo.status = "cloned"
            await db.commit()

        return target_path, True, current_sha

    @staticmethod
    async def save_indexed_sha(repo_id: int, sha: str):
        async with SessionLocal() as db:
            result = await db.execute(select(Repository).filter(Repository.id == repo_id))
            repo = result.scalars().first()
            if repo:
                repo.last_indexed_sha = sha
                await db.commit()

    @staticmethod
    async def _update_repo_status(db: AsyncSession, repo_id: int, status: str):
        result = await db.execute(select(Repository).filter(Repository.id == repo_id))
        repo = result.scalars().first()
        if repo:
            repo.status = status
            await db.commit()