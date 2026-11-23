import os
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

# Use the system's temporary directory for stateless deployment
temp_dir = tempfile.gettempdir()
REPOS_DIR = os.path.join(temp_dir, "codepilot_repos")
os.makedirs(REPOS_DIR, exist_ok=True)

class GitService:
    @staticmethod
    async def clone_repository_task(repo_id: int, github_url: str):
        print(f"DEBUG: Background task executing for repo ID {repo_id}")
        repo_name = github_url.rstrip("/").split("/")[-1]
        target_path = os.path.join(REPOS_DIR, repo_name)
        
        async with SessionLocal() as db:
            try:
                if not os.path.exists(target_path):
                    logger.info(f"Starting clone for {github_url} into {target_path}")
                    Repo.clone_from(github_url, target_path)
                    logger.info(f"Successfully cloned {repo_name}")
                else:
                    logger.info(f"Repository {repo_name} already exists. Pulling latest and cleaning old data...")
                    local_repo = Repo(target_path)
                    local_repo.remotes.origin.pull()

                    # 1. Clear old files and symbols from PostgreSQL 
                    await db.execute(delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id))
                    
                    # 2. Clear old vectors from Qdrant Cloud
                    try:
                        qdrant.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=Filter(
                                must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
                            )
                        )
                    except Exception as q_err:
                        logger.warning(f"Failed to clear Qdrant vectors: {str(q_err)}")

                # Update DB status to 'cloned' so it's ready for Phase 3
                await GitService._update_repo_status(db, repo_id, "cloned")

            except Exception as e:
                logger.error(f"Failed to clone/sync {github_url}: {str(e)}")
                await GitService._update_repo_status(db, repo_id, "error")

    @staticmethod
    async def _update_repo_status(db: AsyncSession, repo_id: int, status: str):
        result = await db.execute(select(Repository).filter(Repository.id == repo_id))
        repo = result.scalars().first()
        if repo:
            repo.status = status
            await db.commit()

    @staticmethod
    def _sync_to_origin_default(target_path: str, github_url: str) -> str:
        """Clones if missing, otherwise hard-resets to origin's default branch and returns HEAD SHA."""
        if not os.path.exists(os.path.join(target_path, ".git")):
            Repo.clone_from(github_url, target_path)
            local_repo = Repo(target_path)
        else:
            local_repo = Repo(target_path)
            local_repo.remotes.origin.fetch()
            try:
                default_ref = local_repo.remotes.origin.refs.HEAD.reference.name
            except Exception:
                default_ref = "origin/main"
            branch_name = default_ref.split("/", 1)[-1]
            local_repo.git.checkout("-B", branch_name, default_ref)
            local_repo.git.reset("--hard", default_ref)
            local_repo.git.clean("-fd")

        return local_repo.head.commit.hexsha

    @staticmethod
    async def ensure_fresh_repo(repo_id: int, github_url: str) -> tuple[str, bool, str]:
        """
        Ensures the local clone matches origin's latest commit.
        Returns (target_path, needs_reindex, current_sha).
        """
        import asyncio

        repo_name = github_url.rstrip("/").split("/")[-1]
        target_path = os.path.join(REPOS_DIR, repo_name)
        current_sha = await asyncio.to_thread(
            GitService._sync_to_origin_default, target_path, github_url
        )

        async with SessionLocal() as db:
            result = await db.execute(select(Repository).filter(Repository.id == repo_id))
            repo = result.scalars().first()
            if not repo:
                raise ValueError(f"Repository {repo_id} not found")

            if repo.last_indexed_sha == current_sha and repo.status == "ready":
                return target_path, False, current_sha

            logger.info(
                f"Repo {repo_id} SHA changed ({repo.last_indexed_sha} -> {current_sha}). Clearing stale index..."
            )
            await db.execute(delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id))
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