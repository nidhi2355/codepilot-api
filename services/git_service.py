import os
import logging
from git import Repo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from api.database import SessionLocal
from api.models.repository import Repository

logger = logging.getLogger(__name__)

# Define where repositories will be stored locally
REPOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "repos"))
os.makedirs(REPOS_DIR, exist_ok=True)

class GitService:
    @staticmethod
    async def clone_repository_task(repo_id: int, github_url: str):
        """Background task to clone the repo and update its status."""
        repo_name = github_url.rstrip("/").split("/")[-1]
        target_path = os.path.join(REPOS_DIR, repo_name)
        
        # Open a new database session specifically for the background task
        async with SessionLocal() as db:
            try:
                logger.info(f"Starting clone for {github_url} into {target_path}")
                
                # Clone if it doesn't exist, otherwise skip
                if not os.path.exists(target_path):
                    # GitPython blocks, but it's acceptable in this background task for MVP
                    Repo.clone_from(github_url, target_path)
                    logger.info(f"Successfully cloned {repo_name}")
                else:
                    logger.info(f"Repository {repo_name} already exists locally.")

                # Update DB status to 'cloned'
                await GitService._update_repo_status(db, repo_id, "cloned")

            except Exception as e:
                logger.error(f"Failed to clone {github_url}: {str(e)}")
                # Update DB status to 'error'
                await GitService._update_repo_status(db, repo_id, "error")

    @staticmethod
    async def _update_repo_status(db: AsyncSession, repo_id: int, status: str):
        """Helper to update the repository status in the database."""
        result = await db.execute(select(Repository).filter(Repository.id == repo_id))
        repo = result.scalars().first()
        if repo:
            repo.status = status
            await db.commit()