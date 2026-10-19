import os
import logging
import git
from sqlalchemy.future import select
from api.database import SessionLocal
from api.models.repository import Repository

logger = logging.getLogger(__name__)

class GitService:
    @staticmethod
    async def clone_repository_task(repo_id: int, github_url: str):
        print(f"DEBUG: Background task executing for repo ID {repo_id}")
        async with SessionLocal() as db:
            try:
                result = await db.execute(select(Repository).filter(Repository.id == repo_id))
                repo = result.scalars().first()
                if not repo:
                    return

                repo.status = "cloning"
                await db.commit()

                repo_name = github_url.split("/")[-1].replace(".git", "")
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "repos"))
                os.makedirs(base_dir, exist_ok=True)
                repo_dir = os.path.join(base_dir, repo_name)

                # THE FIX: Pull latest commits if it exists, otherwise clone
                if os.path.exists(repo_dir) and os.path.isdir(os.path.join(repo_dir, ".git")):
                    logger.info(f"Repository already exists. Pulling latest changes for {repo_name}...")
                    local_repo = git.Repo(repo_dir)
                    local_repo.remotes.origin.pull()
                else:
                    logger.info(f"Starting clone for {github_url} into {repo_dir}")
                    git.Repo.clone_from(github_url, repo_dir)

                repo.status = "cloned"
                await db.commit()
                logger.info(f"Successfully cloned/pulled {repo_name}")

            except Exception as e:
                logger.error(f"Failed to clone/pull repository {repo_id}: {str(e)}")
                if 'repo' in locals() and repo:
                    repo.status = "error"
                    await db.commit()