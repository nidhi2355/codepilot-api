import os
import logging
from git import Repo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from api.database import SessionLocal
from api.models.repository import Repository, RepositoryFile
from services.vector_store import qdrant, COLLECTION_NAME
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

REPOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "repos"))
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

                    # 1. Clear old files and symbols from SQLite 
                    await db.execute(delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id))
                    
                    # 2. Clear old vectors from Qdrant
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