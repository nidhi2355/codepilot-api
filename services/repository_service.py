import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from api.models.repository import Repository
from api.schemas.repository import RepositoryCreate

logger = logging.getLogger(__name__)

class RepositoryService:
    @staticmethod
    async def create_repository(db: AsyncSession, repo_data: RepositoryCreate) -> Repository:
        try:
            # Extract repo name from URL (e.g., https://github.com/user/repo -> repo)
            repo_url_str = str(repo_data.github_url).rstrip("/")
            repo_name = repo_url_str.split("/")[-1]

            # Check if it already exists
            result = await db.execute(select(Repository).filter(Repository.github_url == repo_url_str))
            existing_repo = result.scalars().first()
            if existing_repo:
                return existing_repo

            # Create new record
            new_repo = Repository(
                github_url=repo_url_str,
                name=repo_name,
                status="pending"
            )
            db.add(new_repo)
            await db.commit()
            await db.refresh(new_repo)
            
            return new_repo
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating repository: {str(e)}")
            raise e