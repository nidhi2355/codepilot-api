import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from api.models.repository import Repository
from api.schemas.repository import RepositoryCreate

logger = logging.getLogger(__name__)


class RepositoryService:
    @staticmethod
    async def create_repository(
        db: AsyncSession, repo_data: RepositoryCreate, user_id: int
    ) -> Repository:
        try:
            repo_url_str = str(repo_data.github_url).rstrip("/")
            repo_name = repo_url_str.split("/")[-1]

            result = await db.execute(
                select(Repository).filter(
                    Repository.user_id == user_id,
                    Repository.github_url == repo_url_str,
                )
            )
            existing_repo = result.scalars().first()
            if existing_repo:
                return existing_repo

            new_repo = Repository(
                user_id=user_id,
                github_url=repo_url_str,
                name=repo_name,
                status="pending",
            )
            db.add(new_repo)
            await db.commit()
            await db.refresh(new_repo)
            return new_repo
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating repository: {str(e)}")
            raise e

    @staticmethod
    async def get_repository(
        db: AsyncSession, repo_id: int, user_id: int | None = None
    ) -> Repository | None:
        """Fetches a single repository by ID, scoped to user_id when provided."""
        query = select(Repository).filter(Repository.id == repo_id)
        if user_id is not None:
            query = query.filter(Repository.user_id == user_id)
        result = await db.execute(query)
        return result.scalars().first()

    @staticmethod
    async def get_repositories(db: AsyncSession, user_id: int) -> list[Repository]:
        """Fetches all repositories owned by the authenticated user."""
        result = await db.execute(
            select(Repository)
            .filter(Repository.user_id == user_id)
            .order_by(Repository.created_at.desc())
        )
        return result.scalars().all()