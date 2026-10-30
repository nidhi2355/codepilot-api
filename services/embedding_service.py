import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from api.database import SessionLocal
from api.models.repository import Repository, RepositoryFile

# Import the shared instances
from services.vector_store import qdrant, model, COLLECTION_NAME

logger = logging.getLogger(__name__)


# Ensure Qdrant collection exists
try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

class EmbeddingService:
    @staticmethod
    async def embed_repository_task(repo_id: int):
        """Background task to chunk code, generate embeddings, and store in Qdrant."""
        async with SessionLocal() as db:
            try:
                repo_result = await db.execute(select(Repository).filter(Repository.id == repo_id))
                repo_obj = repo_result.scalars().first()

                if not repo_obj or repo_obj.status != "indexed":
                    logger.error(f"Cannot embed repo {repo_id}: Must be in 'indexed' status.")
                    return

                repo_obj.status = "embedding"
                await db.commit()

                files_result = await db.execute(select(RepositoryFile).filter(RepositoryFile.repository_id == repo_id))
                repo_files = files_result.scalars().all()
                repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "repos", repo_obj.name))

                points = []
                point_id = 1 

                for file_record in repo_files:
                    abs_path = os.path.join(repo_dir, file_record.path)
                    if not os.path.exists(abs_path):
                        continue

                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Basic line-based chunking (~50 lines per chunk)
                    lines = content.split('\n')
                    chunk_size = 50
                    for i in range(0, len(lines), chunk_size):
                        chunk_text = '\n'.join(lines[i:i + chunk_size])
                        if not chunk_text.strip():
                            continue

                        vector = list(model.embed([chunk_text]))[0].tolist()

                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=vector,
                                payload={
                                    "repo_id": repo_id,
                                    "file_path": file_record.path,
                                    "content": chunk_text
                                }
                            )
                        )
                        point_id += 1

                if points:
                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    print(f"DEBUG: Successfully embedded {len(points)} chunks into Qdrant!")

                repo_obj.status = "ready"
                await db.commit()
                logger.info(f"Successfully generated vectors for {repo_obj.name}")

            except Exception as e:
                logger.error(f"Failed to embed repository {repo_id}: {str(e)}")
                if repo_obj:
                    repo_obj.status = "error"
                    await db.commit()