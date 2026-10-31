import os
import uuid
import tempfile
import shutil
import logging
from git import Repo
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from api.database import SessionLocal
from api.models.repository import Repository

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
        async with SessionLocal() as db:
            try:
                repo_result = await db.execute(select(Repository).filter(Repository.id == repo_id))
                repo_obj = repo_result.scalars().first()

                if not repo_obj:
                    logger.error(f"Cannot embed repo {repo_id}: Not found.")
                    return

                repo_obj.status = "embedding"
                await db.commit()

                # 1. Create a stateless temporary directory
                temp_dir = tempfile.mkdtemp()
                try:
                    logger.info(f"Cloning {repo_obj.github_url} into {temp_dir}")
                    # 2. Clone the repo explicitly so the files actually exist to be read
                    Repo.clone_from(repo_obj.github_url, temp_dir, depth=1)

                    points = []
                    
                    # Bypass the DB and read the cloned files directly from the disk
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            # You can add more extensions here if needed (e.g., .js, .ts)
                            if not file.endswith(".py"):
                                continue

                            abs_path = os.path.join(root, file)
                            relative_path = os.path.relpath(abs_path, temp_dir)

                            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
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
                                        id=str(uuid.uuid4()),  # 3. Use UUIDs to prevent vector overwriting!
                                        vector=vector,
                                        payload={
                                            "repo_id": repo_id,
                                            "file_path": relative_path,
                                            "content": chunk_text
                                        }
                                    )
                                )

                    if points:
                        qdrant.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points
                        )
                        logger.info(f"Successfully embedded {len(points)} chunks into Qdrant!")
                    else:
                        logger.warning("No points were generated! The repository might be empty or missing .py files.")

                    repo_obj.status = "ready"
                    await db.commit()
                    logger.info(f"Successfully generated vectors for {repo_obj.name}")

                finally:
                    # 4. Guarantee cleanup of the ephemeral drive
                    shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                logger.error(f"Failed to embed repository {repo_id}: {str(e)}")
                if repo_obj:
                    repo_obj.status = "error"
                    await db.commit()