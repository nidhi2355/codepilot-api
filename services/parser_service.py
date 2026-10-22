import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from api.database import SessionLocal
from api.models.repository import Repository, RepositoryFile, Symbol

logger = logging.getLogger(__name__)

# Initialize Tree-sitter for Python
PY_LANGUAGE = Language(tspython.language(), "python")
parser = Parser(PY_LANGUAGE)

# Query to find class and function names in Python
SYMBOL_QUERY = PY_LANGUAGE.query("""
    (class_definition name: (identifier) @class)
    (function_definition name: (identifier) @function)
""")

IGNORE_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}

class ParserService:
    @staticmethod
    async def index_repository_task(repo_id: int):
        """Background task to parse files and extract symbols."""
        async with SessionLocal() as db:
            try:
                # 1. Get repository path
                repo = await db.execute(select(Repository).filter(Repository.id == repo_id))
                repo_obj = repo.scalars().first()
                
                if not repo_obj or repo_obj.status != "cloned":
                    logger.error(f"Cannot index repo {repo_id}: Not found or not cloned.")
                    return

                # Update status
                repo_obj.status = "indexing"
                await db.commit()

                repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "repos", repo_obj.name))

                # 2. Walk directory and parse files
                for root, dirs, files in os.walk(repo_dir):
                    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS] # Ignore bad folders
                    
                    for file in files:
                        if file.endswith(".py"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, repo_dir)
                            await ParserService._process_file(db, repo_id, file_path, rel_path)
                
                # 3. Mark complete
                repo_obj.status = "indexed"
                await db.commit()
                logger.info(f"Successfully indexed repository {repo_obj.name}")

            except Exception as e:
                logger.error(f"Failed to index repository {repo_id}: {str(e)}")
                if repo_obj:
                    repo_obj.status = "error"
                    await db.commit()

    @staticmethod
    async def _process_file(db: AsyncSession, repo_id: int, abs_path: str, rel_path: str):
        """Reads a file, runs Tree-sitter, and saves symbols."""
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Save file record
            repo_file = RepositoryFile(repository_id=repo_id, path=rel_path, language="python")
            db.add(repo_file)
            await db.flush() # Get the file ID without fully committing

            # Parse with Tree-sitter
            tree = parser.parse(bytes(content, "utf8"))
            captures = SYMBOL_QUERY.captures(tree.root_node)

            # Save symbols
            for node, capture_name in captures:
                symbol = Symbol(
                    file_id=repo_file.id,
                    name=content[node.start_byte:node.end_byte],
                    type=capture_name, # "class" or "function"
                    start_line=node.start_point.row,
                    end_line=node.end_point.row
                )
                db.add(symbol)
                
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Error parsing file {rel_path}: {str(e)}")
