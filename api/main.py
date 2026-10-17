from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.database import engine, Base
from api.routes.base import router as base_router
from api.routes.repositories import router as repo_router
import logging

# Force Python to show INFO level logs in the terminal
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (For development only. Use Alembic for production migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Cleanup on shutdown
    await engine.dispose()

def create_app() -> FastAPI:
    app = FastAPI(
        title="CodePilot API",
        description="Agentic AI Software Engineer for Repository Understanding and Autonomous Coding",
        version="0.1.0",
        lifespan=lifespan
    )
    
    app.include_router(base_router)
    app.include_router(repo_router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)