import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.database import engine, Base
from api.models import user as _user_model  # noqa: F401
from api.routes.base import router as base_router
from api.routes.repositories import router as repo_router
from api.routes import agent, auth

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CodePilot API",
        description="Agentic AI Software Engineer for Repository Understanding and Autonomous Coding",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "Client Error", "message": exc.detail},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.error(f"Unhandled Exception on {request.url}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred on the server.",
            },
        )

    app.include_router(base_router)
    app.include_router(auth.router)
    app.include_router(repo_router)
    app.include_router(agent.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)