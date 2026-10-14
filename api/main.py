from fastapi import FastAPI
from api.routes.base import router as base_router

def create_app() -> FastAPI:
    """Factory function to initialize the FastAPI application."""
    app = FastAPI(
        title="CodePilot API",
        description="Agentic AI Software Engineer for Repository Understanding and Autonomous Coding",
        version="0.1.0"
    )
    
    # Include modular routes
    app.include_router(base_router)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Uvicorn serves as the ASGI server[cite: 10]
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)