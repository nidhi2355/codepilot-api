# CodePilot AI Context & Session State

## Project Overview
CodePilot is an Agentic AI Software Engineer designed to autonomously understand, debug, and modify software repositories.
* **Objective**: Transform a natural language issue into a tested Pull Request.
* **Current Phase**: Phase 1 & 2 - Foundation API & DevPod Setup.

## Tech Stack & Architecture
* **Backend**: FastAPI (Python 3.11), Uvicorn.
* **Containerization**: DevPod with Podman backend (using `.devcontainer/devcontainer.json`).
* **Future Integrations**: LangGraph, Qdrant, PostgreSQL, Redis, GitHub API.

## Key Decisions Log
* **API Framework**: Chosen FastAPI over Litestar/Flask for its native async support, Pydantic validation, and suitability for AI microservices.
* **Containerization**: Utilizing DevPod with Podman for local isolated development to ensure strict environment consistency before deploying to production.
* **Modular Routing**: Separated business logic (`services/base_service.py`) from API routes (`api/routes/base.py`) to keep the codebase maintainable as agents are introduced.

## Completed Steps
- [x] Defined initial project requirements and phase plan.
- [x] Initialized modular FastAPI application structure.
- [x] Configured DevPod/Podman development environment.
- [x] Set up async SQLAlchemy with SQLite for local development.
- [x] Implemented Phase 2 (Ingestion) with async `GitPython` cloning.
- [x] Implemented Phase 3 (Parsing) with Tree-sitter AST extraction.
- [x] Implemented Phase 4 (Vectorization) with local Qdrant.
- [x] Implemented Phase 5 (Hybrid Retrieval) with a semantic search endpoint.
- [x] Implemented Phase 6 (Agentic API) using Groq.
- [x] Implemented Phase 7.1 (Unit Testing) with `pytest`.
- [x] Implemented Phase 7.2 (Error Handling) with global FastAPI exception catchers.
- [x] Implemented Phase 7.3 (API Documentation Polish) via Pydantic Field metadata.
- [x] Implemented Final Phase (README & GitHub Push).
- [x] Implemented Phase 7.1 (Agent State & Supervisor Structure) by initializing the LangGraph `StateGraph` and `TypedDict` memory state.
- [x] Implemented Phase 7.2 (Planner Agent) using `langchain-groq` and strict Pydantic structured outputs to generate architectural plans.
- [x] Updated `requirements.txt` with LangGraph and LangChain dependencies.

## Current Status
* **Status**: 🟡 **IN PROGRESS (Phase 7 - LangGraph Orchestration)**. Transitioning from isolated retrieval API endpoints to stateful multi-agent execution.