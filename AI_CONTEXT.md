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
- [x] Created `/` (home) and `/health` endpoints.
- [x] Configured DevPod/Podman development environment.
- [x] Set up async SQLAlchemy with SQLite for local development.
- [x] Created database models and Pydantic schemas for Repositories.
- [x] Implemented `POST /repositories` endpoint with asynchronous `GitPython` cloning.
- [x] Implemented `GET /repositories/{id}` and `GET /repositories` for status polling.
- [x] Implemented Phase 3 Tree-sitter Code Parsing for Python files.
- [x] Created `POST /repositories/{id}/index` to trigger background AST extraction.

## Current Status & Next Action
* **Status**: The API can now clone repositories and run an AST parser over Python files to extract structural symbols into the database.
* **Next Step**: Begin Phase 4 (Code Knowledge Base) by generating text chunks and preparing the vector embedding pipeline for Qdrant.