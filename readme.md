# CodePilot API 🚀

CodePilot is an Agentic AI Software Engineer designed to autonomously ingest, understand, and reason about GitHub repositories. Built with a robust Retrieval-Augmented Generation (RAG) pipeline, it allows developers to query complex codebases using natural language.

## 🏗 Architecture & Tech Stack

*   **Backend:** FastAPI, Python 3.11, SQLAlchemy (Async)
*   **Database:** SQLite (Relational State), Qdrant (Vector Store)
*   **AST Parsing:** Tree-sitter (Python) for precise structural extraction
*   **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **LLM Engine:** Groq API (`openai/gpt-oss-120b`)
*   **Environment:** DevPod, Docker, Pytest

## ⚙️ Features

1.  **Asynchronous Ingestion:** Non-blocking GitHub repository cloning via `GitPython`.
2.  **Intelligent Chunking:** AST-based code chunking using Tree-sitter instead of naive text splitting.
3.  **Hybrid Semantic Search:** Local vector generation mapped against relational file metadata.
4.  **Agentic QA:** Context-aware responses powered by high-speed LLM inference.

## 🚀 Quick Start

### Prerequisites

Ensure you have Docker/Podman and Python 3.11+ installed.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nidhi2355/codepilot-api.git
    cd codepilot-api
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables (`.env`):**
    ```ini
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Run the API:**
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    ```

## 📖 API Endpoints

Access the interactive Swagger UI at `http://localhost:8000/docs`.

*   `POST /repositories/`: Ingest a new GitHub URL.
*   `POST /repositories/{id}/index`: Parse code and extract AST symbols.
*   `POST /repositories/{id}/embed`: Generate vector embeddings in Qdrant.
*   `POST /repositories/{id}/ask`: Ask natural language questions about the codebase.

## 🧪 Testing

Run the asynchronous test suite using Pytest:

```bash
python -m pytest -v
```
