# CodePilot API 🚀

**CodePilot** is an Agentic AI Software Engineer that autonomously ingests, understands, and reasons about GitHub repositories. Built on a Retrieval-Augmented Generation (RAG) pipeline, it lets developers query complex codebases in natural language.

The backend is **fully stateless and cloud-native**, and it runs on Render's Free Tier (512 MB RAM, ephemeral filesystem).

🌐 **Live API:** https://codepilot-api-q8d6.onrender.com/

📚 **Interactive docs (Swagger):** https://codepilot-api-q8d6.onrender.com/docs

❤️ **Health check:** https://codepilot-api-q8d6.onrender.com/health

> ⏳ **Note:** Render's free tier spins down after inactivity, so the first request after idle time can take a few seconds while the container wakes up.

---

## 🏗 Architecture & Tech Stack

| Component | Technology | Responsibility |
| --- | --- | --- |
| **API Runtime** | FastAPI + Uvicorn (Python 3.11) | Async routing, validation, orchestration |
| **Relational DB** | Supabase (PostgreSQL) via SQLAlchemy 2.0 (async) + asyncpg | Project metadata, indexing job status |
| **Vector DB** | Qdrant Cloud | Embedding storage and HNSW cosine similarity search |
| **Embeddings** | FastEmbed (ONNX Runtime), `BAAI/bge-small-en-v1.5` (384-dim) | Lightweight, CPU-optimized embedding generation |
| **AST Parsing** | Tree-sitter (`tree-sitter==0.21.3`, `tree-sitter-python==0.21.0`) | Structure-aware code chunking |
| **Repo Ingestion** | GitPython (shallow clones in `tempfile` directories) | Ephemeral, self-cleaning repository processing |
| **LLM Engine** | Groq API (`openai/gpt-oss-120b`) | Fast context-augmented answers |
| **Tooling** | Docker, DevPod, Pytest | Containerization, dev environment, testing |

### System Flow

```
Client ──▶ FastAPI (Render)
              │
              ├─▶ Git Service: shallow clone into temp dir (auto-deleted)
              │        └─▶ Tree-sitter: AST chunking (functions / classes)
              │                 └─▶ FastEmbed (ONNX): 384-dim embeddings
              │                          └─▶ Qdrant Cloud: vector storage
              │
              ├─▶ Supabase PostgreSQL (PgBouncer IPv4 pooler): metadata
              │
              └─▶ Groq LLM: answers grounded in retrieved code context
```

---

## ⚙️ Features

1. **Asynchronous Ingestion:** Non-blocking repository cloning with `GitPython`, using shallow clones (`--depth 1`, single branch, no tags or submodules) to keep disk usage minimal.
2. **Intelligent AST Chunking:** Tree-sitter splits code at natural boundaries (functions and classes) instead of naive text splitting.
3. **Hybrid Semantic Search:** Vector similarity in Qdrant combined with relational file metadata.
4. **Agentic QA:** Context-aware answers powered by high-speed Groq inference.
5. **Stateless by Design:** No persistent local files. Repositories are cloned into `tempfile` directories that are always purged, even on errors, and all state lives in Supabase and Qdrant Cloud.
6. **Free-Tier Optimized:** About 125 MB startup RAM, about 2 s cold start, and no PyTorch.

---

## 🧠 Engineering Highlights

| Challenge | Solution |
| --- | --- |
| **OOM crashes** (PyTorch + sentence-transformers used over 500 MB) | Replaced with FastEmbed + ONNX Runtime. Startup RAM went from about 480 MB to about 125 MB, and cold start from about 14 s to about 2 s. |
| **IPv6 failures** (`socket.gaierror`, Render has no IPv6 egress) | Switched to Supabase's IPv4 **PgBouncer transaction pooler** (port `6543`) and disabled asyncpg prepared statement caching. |
| **Ephemeral filesystem** (local Qdrant data lost on restart) | Migrated to **Qdrant Cloud** with a lazily initialized, env-validated client. |
| **Disk exhaustion** (full git history, `OS Error 28`) | Shallow clones inside `tempfile` context management, with guaranteed cleanup. |
| **Tree-sitter breaking changes** (0.22+ API changes) | Pinned `tree-sitter==0.21.3` and `tree-sitter-python==0.21.0`. |
| **Build failures** (Render defaulting to a bleeding-edge Python) | Enforced Python `3.11.8` via `.python-version` and `PYTHON_VERSION`, so prebuilt wheels are used and builds take under a minute. |
| **Health-check timeouts** (hardcoded port) | Bind to Render's dynamic `$PORT`. |

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python **3.11** (3.11.8 recommended; newer versions may lack prebuilt wheels)
- A [Supabase](https://supabase.com) project (PostgreSQL)
- A [Qdrant Cloud](https://cloud.qdrant.io) cluster
- A [Groq](https://console.groq.com) API key
- Optional: Docker/Podman

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/nidhi2355/codepilot-api.git
   cd codepilot-api
```

2. **Create a virtual environment and install dependencies**
```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
```

3. **Configure environment variables** by creating a `.env` file:
```ini
   # Supabase PostgreSQL: use the IPv4 transaction pooler URL (port 6543)
   DATABASE_URL=postgresql+asyncpg://user.project_ref:password@aws-0-<region>.pooler.supabase.com:6543/postgres

   # Qdrant Cloud
   QDRANT_CLOUD_URL=https://<your-cluster-id>.<region>.cloud.qdrant.io:6333
   QDRANT_CLOUD_API_KEY=your_qdrant_api_key_here
   QDRANT_COLLECTION=codepilot_codebase   # optional (default: codepilot_codebase)

   # Groq LLM
   GROQ_API_KEY=your_groq_api_key_here
```

   > `postgresql://` URLs are automatically rewritten to `postgresql+asyncpg://`.
   > Use the **pooler** connection string, not the direct `db.<ref>.supabase.co` one, which resolves to IPv6.

4. **Run the API**
```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Open the docs at **http://localhost:8000/docs**.

---

## 🔐 Environment Variables

| Variable | Required | Description |
| --- | :---: | --- |
| `DATABASE_URL` | ✅ | Supabase PostgreSQL connection string (pooler, port 6543) |
| `QDRANT_CLOUD_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_CLOUD_API_KEY` | ✅ | Qdrant Cloud API key |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `QDRANT_COLLECTION` | ❌ | Vector collection name (default: `codepilot_codebase`) |
| `PYTHON_VERSION` | Render only | Set to `3.11.8` |
| `PORT` | Render only | Injected automatically by Render |

---

## 📖 API Endpoints

Interactive docs: [`/docs`](https://codepilot-api-q8d6.onrender.com/docs) (or `http://localhost:8000/docs` locally).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/repositories/` | Register / ingest a new GitHub repository URL |
| `POST` | `/repositories/{id}/index` | Shallow-clone, parse code, and extract AST symbols |
| `POST` | `/repositories/{id}/embed` | Generate embeddings and store them in Qdrant Cloud |
| `POST` | `/repositories/{id}/ask` | Ask natural language questions about the codebase |

### Example Workflow

```bash
BASE_URL=https://codepilot-api-q8d6.onrender.com

# 1. Health check
curl $BASE_URL/health

# 2. Register a repository
curl -X POST $BASE_URL/repositories/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/<owner>/<repo>"}'

# 3. Index it
curl -X POST $BASE_URL/repositories/1/index

# 4. Generate embeddings
curl -X POST $BASE_URL/repositories/1/embed

# 5. Ask a question
curl -X POST $BASE_URL/repositories/1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is authentication handled?"}'
```

> Request and response schemas may vary slightly. Check `/docs` for the exact payloads.

---

## ☁️ Deployment (Render)

The project deploys to Render's free tier via `render.yaml`:

```yaml
services:
  - type: web
    name: codepilot-api
    env: python
    region: oregon
    plan: free
    buildCommand: "pip install --upgrade pip && pip install -r requirements.txt"
    startCommand: "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.8
      - key: DATABASE_URL
        sync: false
      - key: QDRANT_CLOUD_URL
        sync: false
      - key: QDRANT_CLOUD_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
```

**Deployment checklist**

1. Push the repo to GitHub and create a new **Web Service** on Render (or use the Blueprint with `render.yaml`).
2. Add `.python-version` containing `3.11.8`.
3. Set the secret environment variables in the Render dashboard (`sync: false` values are not stored in the repo).
4. Make sure `DATABASE_URL` uses the Supabase **IPv4 pooler** endpoint.
5. Deploy and verify at `/health`.

---

## 🗂 Project Structure

```
codepilot-api/
├── api/
│   ├── main.py                    # FastAPI app, routes, startup checks
│   ├── core/
│   │   └── database.py            # Async SQLAlchemy engine (PgBouncer-optimized)
│   └── services/
│       ├── git_service.py         # Ephemeral shallow-clone processor
│       ├── parser_service.py      # Tree-sitter AST chunking
│       ├── embedding_service.py   # FastEmbed ONNX singleton
│       └── vector_service.py      # Qdrant Cloud client manager
├── tests/                         # Async Pytest suite
├── requirements.txt
├── render.yaml
└── .python-version
```

> Adjust to match your actual repository layout.

---

## 🧪 Testing

Run the asynchronous test suite with Pytest:

```bash
python -m pytest -v
```

---

## 📌 Notes & Limitations

- **Language support:** AST chunking currently targets **Python** source files (`.py`).
- **Free-tier constraints:** 512 MB RAM and shared CPU, so very large repositories may need to be indexed in batches.
- **Cold starts:** The first request after idle time may be slow.
- **Dependency pinning:** `tree-sitter` and `tree-sitter-python` are intentionally pinned. Do not upgrade them without migrating the parser code.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
---

## 👤 Author

**Nidhi** · [GitHub](https://github.com/nidhi2355)
