# CodePilot API: Architecture Decision Records (ADR)

This document tracks the core architectural decisions, technical trade-offs, and debugging milestones for the CodePilot API. It serves as both a historical record for future development and a study guide for technical interviews.

---

## 1. Core Framework & Async Architecture
**Context:** The API needs to handle heavy I/O bound operations (cloning repositories, reading large files, querying databases) without blocking incoming HTTP requests.
**Decision:** We chose **FastAPI** paired with **Async SQLAlchemy** and `aiosqlite`.
**Rationale:** 
* FastAPI provides native asynchronous support (`async def`).
* By using async database sessions and background tasks, the main thread remains free to accept new API requests while Git cloning or LLM embedding happens in the background.
* **Interview Talking Point:** If asked about scaling, you can discuss how the async event loop prevents the server from locking up during the `git clone` phase, maximizing concurrency.

## 2. Abstract Syntax Tree (AST) vs. Regex for Parsing
**Context:** We needed a way to extract meaningful symbols (functions, classes) from raw Python files.
**Decision:** We selected **Tree-sitter** over Regular Expressions or Python's native `ast` module.
**Rationale:**
* Regex is notoriously fragile for parsing nested code blocks.
* While Python's `ast` is good, Tree-sitter is *language-agnostic*. This decision future-proofs the API to easily support JavaScript, TypeScript, or Go later by simply swapping the Tree-sitter grammar.
* **Challenge Overcome:** Tree-sitter returns node captures as a list of tuples, not a dictionary. We encountered an `AttributeError` when trying to call `.items()` on it, which silently failed the parsing step. Fixing this taught us the importance of strictly validating third-party library data structures.

## 3. Local Vector Database (Qdrant) vs Cloud (Pinecone/Milvus)
**Context:** We needed a vector database to store and semantically search code embeddings.
**Decision:** We chose **Qdrant (Local Mode)** over hosted solutions like Pinecone or heavy local clusters like Milvus.
**Rationale:**
* Qdrant runs entirely within the Python process via SQLite/file-storage, meaning zero external network latency and zero cost during the MVP phase.
* **Challenge Overcome (Crucial Interview Point):** We encountered a `Resource temporarily unavailable [Errno 11]` lock error. Qdrant local mode only allows a *single active connection*. Because we instantiated Qdrant separately in both `embedding_service.py` and `search_service.py`, Uvicorn crashed.
* **Solution:** We implemented the **Singleton Pattern** by creating a centralized `vector_store.py` file to hold the single global instance of the Qdrant client, which all other services import.

## 4. Embedding Model Selection
**Context:** We needed to convert code chunks into high-dimensional vectors.
**Decision:** We chose the `all-MiniLM-L6-v2` model via the `sentence-transformers` library.
**Rationale:**
* It is highly optimized for semantic search and has a very small memory footprint (~80MB), allowing it to run efficiently on the CPU inside our DevPod container without requiring expensive GPUs.
* Produces 384-dimensional vectors, which are faster to query than the larger 1536-dimensional vectors from OpenAI.

## 5. Development Environment & Memory Management
**Context:** Running a FastAPI dev server inside a containerized environment (DevPod).
**Decision:** Disabling aggressive file watchers for heavy data directories.
**Rationale:**
* **Challenge Overcome:** Uvicorn continuously crashed with `Cannot allocate memory [Errno 12]`. This happened because Uvicorn's file watcher (using `inotify`) attempted to recursively scan the massive `.git` folders and the Qdrant `data/` directory, immediately exhausting the container's OS-level memory limits.
* **Solution:** We optimized the startup command using `--reload-dir api --reload-dir services` to strictly limit the watcher. Ultimately, as the database grew, we dropped the `--reload` flag entirely for stability. This demonstrates a deep understanding of how containerized Linux environments handle file descriptors.

## 6. Data Staleness and Git Syncing
**Context:** Keeping the local code chunks in sync with the remote GitHub repository.
**Decision:** Implementing idempotent Git operations (`git pull` vs `git clone`).
**Rationale:**
* **Challenge Overcome:** If the repository was already cloned, re-running the pipeline wouldn't capture newly pushed code. 
* **Solution:** Upgraded `git_service.py` to check for the existence of the `.git` directory. If it exists, it executes `git pull` to fetch the delta; if not, it does a full `git clone`. This ensures the RAG (Retrieval-Augmented Generation) context is never stale.

## 7. Hybrid Retrieval Strategy (Top-K)
**Context:** Fetching the right code context for the LLM.
**Decision:** Returning an array of top results (Limit = 3 to 5) rather than just the highest-scoring chunk.
**Rationale:**
* Semantic search maps meaning, not exact keywords. Sometimes the exact answer (e.g., `api/database.py`) scores slightly lower than a file that heavily discusses the topic (e.g., `services/git_service.py`). By passing the "Top-K" chunks to the LLM, we give the AI enough surrounding context to deduce the perfect answer.

### 8. Agentic API & LLM Integration (Groq & Llama 3.1)
* **Decision**: Integrated the Groq API using `llama-3.1-8b-instant` for the final RAG generation step.
* **Reasoning**: While earlier phases focused on strict local execution (SQLite, Qdrant, CPU-based `sentence-transformers`), running a local generation LLM inside a DevPod container would severely throttle memory and CPU. Groq provides near-instant inference, making the RAG pipeline highly responsive.
* **Trade-off**: This introduces an external API dependency, but the speed and reliability gains for code generation heavily outweigh the memory constraints of containerized local development.