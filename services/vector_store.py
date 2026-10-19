import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize local Qdrant centrally
QDRANT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "qdrant"))
os.makedirs(QDRANT_PATH, exist_ok=True)

# Single global instance
qdrant = QdrantClient(path=QDRANT_PATH)

# Single global model instance to save memory
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "code_chunks"