import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize Qdrant Client for Cloud
qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY")
) 

# Single global model instance to save memory
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "code_chunks"