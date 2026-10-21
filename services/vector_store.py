import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY")
) 

# Lightweight ONNX runner, no PyTorch needed
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "code_chunks"