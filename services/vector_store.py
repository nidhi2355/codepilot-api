import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY")
) 

COLLECTION_NAME = "code_chunks"

# Create the required integer index for the repo_id field
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="repo_id",
    field_schema=PayloadSchemaType.INTEGER
)

# Lightweight ONNX runner, no PyTorch needed
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
