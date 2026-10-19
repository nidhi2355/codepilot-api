from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    limit: int = 5