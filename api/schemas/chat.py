from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The natural language question to ask the agentic LLM about the codebase.", 
        json_schema_extra={"example": "Where is the Qdrant client initialized?"}
    )
    limit: int = Field(
        default=5, 
        ge=1, 
        le=10, 
        description="Number of vector context chunks to retrieve. Bounded between 1 and 10 to manage prompt context windows."
    )