from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from services.vector_store import qdrant, model, COLLECTION_NAME

class SearchService:
    @staticmethod
    def search_codebase(repo_id: int, query: str, limit: int = 5):
        """Embeds the query and searches Qdrant for matching code chunks."""
        # Convert the user query into a vector
        query_vector = model.encode(query).tolist()
        
        # Search Qdrant, filtering strictly by the requested repository ID
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_id",
                        match=MatchValue(value=repo_id)
                    )
                ]
            ),
            limit=limit
        )
        
        # Format and return the payload data
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "file_path": hit.payload.get("file_path"),
                "content": hit.payload.get("content")
            })
        return results