from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from services.vector_store import qdrant, model, COLLECTION_NAME


class SearchService:
    @staticmethod
    def search_codebase(repo_id: int, query: str, limit: int = 5):
        query_vector = list(model.embed([query]))[0].tolist()
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="repo_id",
                    match=MatchValue(value=repo_id)
                )
            ]
        )

        if hasattr(qdrant, "query_points"):
            search_result = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points
        else:
            search_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )

        return [
            {
                "score": hit.score,
                "file_path": hit.payload.get("file_path"),
                "content": hit.payload.get("content")
            }
            for hit in search_result
        ]