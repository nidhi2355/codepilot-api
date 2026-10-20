import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

# We use pytest-asyncio to handle async API calls in our tests
pytestmark = pytest.mark.asyncio

async def test_health_check():
    """Test if the base API is up and running."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    
    assert response.status_code == 200
    assert response.json() == {
            "name": "CodePilot API",
            "status": "online",
            "message": "Welcome to CodePilot - From Issue to Pull Request"
        }

async def test_invalid_repository_url():
    """Test the repository ingestion with an invalid payload."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Missing the github_url field
        response = await ac.post("/repositories/", json={"wrong_field": "test"})
    
    assert response.status_code == 422 # FastAPI's default validation error