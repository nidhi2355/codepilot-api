from fastapi import APIRouter, HTTPException
from services.base_service import BaseService

router = APIRouter()

@router.get("/", tags=["Home"])
async def home_route():
    """Root endpoint for API verification."""
    response = BaseService.get_home_details()
    if "error" in response:
        raise HTTPException(status_code=500, detail=response["error"])
    return response

@router.get("/health", tags=["Health"])
async def health_check_route():
    """System health check endpoint."""
    response = BaseService.check_system_health()
    if response.get("status") == "unhealthy":
        raise HTTPException(status_code=503, detail=response)
    return response