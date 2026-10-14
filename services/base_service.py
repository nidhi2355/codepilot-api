import time
import platform
import logging

logger = logging.getLogger(__name__)

class BaseService:
    @staticmethod
    def get_home_details() -> dict:
        """Returns foundational API information."""
        try:
            return {
                "name": "CodePilot API",
                "status": "online",
                "message": "Welcome to CodePilot - From Issue to Pull Request"
            }
        except Exception as e:
            logger.error(f"Error fetching home details: {str(e)}")
            return {"error": "Internal server error"}

    @staticmethod
    def check_system_health() -> dict:
        """Evaluates basic system health metrics."""
        try:
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "system": platform.system(),
                "python_version": platform.python_version()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }