import os
import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from api.database import get_db
from api.models.user import User
from api.dependencies.auth import create_access_token, get_current_user
from services.crypto_service import encrypt_token
from urllib.parse import urlencode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


class GitHubOAuthRequest(BaseModel):
    code: str


class UserPublic(BaseModel):
    id: int
    github_id: int
    username: str
    avatar_url: str | None = None
    email: str | None = None

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


@router.post("/github", response_model=AuthResponse)
async def github_oauth_callback(
    payload: GitHubOAuthRequest,
    db: AsyncSession = Depends(get_db),
):
    """Exchanges a GitHub OAuth authorization code for a user session JWT."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth credentials are not configured on the server",
        )

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_res = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            json={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": payload.code,
            },
        )
        token_data = token_res.json()

        github_token = token_data.get("access_token")
        if not github_token:
            err_desc = token_data.get("error_description") or token_data.get("error") or "OAuth exchange failed"
            logger.warning(f"GitHub OAuth token exchange failed: {err_desc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_desc,
            )

        user_res = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        if user_res.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to fetch GitHub user profile",
            )
        gh_user = user_res.json()

    github_id = gh_user["id"]
    username = gh_user["login"]
    avatar_url = gh_user.get("avatar_url")
    email = gh_user.get("email")
    encrypted = encrypt_token(github_token)

    result = await db.execute(select(User).filter(User.github_id == github_id))
    user = result.scalars().first()

    if user:
        user.username = username
        user.avatar_url = avatar_url
        if email:
            user.email = email
        user.encrypted_github_token = encrypted
    else:
        user = User(
            github_id=github_id,
            username=username,
            avatar_url=avatar_url,
            email=email,
            encrypted_github_token=encrypted,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)

    jwt_token = create_access_token(user.id)
    return AuthResponse(
        access_token=jwt_token,
        user=UserPublic.model_validate(user),
    )


@router.get("/me", response_model=UserPublic)
async def read_current_user(current_user: User = Depends(get_current_user)):
    """Returns the authenticated user's profile."""
    return current_user


@router.get("/github/login")
async def github_oauth_login(redirect_uri: str | None = None):
    """Constructs the GitHub OAuth authorization URL server-side."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GITHUB_CLIENT_ID is not configured on the server",
        )

    params = {
        "client_id": client_id,
        "scope": "repo read:user user:email",
    }
    if redirect_uri:
        params["redirect_uri"] = redirect_uri

    return {"url": f"https://github.com/login/oauth/authorize?{urlencode(params)}"}