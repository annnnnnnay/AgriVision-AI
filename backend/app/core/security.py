"""
Security utilities — JWT tokens, password hashing, auth dependencies.
Uses PyJWT for proper JWT signing and verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.exceptions import UnauthorizedException

settings = get_settings()
security_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[dict]:
    """Validate a legacy app token or a Supabase Auth access token."""
    if not credentials:
        return None

    token = credentials.credentials
    payload = decode_access_token(token)
    if payload:
        return payload

    supabase_service = getattr(request.app.state, "supabase_service", None)
    if supabase_service is not None:
        user = await run_in_threadpool(
            supabase_service.get_authenticated_user, token
        )
        if user:
            return user

    raise UnauthorizedException("Invalid or expired authentication token.")


async def get_current_user(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
) -> dict:
    """Extract user from token — raises 401 if not authenticated."""
    supabase_service = getattr(request.app.state, "supabase_service", None)
    if (
        not user
        and settings.LOCAL_DEMO_MODE
        and not getattr(supabase_service, "auth_enabled", False)
    ):
        return {"sub": "local-demo", "email": "local@agrivision.local"}
    if not user:
        raise UnauthorizedException("Authentication required. Please log in.")
    return user
