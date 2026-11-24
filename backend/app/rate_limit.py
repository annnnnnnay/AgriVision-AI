"""Rate limits for public prediction requests."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

settings = get_settings()


def prediction_rate_limit_key(request) -> str:
    identity = "authenticated" if request.headers.get("authorization") else "guest"
    return f"{identity}:{get_remote_address(request)}"


def prediction_rate_limit(key: str) -> str:
    if key.startswith("guest:"):
        return f"{settings.GUEST_ANALYSIS_LIMIT_PER_HOUR}/hour"
    return f"{settings.RATE_LIMIT_PER_MINUTE}/minute"


limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
)
