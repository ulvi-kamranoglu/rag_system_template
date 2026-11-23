"""Optional service-to-service authentication for answer endpoints."""
import os
from secrets import compare_digest

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(key_header)) -> None:
    expected = os.environ.get("ASSISTANT_API_KEY", "")
    if expected and (not provided or not compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_admin_key(provided: str | None = Security(key_header)) -> None:
    """Admin routes are disabled unless a separate administrative key is configured."""
    expected = os.environ.get("ASSISTANT_ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Pipeline administration is disabled")
    if not provided or not compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
