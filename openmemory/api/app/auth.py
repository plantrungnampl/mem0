"""
Lightweight API-key authentication for OpenMemory.

Set the OPENMEMORY_API_KEY environment variable to enable protection.
When the variable is unset or empty, authentication is disabled (local development mode).
"""

import os
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

OPENMEMORY_API_KEY = os.environ.get("OPENMEMORY_API_KEY", "")
AUTH_ENABLED = bool(OPENMEMORY_API_KEY)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """Verify the request carries a valid API key.

    Skips verification when OPENMEMORY_API_KEY is not configured (local dev).
    """
    if not AUTH_ENABLED:
        return

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide an X-API-Key header.",
        )

    if not secrets.compare_digest(api_key, OPENMEMORY_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")
