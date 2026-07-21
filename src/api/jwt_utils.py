"""JWT helpers for HAIP auth.

Stateless sessions: on login we sign a token containing the username and
hospital. Protected endpoints verify the signature instead of hitting the DB.
"""
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SECRET_KEY = os.getenv("HAIP_JWT_SECRET", "haip-dev-secret-change-in-prod")
ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24

_bearer = HTTPBearer(auto_error=False)


def create_access_token(username: str, hospital: str) -> str:
    """Sign a JWT carrying the user identity and their hospital scope."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "hospital": hospital,
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Verify signature + expiry. Raises 401 on any failure."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """FastAPI dependency: returns {username, hospital} for a valid token."""
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    payload = decode_access_token(creds.credentials)
    return {"username": payload.get("sub"), "hospital": payload.get("hospital")}
