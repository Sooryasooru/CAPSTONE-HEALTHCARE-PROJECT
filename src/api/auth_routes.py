"""Auth endpoints — thin HTTP wrappers over the existing rag.auth functions.

No new auth logic here: register() and verify_login() already handle
validation, password hashing, and duplicate checks. This just exposes them.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from rag.auth import register, verify_login
from api.jwt_utils import create_access_token, get_current_user
from fastapi import Depends

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    hospital_name: str
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register_admin(req: RegisterRequest):
    """Create a hospital admin. Returns success + message."""
    ok, message = register(req.hospital_name, req.username, req.password)
    return {"success": ok, "message": message}


@router.post("/login")
def login_admin(req: LoginRequest):
    """Verify credentials. Returns success + hospital name (or null)."""
    ok, hospital, role = verify_login(req.username, req.password)
    if ok:
        token = create_access_token(req.username, hospital, role)
        return {
            "success": True,
            "hospital": hospital,
            "role": role,
            "access_token": token,
            "token_type": "bearer",
        }
    return {"success": False, "hospital": None, "message": "Invalid username or password."}


@router.get("/me")
def whoami(user: dict = Depends(get_current_user)):
    """Return the identity carried by the caller's token."""
    return {"success": True, **user}
