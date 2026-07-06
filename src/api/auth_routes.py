"""Auth endpoints — thin HTTP wrappers over the existing rag.auth functions.

No new auth logic here: register() and verify_login() already handle
validation, password hashing, and duplicate checks. This just exposes them.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from rag.auth import register, verify_login

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
    ok, hospital = verify_login(req.username, req.password)
    if ok:
        return {"success": True, "hospital": hospital}
    return {"success": False, "hospital": None, "message": "Invalid username or password."}
