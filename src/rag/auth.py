"""
HAIP - Simple hospital-admin authentication (SQLite-backed).
==============================================================
Each hospital registers one admin account. Login ties the session
to that hospital_name so uploads are never free-typed / spoofable.

Deliberately minimal for capstone scope: one admin per hospital,
no roles, no password reset flow. Passwords are hashed (never
stored in plaintext) via werkzeug's PBKDF2 implementation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "auth.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hospital_admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hospital_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def register(hospital_name: str, username: str, password: str) -> tuple[bool, str]:
    """Register a new hospital admin. Returns (success, message)."""
    if not (hospital_name and username and password):
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    init_db()
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO hospital_admins (hospital_name, username, password_hash) "
                "VALUES (?, ?, ?)",
                (hospital_name.strip(), username.strip(), generate_password_hash(password)),
            )
        return True, f"Registered admin for '{hospital_name.strip()}'."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def verify_login(username: str, password: str) -> tuple[bool, str | None]:
    """Check credentials. Returns (success, hospital_name_or_None)."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT hospital_name, password_hash FROM hospital_admins WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row and check_password_hash(row[1], password):
        return True, row[0]
    return False, None
