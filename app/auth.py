"""Authentication helpers using Supabase Auth."""
from __future__ import annotations

from fastapi import Request, Response
from supabase import Client

from app.db import get_client

ACCESS_TOKEN_COOKIE = "sb_access_token"
REFRESH_TOKEN_COOKIE = "sb_refresh_token"


def set_auth_cookies(response: Response, session: dict) -> None:
    """Set access and refresh token cookies from a Supabase session."""
    response.set_cookie(
        ACCESS_TOKEN_COOKIE,
        session["access_token"],
        httponly=True,
        samesite="lax",
        max_age=session.get("expires_in", 3600),
    )
    response.set_cookie(
        REFRESH_TOKEN_COOKIE,
        session["refresh_token"],
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,  # 30 days
    )


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_TOKEN_COOKIE)
    response.delete_cookie(REFRESH_TOKEN_COOKIE)


def get_current_user(request: Request) -> dict | None:
    """Get the current user from cookies. Returns user dict or None."""
    access_token = request.cookies.get(ACCESS_TOKEN_COOKIE)
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)

    if not access_token:
        return None

    try:
        client: Client = get_client()
        resp = client.auth.set_session(access_token, refresh_token)
        return resp.user.model_dump() if resp.user else None
    except Exception:
        return None
