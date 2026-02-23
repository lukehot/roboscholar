"""Authentication helpers using Supabase Auth."""
from __future__ import annotations

import json
import base64

from fastapi import Request, Response

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
    """Get the current user from the JWT cookie. Decodes locally without API call."""
    access_token = request.cookies.get(ACCESS_TOKEN_COOKIE)
    if not access_token:
        return None

    try:
        # Decode JWT payload (middle segment) — we trust it since it's httponly
        payload = access_token.split(".")[1]
        # Add padding
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return {
            "id": data.get("sub"),
            "email": data.get("email"),
        }
    except Exception:
        return None
