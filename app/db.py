from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, SUPABASE_SECRET_KEY


def get_client() -> Client:
    """Public client using publishable key (respects RLS)."""
    return create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)


def get_admin_client() -> Client:
    """Admin client using secret key (bypasses RLS)."""
    return create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
