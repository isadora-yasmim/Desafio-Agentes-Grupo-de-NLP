from supabase import Client, create_client

from core.config import settings


def get_supabase_client() -> Client:
    if not settings.SUPABASE_URL:
        raise ValueError("SUPABASE_URL não encontrada no .env")

    if not settings.SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_SERVICE_KEY não encontrada no .env")

    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY,
    )