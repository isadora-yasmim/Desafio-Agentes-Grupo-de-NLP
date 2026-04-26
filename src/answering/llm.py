from __future__ import annotations

from langchain_openai import ChatOpenAI

from core.config import settings


def build_llm():
    return ChatOpenAI(
        model=getattr(settings, "OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
    )