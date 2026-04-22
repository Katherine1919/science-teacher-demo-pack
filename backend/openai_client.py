import os
from typing import Type

from openai import OpenAI


def get_llm_provider() -> str:
    requested = os.getenv("LLM_PROVIDER", "").strip().lower()
    if requested in {"openai", "minimax", "deepseek"}:
        return requested
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    return "openai"


def llm_provider_configured() -> bool:
    provider = get_llm_provider()
    if provider == "deepseek":
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    if provider == "minimax":
        return bool(os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ORG_ID") or os.getenv("CODEX_ORG_ID"))


def get_default_chat_model() -> str:
    configured_model = os.getenv("CHAT_MODEL")
    if configured_model:
        return configured_model

    provider = get_llm_provider()
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    if provider == "minimax":
        return os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.5")
    return "gpt-4o-mini"


def build_openai_client(error_cls: Type[Exception] = RuntimeError) -> OpenAI:
    provider = get_llm_provider()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise error_cls("DEEPSEEK_API_KEY is not set")

        base_url = (
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.deepseek.com/v1"
        )
        client_kwargs = {"api_key": api_key, "base_url": base_url.rstrip("/")}
        return OpenAI(**client_kwargs)

    if provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise error_cls("MINIMAX_API_KEY is not set")

        base_url = (
            os.getenv("MINIMAX_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.minimaxi.com/v1"
        )
        client_kwargs = {"api_key": api_key, "base_url": base_url.rstrip("/")}
        return OpenAI(**client_kwargs)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        organization = os.getenv("OPENAI_ORG_ID") or os.getenv("CODEX_ORG_ID")
        if organization:
            client_kwargs = {"organization": organization}
        else:
            raise error_cls("OPENAI_API_KEY or OPENAI_ORG_ID is not set")
    else:
        client_kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

    codex_base_url = os.getenv("CODEX_BASE_URL")
    if codex_base_url:
        client_kwargs["base_url"] = codex_base_url.rstrip("/")

    return OpenAI(**client_kwargs)
