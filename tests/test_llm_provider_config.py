import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_client_module():
    if "backend.openai_client" in sys.modules:
        return importlib.reload(sys.modules["backend.openai_client"])
    return importlib.import_module("backend.openai_client")


def test_build_openai_client_supports_minimax(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    client_module = load_client_module()
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)

    client = client_module.build_openai_client(RuntimeError)

    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "test-minimax-key"
    assert captured["base_url"] == "https://api.minimaxi.com/v1"
    assert client_module.get_llm_provider() == "minimax"
    assert client_module.llm_provider_configured() is True


def test_build_openai_client_supports_deepseek(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    client_module = load_client_module()
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)

    client = client_module.build_openai_client(RuntimeError)

    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "test-deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert client_module.get_llm_provider() == "deepseek"
    assert client_module.get_default_chat_model() == "deepseek-chat"
    assert client_module.llm_provider_configured() is True
