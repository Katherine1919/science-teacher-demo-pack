# tests/test_agent_output.py

import os
import pytest

REQUIRED_KEYS = {
    "question_type",
    "language",
    "answer_short",
    "answer_full",
    "follow_up",
    "show_image",
    "safety_note",
}


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY") or os.getenv("DEEPSEEK_API_KEY")),
    reason="No OpenAI, MiniMax, or DeepSeek API key set",
)
def test_agent_output_schema():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))

    from agent import run_agent

    result = run_agent("为什么会下雨？", grade="4", language="zh")
    assert isinstance(result, dict)

    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    assert isinstance(result["question_type"], str)
    assert isinstance(result["language"], str)
    assert isinstance(result["answer_short"], str)
    assert isinstance(result["answer_full"], list)
    assert isinstance(result["follow_up"], str)
    assert isinstance(result["show_image"], str)
    assert isinstance(result["safety_note"], str)
    assert len(result["answer_full"]) >= 1
