# tests/test_retriever.py

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))

from retriever import retrieve_context


def test_retriever_returns_list():
    result = retrieve_context("为什么会下雨？", grade="4", top_k=3)
    assert isinstance(result, list)


def test_retriever_prefers_gravity_content_for_gravity_question():
    result = retrieve_context("重力是怎么形成的？", grade="4", top_k=3)
    assert result
    assert result[0]["topic"] == "grade4_gravity"
    assert result[0]["retrieval_confident"] is True


def test_retriever_keeps_shadow_question_on_shadow_topic():
    result = retrieve_context("影子是怎么形成的？", grade="4", top_k=3)
    assert result
    assert result[0]["topic"] in {"grade4_light_shadow", "kids_science_shadows"}
    assert result[0]["retrieval_confident"] is True
