import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from agent import build_answer_preview


def test_build_answer_preview_uses_gravity_knowledge():
    result = build_answer_preview("重力是怎么形成的？", grade="4", language="zh")
    assert result is not None
    assert result["fast_answer_confident"] is True
    assert "重力" in result["answer_short"]


def test_build_answer_preview_skips_low_confidence_questions():
    result = build_answer_preview("月亮为什么会发光？", grade="4", language="zh")
    assert result is None
