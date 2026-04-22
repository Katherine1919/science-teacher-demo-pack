import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_agent_module():
    if "backend.agent" in sys.modules:
        return importlib.reload(sys.modules["backend.agent"])
    return importlib.import_module("backend.agent")


def test_extract_json_object_handles_json_prefix_and_fences():
    agent_module = load_agent_module()

    raw = """json
```json
{
  "question_type": "concept",
  "language": "zh",
  "answer_short": "雨来自云里的小水滴。",
  "answer_full": ["空气中的水蒸气会上升。", "遇冷后会变成小水滴。", "小水滴变重后会落下来。"],
  "follow_up": "你见过乌云吗？",
  "show_image": "下雨过程图",
  "safety_note": ""
}
```"""

    payload = agent_module.extract_json_object(raw)

    assert payload["answer_short"] == "雨来自云里的小水滴。"
    assert payload["show_image"] == "下雨过程图"


def test_normalize_formatted_answer_limits_length_and_show_image():
    agent_module = load_agent_module()

    payload = agent_module.normalize_formatted_answer(
        {
            "answer_short": "这是总结。",
            "answer_full": ["第一点", "第二点", "第三点", "第四点"],
            "follow_up": "接下来你想观察什么？",
            "show_image": "水循环示意图，带云、雨、蒸发和河流的完整讲解长句",
        },
        question_type="concept",
        language="zh",
    )

    assert payload["answer_full"] == ["第一点", "第二点", "第三点"]
    assert payload["show_image"] == "水循环示意图"


def test_extract_json_object_recovers_truncated_schema_output():
    agent_module = load_agent_module()

    raw = """{
  "question_type": "concept",
  "language": "zh",
  "answer_short": "雨来自云里的小水滴。",
  "answer_full": ["空气中的水蒸气会上升。", "遇冷后会变成小水滴。", "小水滴变重后会落下来。"],
  "follow_up": "你见过乌云吗？",
"""

    payload = agent_module.extract_json_object(raw)

    assert payload["question_type"] == "concept"
    assert payload["answer_full"] == ["空气中的水蒸气会上升。", "遇冷后会变成小水滴。", "小水滴变重后会落下来。"]
    assert payload["follow_up"] == "你见过乌云吗？"


def test_build_spoken_answer_includes_full_answer_and_follow_up():
    agent_module = load_agent_module()

    spoken = agent_module.build_spoken_answer(
        "重力是地球把我们拉向它的力量。",
        [
            "你跳起来后会落回地面。",
            "苹果会从树上掉下来。",
            "书本不会飘在空中。",
        ],
        "如果没有重力会怎样呢？",
    )

    assert spoken == "\n".join(
        [
            "重力是地球把我们拉向它的力量。",
            "你跳起来后会落回地面。",
            "苹果会从树上掉下来。",
            "书本不会飘在空中。",
            "如果没有重力会怎样呢？",
        ]
    )


def test_heuristic_classify_question_prefers_concept_over_greeting():
    agent_module = load_agent_module()

    result = agent_module.heuristic_classify_question("你好，请问什么是重力？")

    assert result == "concept"


def test_build_fast_local_answer_from_context():
    agent_module = load_agent_module()

    payload = agent_module.build_fast_local_answer(
        "concept",
        "zh",
        [
            {
                "text": "# 四年级 水循环\n水循环是地球上的水不断蒸发、凝结和降水的过程。\n太阳晒过以后，地上的水会慢慢蒸发成水蒸气。\n水蒸气遇冷后会变成小水滴。",
                "tags": ["core", "水循环"],
            }
        ],
    )

    assert payload is not None
    assert payload["answer_short"] == "四年级 水循环 水循环是地球上的水不断蒸发、凝结和降水的过程。"
    assert payload["answer_full"] == [
        "太阳晒过以后，地上的水会慢慢蒸发成水蒸气。",
        "水蒸气遇冷后会变成小水滴。",
    ]
    assert payload["show_image"] == "水循环"
