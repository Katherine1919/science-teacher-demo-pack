import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_app_module():
    if "backend.app" in sys.modules:
        return importlib.reload(sys.modules["backend.app"])
    return importlib.import_module("backend.app")


def test_demo_trigger_offline_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app_module = load_app_module()

    question_audio = tmp_path / "question.wav"
    latest_json = tmp_path / "answer.json"

    def fake_record_question():
        question_audio.write_bytes(b"RIFFdemo")
        return question_audio

    monkeypatch.setattr(app_module, "LATEST_JSON", latest_json)
    monkeypatch.setattr(app_module, "QUESTION_AUDIO", question_audio)
    monkeypatch.setattr(app_module, "record_question", fake_record_question)

    client = TestClient(app_module.app)

    response = client.post("/demo/trigger")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ok"] is True
    assert payload["demo_mode"] == "offline"
    assert payload["audio_available"] is False

    latest = client.get("/latest/answer.json")
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["demo_mode"] == "offline"
    assert latest_payload["question"] == "（已录音，未转写）"
    assert latest_payload["question_audio_saved"] is True
    assert "DASHSCOPE_API_KEY" in latest_payload["tts_error"]


def test_demo_trigger_uses_local_audio_playback(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")

    app_module = load_app_module()

    question_audio = tmp_path / "question.wav"
    latest_json = tmp_path / "answer.json"
    latest_audio = tmp_path / "answer.mp3"

    def fake_record_question():
        question_audio.write_bytes(b"RIFFdemo")
        return question_audio

    class FakeProcess:
        def wait(self):
            return 0

    monkeypatch.setattr(app_module, "LATEST_JSON", latest_json)
    monkeypatch.setattr(app_module, "LATEST_AUDIO", latest_audio)
    monkeypatch.setattr(app_module, "QUESTION_AUDIO", question_audio)
    monkeypatch.setattr(app_module, "record_question", fake_record_question)
    monkeypatch.setattr(app_module, "stop_local_audio_playback", lambda: None)
    monkeypatch.setattr(app_module, "transcribe_question", lambda: "雨是怎么形成的？")
    monkeypatch.setattr(
        app_module,
        "run_agent",
        lambda question: {
            "question_type": "concept",
            "language": "zh",
            "answer_short": "雨是水蒸气冷却后形成的小水滴落下来。",
            "answer_full": ["空气中的水蒸气遇冷会凝结成小水滴。", "小水滴变多变重后会落下来形成雨。"],
            "follow_up": "你见过云吗？",
            "show_image": "",
            "safety_note": "",
            "retrieved_chunks": [],
            "mode": "science",
            "grade": "auto",
            "spoken_answer": "雨是空气里的水蒸气遇冷后形成的小水滴落下来。",
        },
    )
    monkeypatch.setattr(app_module, "generate_answer_audio", lambda *args, **kwargs: latest_audio.write_bytes(b"ID3demo"))
    monkeypatch.setattr(app_module, "start_local_audio_playback", lambda *_args, **_kwargs: FakeProcess())

    client = TestClient(app_module.app)

    response = client.post("/demo/trigger")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ok"] is True
    assert payload["audio_available"] is True
    assert payload["local_audio_playback"] is True

    latest = client.get("/latest/answer.json")
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["local_audio_playback"] is True
    assert latest_payload["estimated_answer_playback_ms"] >= 2200


def test_demo_trigger_handles_no_speech_as_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")

    app_module = load_app_module()

    question_audio = tmp_path / "question.wav"
    latest_json = tmp_path / "answer.json"

    def fake_record_question():
        question_audio.write_bytes(b"RIFFdemo")
        return question_audio

    monkeypatch.setattr(app_module, "LATEST_JSON", latest_json)
    monkeypatch.setattr(app_module, "QUESTION_AUDIO", question_audio)
    monkeypatch.setattr(app_module, "record_question", fake_record_question)

    def raise_no_speech():
        raise app_module.ASRNoSpeechError("Aliyun ASR detected no speech")

    monkeypatch.setattr(app_module, "transcribe_question", raise_no_speech)
    monkeypatch.setattr(app_module, "stop_local_audio_playback", lambda: None)

    client = TestClient(app_module.app)

    response = client.post("/demo/trigger")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ok"] is True
    assert payload["no_speech"] is True
    assert payload["audio_available"] is False

    latest = client.get("/latest/answer.json")
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["question"] == "（未识别到问题）"
    assert latest_payload["no_speech"] is True
    assert latest_payload["tts_provider"] == "skipped_no_speech"


def test_demo_ask_uses_direct_question_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    app_module = load_app_module()

    latest_json = tmp_path / "answer.json"
    latest_audio = tmp_path / "answer.mp3"

    class FakeProcess:
        def wait(self):
            return 0

    monkeypatch.setattr(app_module, "LATEST_JSON", latest_json)
    monkeypatch.setattr(app_module, "LATEST_AUDIO", latest_audio)
    monkeypatch.setattr(app_module, "stop_local_audio_playback", lambda: None)
    monkeypatch.setattr(app_module, "run_agent", lambda question: {
        "question_type": "concept",
        "language": "zh",
        "answer_short": f"{question}的答案来了。",
        "answer_full": ["第一点。", "第二点。"],
        "follow_up": "你还想继续问吗？",
        "show_image": "",
        "safety_note": "",
        "retrieved_chunks": [],
        "mode": "science",
        "grade": "auto",
        "spoken_answer": "这是实时识别后的直接回答。",
    })
    monkeypatch.setattr(app_module, "generate_answer_audio", lambda *args, **kwargs: latest_audio.write_bytes(b"ID3demo"))
    monkeypatch.setattr(app_module, "start_local_audio_playback", lambda *_args, **_kwargs: FakeProcess())

    client = TestClient(app_module.app)

    response = client.post("/demo/ask", json={"question": "什么是重力？"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["ok"] is True
    assert payload["question"] == "什么是重力？"
    assert payload["audio_available"] is True

    latest = client.get("/latest/answer.json")
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["question"] == "什么是重力？"
    assert latest_payload["asr_provider"] == "browser_speech"
