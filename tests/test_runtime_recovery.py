import asyncio
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_app_module():
    if "backend.app" in sys.modules:
        return importlib.reload(sys.modules["backend.app"])
    return importlib.import_module("backend.app")


def test_run_voice_pipeline_releases_busy_state_on_failure(monkeypatch):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    app_module = load_app_module()

    calls = []

    async def fake_broadcast(payload):
        calls.append(payload)

    def raise_recording_error(*_args, **_kwargs):
        raise RuntimeError("recording boom")

    monkeypatch.setattr(app_module.ws_manager, "broadcast", fake_broadcast)
    monkeypatch.setattr(app_module, "pause_wakeword_listener", lambda: calls.append("pause"))
    monkeypatch.setattr(app_module, "reactivate_wakeword_listener", lambda: calls.append("reactivate"))
    monkeypatch.setattr(app_module, "stop_local_audio_playback", lambda: None)
    monkeypatch.setattr(app_module, "record_question_for_trigger", raise_recording_error)

    app_module.pipeline_busy = False
    app_module.pipeline_busy_event.clear()

    result = asyncio.run(app_module.run_voice_pipeline("manual"))

    assert result["ok"] is False
    assert result["error"] == "recording boom"
    assert app_module.pipeline_busy is False
    assert app_module.pipeline_busy_event.is_set() is False
    assert "reactivate" in calls


def test_watch_local_audio_playback_resumes_listener_after_playback(monkeypatch):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    app_module = load_app_module()

    calls = []

    class FakeProcess:
        def wait(self):
            return 0

    async def fake_broadcast(payload):
        calls.append(payload)

    monkeypatch.setattr(app_module.ws_manager, "broadcast", fake_broadcast)
    monkeypatch.setattr(app_module, "clear_local_audio_process", lambda _process: calls.append("clear"))
    monkeypatch.setattr(app_module, "reactivate_wakeword_listener", lambda: calls.append("reactivate"))

    asyncio.run(app_module.watch_local_audio_playback(FakeProcess(), 123))

    assert "clear" in calls
    assert "reactivate" in calls
    assert {"type": "local_audio_finished", "ts": 123} in calls


def test_on_wakeword_detected_skips_when_busy_event_is_set(monkeypatch):
    monkeypatch.setenv("WAKEWORD_ENABLED", "0")
    app_module = load_app_module()

    scheduled = []
    app_module.app.state.main_loop = object()
    app_module.pipeline_busy = False
    app_module.pipeline_busy_event.set()

    def fake_run_coroutine_threadsafe(*_args, **_kwargs):
        scheduled.append(True)
        raise AssertionError("wake pipeline should not be scheduled while busy")

    monkeypatch.setattr(app_module.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    event = app_module.WakeWordEvent(
        model_name="alexa",
        label="alexa",
        score=0.92,
        timestamp=0.0,
    )
    app_module.on_wakeword_detected(event)

    assert not scheduled
