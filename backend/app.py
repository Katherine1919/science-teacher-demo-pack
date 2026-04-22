import asyncio
import inspect
import json
import logging
import os
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .agent import build_answer_preview, run_agent
from .openai_client import get_llm_provider, llm_provider_configured
from .services.asr_aliyun import ASRNoSpeechError, transcribe_question
from .services.local_audio import (
    LOCAL_AUDIO_PLAYBACK_ENABLED,
    clear_local_audio_process,
    start_local_audio_playback,
    stop_local_audio_playback,
)
from .services.recorder import QUESTION_AUDIO, RECORDER_MAX_SECONDS, record_question
from .services.tts import WAKEWORD_ACK_AUDIO, generate_answer_audio, generate_tts_audio
from .services.wakeword import (
    WAKEWORD_ENABLED,
    WAKEWORD_DEBOUNCE_SECONDS,
    WAKEWORD_INFERENCE_FRAMEWORK,
    WAKEWORD_INPUT_DEVICE,
    WAKEWORD_MODEL,
    WAKEWORD_THRESHOLD,
    WakeWordEvent,
    WakeWordListener,
    describe_input_device,
    list_input_devices,
)


def configure_logging() -> str:
    level_name = os.getenv("SCIENCE_TEACHER_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    base_logger = logging.getLogger("science_teacher")
    log_dir = Path(__file__).resolve().parents[1] / "latest"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "current_run.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(getattr(handler, "_science_teacher_handler", False) for handler in base_logger.handlers):
        handler = logging.StreamHandler()
        handler._science_teacher_handler = True
        handler.setFormatter(formatter)
        base_logger.addHandler(handler)

    if not any(getattr(handler, "_science_teacher_file_handler", False) for handler in base_logger.handlers):
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler._science_teacher_file_handler = True
        file_handler.setFormatter(formatter)
        base_logger.addHandler(file_handler)

    base_logger.setLevel(level)
    base_logger.propagate = False
    return level_name

ROOT = Path(__file__).resolve().parents[1]
LATEST_DIR = ROOT / "latest"
FRONTEND_DIR = ROOT / "frontend"

LATEST_JSON = LATEST_DIR / "answer.json"
LATEST_AUDIO = LATEST_DIR / "answer.mp3"

LOG_LEVEL_NAME = configure_logging()
logger = logging.getLogger("science_teacher.app")

app = FastAPI(title="Science Teacher MVP")

# Serve frontend assets if needed
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

class WSManager:
    def __init__(self):
        self.clients = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

ws_manager = WSManager()
pipeline_state_lock = asyncio.Lock()
pipeline_busy = False
pipeline_busy_event = threading.Event()

DEFAULT_PAYLOAD = {
    "question": "为什么会下雨？",
    "answer": "下雨是因为空气中的水蒸气变成了小水滴，并从云里落下来。\n\n1. 太阳会把地面上的水慢慢晒成水蒸气。\n2. 水蒸气升到高空后遇冷，会变成很多小水滴。\n3. 这些小水滴聚在一起形成云。\n4. 当云里的小水滴越来越多、越来越重时，就会变成雨落下来。\n\n你觉得太阳出来以后，地上的水会发生什么变化呢？",
    "lang": "zh",
    "ts": 1711111111,
    "question_type": "concept",
    "follow_up": "你觉得太阳出来以后，地上的水会发生什么变化呢？"
}

WAKEWORD_ACK_TEXT = os.getenv("WAKEWORD_ACK_TEXT", "我在").strip() or "我在"
WAKEWORD_ACK_DELAY_SECONDS = max(0.0, min(float(os.getenv("WAKEWORD_ACK_DELAY_SECONDS", "0.0")), 0.25))
WAKEWORD_ACK_MAX_SECONDS = max(0.0, float(os.getenv("WAKEWORD_ACK_MAX_SECONDS", "0.18")))
WAKEWORD_RECORDING_MAX_SECONDS = float(os.getenv("WAKEWORD_RECORDING_MAX_SECONDS", str(RECORDER_MAX_SECONDS)))


def providers_configured() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY")) and llm_provider_configured()


def load_latest_payload() -> dict:
    if LATEST_JSON.exists():
        try:
            return json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to load latest payload; using default payload")
    return dict(DEFAULT_PAYLOAD)


def build_offline_demo_payload(reason: str) -> dict:
    payload = load_latest_payload()
    payload["ts"] = int(time.time())
    payload["demo_mode"] = "offline"
    payload["demo_reason"] = reason
    payload["asr_provider"] = payload.get("asr_provider") or "offline_demo"
    payload["llm_provider"] = payload.get("llm_provider") or "offline_demo"
    payload["tts_provider"] = payload.get("tts_provider") or "offline_demo"
    return payload


def build_recorded_but_unprocessed_payload(reason: str) -> dict:
    llm_provider = get_llm_provider()
    answer_short = "我已经录到你的声音，但当前还不能识别并回答你的真实问题。"
    answer_full = [
        "录音已保存到 latest/question.wav。",
        f"当前缺少 DASHSCOPE_API_KEY 或 {llm_provider.upper()} 的 API 配置，所以云端 ASR/LLM 没有启动。",
        "配置好密钥后再试一次，我才会转写你刚才的问题并生成新的回答。",
    ]
    follow_up = "配置好密钥后，请再问我一次。"
    answer_text = "\n\n".join(
        [
            answer_short,
            "\n".join(f"{idx}. {item}" for idx, item in enumerate(answer_full, start=1)),
            follow_up,
        ]
    )
    return {
        "question": "（已录音，未转写）",
        "answer": answer_text,
        "lang": "zh",
        "language": "zh",
        "ts": int(time.time()),
        "question_type": "system",
        "follow_up": follow_up,
        "answer_short": answer_short,
        "answer_full": answer_full,
        "show_image": "",
        "safety_note": "",
        "retrieved_chunks": [],
        "mode": "science",
        "grade": "auto",
        "original_question": "",
        "asr_provider": "unconfigured",
        "llm_provider": llm_provider,
        "tts_provider": "disabled_offline",
        "tts_error": f"已录音，但未配置 DASHSCOPE_API_KEY / {llm_provider.upper()} API，因此没有转写或播放旧回答音频。",
        "demo_mode": "offline",
        "demo_reason": reason,
        "question_audio_saved": QUESTION_AUDIO.exists(),
        "question_audio_path": str(QUESTION_AUDIO),
    }


def build_no_speech_payload(trigger_source: str, detail: str) -> dict:
    answer_short = "我没有听清你刚才的问题。"
    answer_full = [
        "唤醒词已经触发，但录音里没有识别到有效语音。",
        "请在我说“我在，请讲”之后，直接提一个简短问题。",
        "如果现场环境较吵，请靠近麦克风并提高音量。",
    ]
    follow_up = "请再试一次，例如：雨是怎么形成的？"
    answer_text = "\n\n".join(
        [
            answer_short,
            "\n".join(f"{idx}. {item}" for idx, item in enumerate(answer_full, start=1)),
            follow_up,
        ]
    )
    return {
        "question": "（未识别到问题）",
        "answer": answer_text,
        "spoken_answer": "",
        "lang": "zh",
        "language": "zh",
        "ts": int(time.time()),
        "question_type": "system",
        "follow_up": follow_up,
        "answer_short": answer_short,
        "answer_full": answer_full,
        "show_image": "",
        "safety_note": "",
        "retrieved_chunks": [],
        "mode": "science",
        "grade": "auto",
        "original_question": "",
        "asr_provider": "aliyun_fun_asr",
        "llm_provider": get_llm_provider(),
        "tts_provider": "skipped_no_speech",
        "demo_mode": "no_speech_retry",
        "demo_reason": detail,
        "trigger_source": trigger_source,
        "question_audio_saved": QUESTION_AUDIO.exists(),
        "question_audio_path": str(QUESTION_AUDIO),
        "no_speech": True,
    }


def build_question_preview_payload(question: str, trigger_source: str) -> dict:
    return {
        "type": "question_ready",
        "question": question,
        "trigger_source": trigger_source,
        "ts": int(time.time()),
    }


def record_question_for_trigger(duration_seconds: float):
    record_signature = inspect.signature(record_question)
    if "duration_seconds" in record_signature.parameters:
        return record_question(duration_seconds=duration_seconds)
    return record_question()


async def run_offline_demo(trigger_source: str, reason: str, *, recorded_audio: bool = False) -> dict:
    logger.warning(
        "offline demo fallback activated: trigger_source=%s reason=%s recorded_audio=%s",
        trigger_source,
        reason,
        recorded_audio,
    )
    if recorded_audio:
        await ws_manager.broadcast(
            {
                "type": "state",
                "state": "thinking",
                "text": "已录音，但当前未配置云端识别能力 / Recorded audio, but cloud ASR is not configured...",
            }
        )
        payload = build_recorded_but_unprocessed_payload(reason)
        audio_available = False
    else:
        await ws_manager.broadcast({"type": "state", "state": "listening", "text": "使用本地演示素材… / Using local demo assets..."})
        await asyncio.sleep(0.2)
        await ws_manager.broadcast({"type": "state", "state": "thinking", "text": "正在准备演示回答… / Preparing demo answer..."})
        await asyncio.sleep(0.2)
        payload = build_offline_demo_payload(reason)
        audio_available = LATEST_AUDIO.exists() and LATEST_AUDIO.stat().st_size > 0

    try:
        await asyncio.to_thread(save_answer_payload, payload)
    except Exception:
        logger.exception("failed to persist offline demo payload; continuing with in-memory payload")

    await ws_manager.broadcast({"type": "answer_ready", "ts": payload["ts"], "audio_available": audio_available})
    return {
        "ok": True,
        "question": payload.get("question", ""),
        "audio_available": audio_available,
        "demo_mode": "offline",
        "demo_reason": reason,
    }


async def run_no_speech_retry(trigger_source: str, detail: str) -> dict:
    logger.warning("no speech detected: trigger_source=%s detail=%s", trigger_source, detail)
    payload = build_no_speech_payload(trigger_source, detail)
    await asyncio.to_thread(save_answer_payload, payload)
    await ws_manager.broadcast(
        {
            "type": "answer_ready",
            "ts": payload["ts"],
            "audio_available": False,
            "no_speech": True,
        }
    )
    await ws_manager.broadcast(
        {
            "type": "state",
            "state": "idle",
            "text": "没有听清，请再说一次 / I didn't catch that.",
        }
    )
    return {
        "ok": True,
        "no_speech": True,
        "question": payload["question"],
        "audio_available": False,
    }


def get_wakeword_listener():
    return getattr(app.state, "wakeword_listener", None)


def pause_wakeword_listener() -> None:
    listener = get_wakeword_listener()
    if listener is not None:
        listener.pause()


def resume_wakeword_listener() -> None:
    listener = get_wakeword_listener()
    if listener is not None:
        listener.resume()


def reactivate_wakeword_listener() -> None:
    listener = get_wakeword_listener()
    if listener is None:
        return

    try:
        listener.stop()
        listener.start()
        logger.info("wake word listener reactivated")
    except Exception:
        logger.exception("wake word listener reactivation failed; falling back to resume")
        listener.resume()


async def try_begin_pipeline(trigger_source: str, *, ignore_if_busy: bool = False):
    global pipeline_busy

    async with pipeline_state_lock:
        if pipeline_busy:
            message = "pipeline already running"
            if ignore_if_busy:
                logger.info("%s trigger ignored: %s", trigger_source, message)
                return {"ok": False, "ignored": True, "error": message}
            logger.info("%s trigger rejected: %s", trigger_source, message)
            return {"ok": False, "error": message}

        pipeline_busy = True
        pipeline_busy_event.set()
        return None


async def finish_pipeline() -> None:
    global pipeline_busy

    async with pipeline_state_lock:
        pipeline_busy = False
        pipeline_busy_event.clear()


def estimate_answer_playback_ms(text: str, language: str, minimum_ms: int = 2200) -> int:
    cleaned_text = "".join(str(text or "").split())
    if not cleaned_text:
        return minimum_ms

    chars_per_second = 4.0 if str(language).lower().startswith("zh") else 12.0
    estimated = 1200 + int((len(cleaned_text) / chars_per_second) * 1000)
    return max(minimum_ms, min(30000, estimated))


def handle_wakeword_future(future) -> None:
    try:
        future.result()
    except Exception:
        logger.exception("wake word pipeline scheduling failed")


async def handle_wakeword_event(event: WakeWordEvent) -> None:
    ack_audio_path = getattr(app.state, "wakeword_ack_audio_path", None)
    ack_animation_ms = estimate_answer_playback_ms(WAKEWORD_ACK_TEXT, "zh", minimum_ms=700)
    logger.info(
        "wake detected: model=%s label=%s score=%.3f ts=%.3f",
        event.model_name,
        event.label,
        event.score,
        event.timestamp,
    )
    logger.info(
        "pipeline started from wake trigger: model=%s label=%s score=%.3f",
        event.model_name,
        event.label,
        event.score,
    )
    result = await run_voice_pipeline(
        trigger_source=f"wakeword:{event.label}",
        ignore_if_busy=True,
        pre_listening_state={
            "type": "state",
            "state": "wake_detected",
            "text": f"{WAKEWORD_ACK_TEXT} / I'm listening...",
            "animate_for_ms": ack_animation_ms,
        },
        pre_listening_delay=WAKEWORD_ACK_DELAY_SECONDS,
        pre_listening_audio_path=ack_audio_path,
    )
    if not result.get("ok") and not result.get("ignored"):
        logger.error("wake word trigger failed: %s", result.get("error"))


async def watch_local_audio_playback(process, answer_ts: int) -> None:
    try:
        return_code = await asyncio.to_thread(process.wait)
        if return_code == 0:
            logger.info("playback ended: mode=local ts=%s returncode=%s", answer_ts, return_code)
        else:
            logger.error("playback failed: mode=local ts=%s returncode=%s", answer_ts, return_code)
    except Exception:
        logger.exception("local audio playback monitor failed: ts=%s", answer_ts)
        return_code = -1
    finally:
        clear_local_audio_process(process)
        reactivate_wakeword_listener()

    if return_code == 0:
        await ws_manager.broadcast({"type": "local_audio_finished", "ts": answer_ts})
    else:
        await ws_manager.broadcast(
            {
                "type": "playback_error",
                "ts": answer_ts,
                "text": "本机音频播放失败 / Local audio playback failed",
            }
        )


def on_wakeword_detected(event: WakeWordEvent) -> None:
    loop = getattr(app.state, "main_loop", None)
    if loop is None:
        logger.warning("wake word detected before app loop was ready; ignoring")
        return

    if pipeline_busy_event.is_set():
        logger.info(
            "wake word detected while pipeline active; ignoring: model=%s label=%s score=%.3f",
            event.model_name,
            event.label,
            event.score,
        )
        return

    future = asyncio.run_coroutine_threadsafe(handle_wakeword_event(event), loop)
    future.add_done_callback(handle_wakeword_future)


async def run_voice_pipeline(
    trigger_source: str,
    *,
    ignore_if_busy: bool = False,
    pre_listening_state: dict | None = None,
    pre_listening_delay: float = 0.0,
    pre_listening_audio_path: Path | None = None,
) -> dict:
    busy_result = await try_begin_pipeline(trigger_source, ignore_if_busy=ignore_if_busy)
    if busy_result is not None:
        return busy_result

    pause_wakeword_listener()
    pipeline_started_at = time.perf_counter()
    resume_wakeword_in_finally = True
    try:
        logger.info("pipeline started: trigger_source=%s", trigger_source)
        logger.info("%s trigger accepted", trigger_source)
        await asyncio.to_thread(stop_local_audio_playback)

        if pre_listening_state is not None:
            await ws_manager.broadcast(pre_listening_state)
        if pre_listening_audio_path is not None:
            try:
                logger.info(
                    "playback started: trigger_source=%s mode=local_ack path=%s",
                    trigger_source,
                    pre_listening_audio_path,
                )
                ack_process = await asyncio.to_thread(start_local_audio_playback, pre_listening_audio_path)
                ack_wait_seconds = WAKEWORD_ACK_MAX_SECONDS if trigger_source.startswith("wakeword:") else 0.0
                if ack_wait_seconds > 0:
                    await asyncio.sleep(ack_wait_seconds)
                    if ack_process.poll() is None:
                        logger.info(
                            "wake acknowledgement playback truncated after %.2fs to arm recording sooner",
                            ack_wait_seconds,
                        )
                        await asyncio.to_thread(stop_local_audio_playback)
                    else:
                        clear_local_audio_process(ack_process)
                else:
                    ack_return_code = await asyncio.to_thread(ack_process.wait)
                    clear_local_audio_process(ack_process)
                    if ack_return_code == 0:
                        logger.info("playback ended: mode=local_ack returncode=%s", ack_return_code)
                    else:
                        logger.error("playback failed: mode=local_ack returncode=%s", ack_return_code)
            except Exception:
                logger.exception("wake acknowledgement playback failed")
        if pre_listening_delay > 0:
            await asyncio.sleep(pre_listening_delay)

        recording_duration = RECORDER_MAX_SECONDS
        if trigger_source.startswith("wakeword:"):
            recording_duration = WAKEWORD_RECORDING_MAX_SECONDS

        logger.info("recording started: trigger_source=%s", trigger_source)
        await ws_manager.broadcast({"type": "state", "state": "listening", "text": "我在，请讲 / I'm listening..."})
        recording_started_at = time.perf_counter()
        question_audio = await asyncio.to_thread(record_question_for_trigger, recording_duration)
        recording_elapsed = time.perf_counter() - recording_started_at
        logger.info("recording ended: trigger_source=%s path=%s", trigger_source, question_audio)
        logger.info("latency: trigger_source=%s stage=recording seconds=%.3f", trigger_source, recording_elapsed)
        logger.info("transition to thinking: trigger_source=%s stage=transcription", trigger_source)
        await ws_manager.broadcast(
            {
                "type": "state",
                "state": "thinking",
                "text": "正在识别问题… / Transcribing your question...",
            }
        )

        if not providers_configured():
            logger.warning(
                "providers not configured after recording: trigger_source=%s saved_audio=%s",
                trigger_source,
                question_audio,
            )
            return await run_offline_demo(
                trigger_source,
                f"Missing DASHSCOPE_API_KEY or {get_llm_provider().upper()} API configuration",
                recorded_audio=True,
            )

        asr_started_at = time.perf_counter()
        try:
            question = await asyncio.to_thread(transcribe_question)
        except ASRNoSpeechError as exc:
            return await run_no_speech_retry(trigger_source, str(exc))
        asr_elapsed = time.perf_counter() - asr_started_at
        logger.info(
            "ASR completed: trigger_source=%s chars=%s text=%s",
            trigger_source,
            len(question.strip()),
            question,
        )
        logger.info("latency: trigger_source=%s stage=asr seconds=%.3f", trigger_source, asr_elapsed)
        result = await deliver_answer_for_question(
            question,
            trigger_source,
            asr_provider="aliyun_fun_asr",
            pipeline_started_at=pipeline_started_at,
        )
        if result.get("local_audio_playback"):
            resume_wakeword_in_finally = False
        return result
    except Exception as exc:
        logger.exception("voice pipeline failed")
        await ws_manager.broadcast(
            {
                "type": "error",
                "state": "error",
                "text": f"流程失败 / Pipeline failed: {exc}",
            }
        )
        return {"ok": False, "error": str(exc)}
    finally:
        if resume_wakeword_in_finally:
            reactivate_wakeword_listener()
        await finish_pipeline()
        logger.info("pipeline reset complete: trigger_source=%s", trigger_source)

@app.on_event("startup")
async def startup_event():
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    if not LATEST_JSON.exists():
        LATEST_JSON.write_text(json.dumps(DEFAULT_PAYLOAD, ensure_ascii=False, indent=2), encoding="utf-8")
    app.state.main_loop = asyncio.get_running_loop()
    app.state.wakeword_listener = None
    app.state.wakeword_ack_audio_path = None

    logger.info(
        "startup configuration: log_level=%s wakeword_enabled=%s wakeword_model=%s wakeword_input_device=%s wakeword_threshold=%.2f wakeword_debounce=%.2f wakeword_ack_delay=%.2f wakeword_framework=%s llm_provider=%s providers_configured=%s",
        LOG_LEVEL_NAME,
        WAKEWORD_ENABLED,
        WAKEWORD_MODEL,
        WAKEWORD_INPUT_DEVICE or "default",
        WAKEWORD_THRESHOLD,
        WAKEWORD_DEBOUNCE_SECONDS,
        WAKEWORD_ACK_DELAY_SECONDS,
        WAKEWORD_INFERENCE_FRAMEWORK,
        get_llm_provider(),
        providers_configured(),
    )
    logger.info("local audio playback enabled=%s", LOCAL_AUDIO_PLAYBACK_ENABLED)

    if WAKEWORD_ENABLED:
        if WAKEWORD_ACK_TEXT:
            try:
                app.state.wakeword_ack_audio_path = await asyncio.to_thread(
                    generate_tts_audio,
                    WAKEWORD_ACK_TEXT,
                    WAKEWORD_ACK_AUDIO,
                    "zh",
                )
                logger.info("wake word acknowledgement audio ready: %s", app.state.wakeword_ack_audio_path)
            except Exception:
                logger.exception("failed to prepare wake word acknowledgement audio")
        try:
            logger.info("available input devices:\n%s", list_input_devices())
            logger.info("selected wake word input device: %s", describe_input_device(WAKEWORD_INPUT_DEVICE))
            logger.info("initializing wake word listener")
            listener = WakeWordListener(on_wake=on_wakeword_detected)
            app.state.wakeword_listener = listener
            listener.start()
            logger.info("wake word listener thread started")
        except Exception:
            logger.exception("wake listener error")
    else:
        logger.info("wake word listener disabled by WAKEWORD_ENABLED")


@app.on_event("shutdown")
async def shutdown_event():
    listener = get_wakeword_listener()
    if listener is not None:
        listener.stop()
    await asyncio.to_thread(stop_local_audio_playback)

@app.get("/")
def index():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/latest/answer.json")
def latest_answer_json():
    if not LATEST_JSON.exists():
        return JSONResponse(DEFAULT_PAYLOAD)
    return JSONResponse(json.loads(LATEST_JSON.read_text(encoding="utf-8")))

@app.get("/latest/answer.mp3")
def latest_answer_mp3():
    if LATEST_AUDIO.exists():
        return FileResponse(str(LATEST_AUDIO), media_type="audio/mpeg")
    return JSONResponse({"error": "audio not found; run real TTS pipeline later"}, status_code=404)

@app.get("/latest/question.wav")
def latest_question_wav():
    if QUESTION_AUDIO.exists():
        return FileResponse(str(QUESTION_AUDIO), media_type="audio/wav")
    return JSONResponse({"error": "question audio not found"}, status_code=404)

def build_answer_text(agent_result: dict) -> str:
    parts = []
    answer_short = str(agent_result.get("answer_short", "")).strip()
    if answer_short:
        parts.append(answer_short)

    answer_full = [
        str(item).strip()
        for item in agent_result.get("answer_full", [])
        if str(item).strip()
    ]
    if answer_full:
        parts.append("\n".join(f"{idx}. {item}" for idx, item in enumerate(answer_full, start=1)))

    follow_up = str(agent_result.get("follow_up", "")).strip()
    if follow_up:
        parts.append(follow_up)

    return "\n\n".join(parts).strip()


def build_answer_payload(question: str, agent_result: dict, *, asr_provider: str = "aliyun_fun_asr") -> dict:
    answer_text = build_answer_text(agent_result)
    return {
        "question": question,
        "answer": answer_text,
        "spoken_answer": agent_result.get("spoken_answer", agent_result.get("answer_short", answer_text)),
        "lang": agent_result.get("language", "zh"),
        "language": agent_result.get("language", "zh"),
        "ts": int(time.time()),
        "question_type": agent_result.get("question_type", "unknown"),
        "follow_up": agent_result.get("follow_up", ""),
        "answer_short": agent_result.get("answer_short", ""),
        "answer_full": agent_result.get("answer_full", []),
        "show_image": agent_result.get("show_image", ""),
        "safety_note": agent_result.get("safety_note", ""),
        "retrieved_chunks": agent_result.get("retrieved_chunks", []),
        "mode": agent_result.get("mode", "science"),
        "grade": agent_result.get("grade", "auto"),
        "original_question": agent_result.get("original_question", question),
        "asr_provider": asr_provider,
        "llm_provider": get_llm_provider(),
        "tts_provider": "edge_tts",
        "audio_pending": False,
        "local_audio_playback": False,
        "answer_phase": "preview" if agent_result.get("preview") else "final",
    }


def save_answer_payload(payload: dict) -> None:
    LATEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def deliver_answer_for_question(
    question: str,
    trigger_source: str,
    *,
    asr_provider: str,
    pipeline_started_at: float,
) -> dict:
    await ws_manager.broadcast(build_question_preview_payload(question, trigger_source))
    await ws_manager.broadcast({"type": "state", "state": "thinking", "text": "我在思考… / Thinking..."})

    preview_result = await asyncio.to_thread(build_answer_preview, question)
    if preview_result is not None:
        preview_payload = await asyncio.to_thread(build_answer_payload, question, preview_result, asr_provider=asr_provider)
        preview_payload["audio_pending"] = True
        preview_payload["local_audio_playback"] = False
        preview_payload["estimated_answer_playback_ms"] = estimate_answer_playback_ms(
            preview_payload.get("spoken_answer", preview_payload.get("answer", "")),
            preview_payload.get("lang", "zh"),
        )
        await asyncio.to_thread(save_answer_payload, preview_payload)
        await ws_manager.broadcast(
            {
                "type": "answer_text_ready",
                "ts": preview_payload["ts"],
                "audio_pending": True,
                "preview": True,
                "estimated_answer_playback_ms": preview_payload["estimated_answer_playback_ms"],
            }
        )

    agent_result = None
    llm_elapsed = 0.0
    if preview_result is not None and preview_result.get("fast_answer_confident"):
        agent_result = dict(preview_result)
        agent_result.pop("preview", None)
        logger.info("agent shortcut: trigger_source=%s using confident local answer", trigger_source)
    else:
        llm_started_at = time.perf_counter()
        agent_result = await asyncio.to_thread(run_agent, question)
        llm_elapsed = time.perf_counter() - llm_started_at
    payload = await asyncio.to_thread(build_answer_payload, question, agent_result, asr_provider=asr_provider)
    logger.info(
        "agent completed: trigger_source=%s question_type=%s answer_chars=%s",
        trigger_source,
        payload.get("question_type", "unknown"),
        len(str(payload.get("answer", "")).strip()),
    )
    logger.info("latency: trigger_source=%s stage=llm seconds=%.3f", trigger_source, llm_elapsed)
    payload["local_audio_playback"] = False
    payload["audio_pending"] = True
    payload["estimated_answer_playback_ms"] = estimate_answer_playback_ms(
        payload.get("spoken_answer", payload.get("answer", "")),
        payload.get("lang", "zh"),
    )
    await asyncio.to_thread(save_answer_payload, payload)
    await ws_manager.broadcast(
        {
            "type": "answer_text_ready",
            "ts": payload["ts"],
            "audio_pending": True,
            "estimated_answer_playback_ms": payload["estimated_answer_playback_ms"],
        }
    )

    audio_available = True
    local_audio_playback = False
    try:
        tts_started_at = time.perf_counter()
        await asyncio.to_thread(generate_answer_audio, payload["spoken_answer"], LATEST_AUDIO, payload["lang"])
        tts_elapsed = time.perf_counter() - tts_started_at
        logger.info(
            "TTS completed: trigger_source=%s path=%s lang=%s",
            trigger_source,
            LATEST_AUDIO,
            payload.get("lang", "zh"),
        )
        logger.info("latency: trigger_source=%s stage=tts seconds=%.3f", trigger_source, tts_elapsed)
    except Exception as exc:
        if LATEST_AUDIO.exists():
            LATEST_AUDIO.unlink()
        audio_available = False
        payload["audio_pending"] = False
        payload["tts_error"] = str(exc)
        logger.exception("TTS failed")
        await asyncio.to_thread(save_answer_payload, payload)
    else:
        if LOCAL_AUDIO_PLAYBACK_ENABLED:
            try:
                local_audio_process = await asyncio.to_thread(start_local_audio_playback, LATEST_AUDIO)
                local_audio_playback = True
                payload["local_audio_playback"] = True
                logger.info(
                    "playback started: trigger_source=%s mode=local path=%s estimated_ms=%s",
                    trigger_source,
                    LATEST_AUDIO,
                    payload["estimated_answer_playback_ms"],
                )
                asyncio.create_task(watch_local_audio_playback(local_audio_process, payload["ts"]))
            except Exception as exc:
                payload["local_audio_playback_error"] = str(exc)
                logger.exception("local audio playback failed")
        payload["audio_pending"] = False
        await asyncio.to_thread(save_answer_payload, payload)

    await ws_manager.broadcast(
        {
            "type": "answer_ready",
            "ts": payload["ts"],
            "audio_available": audio_available,
            "local_audio_playback": local_audio_playback,
            "estimated_answer_playback_ms": payload["estimated_answer_playback_ms"],
        }
    )
    logger.info("frontend notified")
    logger.info(
        "latency: trigger_source=%s stage=total seconds=%.3f",
        trigger_source,
        time.perf_counter() - pipeline_started_at,
    )
    return {
        "ok": True,
        "question": question,
        "audio_available": audio_available,
        "local_audio_playback": local_audio_playback,
    }


async def run_text_pipeline(
    trigger_source: str,
    question: str,
    *,
    ignore_if_busy: bool = False,
    asr_provider: str = "browser_speech",
) -> dict:
    question = str(question or "").strip()
    if not question:
        return {"ok": False, "error": "question is empty"}

    busy_result = await try_begin_pipeline(trigger_source, ignore_if_busy=ignore_if_busy)
    if busy_result is not None:
        return busy_result

    pause_wakeword_listener()
    pipeline_started_at = time.perf_counter()
    try:
        logger.info(
            "text pipeline started: trigger_source=%s chars=%s question=%s",
            trigger_source,
            len(question),
            question,
        )
        await asyncio.to_thread(stop_local_audio_playback)

        if not llm_provider_configured():
            raise RuntimeError(f"{get_llm_provider().upper()} API is not configured")

        return await deliver_answer_for_question(
            question,
            trigger_source,
            asr_provider=asr_provider,
            pipeline_started_at=pipeline_started_at,
        )
    except Exception as exc:
        logger.exception("text pipeline failed")
        await ws_manager.broadcast(
            {
                "type": "error",
                "state": "error",
                "text": f"流程失败 / Pipeline failed: {exc}",
            }
        )
        return {"ok": False, "error": str(exc)}
    finally:
        reactivate_wakeword_listener()
        await finish_pipeline()
        logger.info("pipeline reset complete: trigger_source=%s", trigger_source)

@app.post("/demo/trigger")
async def demo_trigger():
    result = await run_voice_pipeline("manual")
    if result.get("ok"):
        return result

    status_code = 409 if result.get("error") == "pipeline already running" else 500
    return JSONResponse(result, status_code=status_code)


@app.post("/demo/ask")
async def demo_ask(payload: dict = Body(...)):
    question = str((payload or {}).get("question", "")).strip()
    if not question:
        return JSONResponse({"ok": False, "error": "question is empty"}, status_code=400)

    trigger_source = str((payload or {}).get("trigger_source", "manual:browser_speech")).strip() or "manual:browser_speech"
    asr_provider = str((payload or {}).get("asr_provider", "browser_speech")).strip() or "browser_speech"

    result = await run_text_pipeline(
        trigger_source=trigger_source,
        question=question,
        asr_provider=asr_provider,
    )
    if result.get("ok"):
        return result

    status_code = 409 if result.get("error") == "pipeline already running" else 500
    return JSONResponse(result, status_code=status_code)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        ws_manager.disconnect(ws)
