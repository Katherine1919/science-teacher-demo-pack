import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger("science_teacher.local_audio")

LOCAL_AUDIO_PLAYBACK_ENABLED = (
    os.getenv("LOCAL_AUDIO_PLAYBACK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
)
LOCAL_AUDIO_PLAYER = os.getenv("LOCAL_AUDIO_PLAYER", "").strip()

_active_process_lock = threading.Lock()
_active_process: subprocess.Popen | None = None


class LocalAudioPlaybackError(RuntimeError):
    pass


def _build_player_command(audio_path: Path) -> list[str]:
    if LOCAL_AUDIO_PLAYER:
        command = shlex.split(LOCAL_AUDIO_PLAYER)
        if not command:
            raise LocalAudioPlaybackError("LOCAL_AUDIO_PLAYER is set but empty after parsing.")
        return [*command, str(audio_path)]

    if sys.platform == "darwin":
        return ["afplay", str(audio_path)]

    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)]

    if shutil.which("aplay"):
        return ["aplay", str(audio_path)]

    raise LocalAudioPlaybackError(
        "No supported local audio player found. Set LOCAL_AUDIO_PLAYER or install afplay/ffplay/aplay."
    )


def stop_local_audio_playback() -> None:
    global _active_process

    with _active_process_lock:
        process = _active_process
        _active_process = None

    if process is None or process.poll() is not None:
        return

    logger.info("stopping previous local audio playback: pid=%s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def start_local_audio_playback(audio_path: Path) -> subprocess.Popen:
    global _active_process

    if not LOCAL_AUDIO_PLAYBACK_ENABLED:
        raise LocalAudioPlaybackError("Local audio playback is disabled.")

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise LocalAudioPlaybackError(f"Audio file not found: {audio_path}")

    command = _build_player_command(audio_path)
    stop_local_audio_playback()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise LocalAudioPlaybackError(f"Local audio player not found: {command[0]}") from exc
    except Exception as exc:
        raise LocalAudioPlaybackError(f"Local audio playback failed to start: {exc}") from exc

    with _active_process_lock:
        _active_process = process

    # Catch immediate player failures so the caller can fall back instead of assuming sound started.
    time.sleep(0.15)
    return_code = process.poll()
    if return_code not in {None, 0}:
        clear_local_audio_process(process)
        raise LocalAudioPlaybackError(
            f"Local audio player exited immediately with return code {return_code}: {command[0]}"
        )

    logger.info("local audio playback started: player=%s pid=%s path=%s", command[0], process.pid, audio_path)
    return process


def clear_local_audio_process(process: subprocess.Popen) -> None:
    global _active_process

    with _active_process_lock:
        if _active_process is process:
            _active_process = None
