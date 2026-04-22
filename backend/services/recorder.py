import logging
import math
import os
import subprocess
import time
import wave
from collections import deque
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime environment dependent
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - runtime environment dependent
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[2]
LATEST_DIR = ROOT / "latest"
QUESTION_AUDIO = LATEST_DIR / "question.wav"

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
RECORDER_MODE = os.getenv("RECORDER_MODE", "adaptive").strip().lower() or "adaptive"
RECORDER_INPUT = os.getenv("RECORDER_INPUT", ":0")
RECORDER_INPUT_DEVICE = os.getenv("RECORDER_INPUT_DEVICE", "").strip()
RECORDER_MAX_SECONDS = float(os.getenv("RECORDER_MAX_SECONDS", "3.5"))
RECORDER_MIN_SECONDS = float(os.getenv("RECORDER_MIN_SECONDS", "0.35"))
RECORDER_SAMPLE_RATE = int(os.getenv("RECORDER_SAMPLE_RATE", "16000"))
RECORDER_CHUNK_FRAMES = int(os.getenv("RECORDER_CHUNK_FRAMES", "1280"))
RECORDER_SPEECH_THRESHOLD = float(os.getenv("RECORDER_SPEECH_THRESHOLD", "0.009"))
RECORDER_SILENCE_STOP_SECONDS = float(os.getenv("RECORDER_SILENCE_STOP_SECONDS", "0.40"))
RECORDER_PRE_ROLL_SECONDS = float(os.getenv("RECORDER_PRE_ROLL_SECONDS", "0.60"))
RECORDER_WALL_CLOCK_GRACE_SECONDS = float(os.getenv("RECORDER_WALL_CLOCK_GRACE_SECONDS", "1.0"))

logger = logging.getLogger("science_teacher.recorder")


class RecorderError(RuntimeError):
    pass


def _resolve_sounddevice_input_device(input_device: str = RECORDER_INPUT_DEVICE):
    if input_device in {"", None}:
        return None
    try:
        return int(input_device)
    except (TypeError, ValueError):
        return input_device


def _write_wav_file(output_path: Path, frames: list) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RECORDER_SAMPLE_RATE)
        wav_file.writeframes(b"".join(frame.tobytes() for frame in frames))


def _list_sounddevice_audio_devices() -> str:
    if _SOUNDDEVICE_IMPORT_ERROR is not None:
        return f"sounddevice import failed: {_SOUNDDEVICE_IMPORT_ERROR}"
    try:
        devices = sd.query_devices()
    except Exception as exc:
        return f"sounddevice device query failed: {exc}"

    lines = []
    for idx, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels > 0:
            lines.append(f"[{idx}] {device['name']} (inputs={max_input_channels})")
    return "\n".join(lines) or "No input-capable sounddevice devices were reported."


def list_audio_devices() -> str:
    if RECORDER_MODE == "adaptive":
        return _list_sounddevice_audio_devices()

    result = subprocess.run(
        [FFMPEG_BIN, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    details = (result.stderr or "").strip()
    return details or "ffmpeg did not return any AVFoundation device details."


def _record_question_ffmpeg(
    output_path: Path,
    duration_seconds: float,
    input_device: str,
) -> Path:
    logger.info(
        "recording started: mode=ffmpeg input=%s duration=%ss sample_rate=%s",
        input_device,
        duration_seconds,
        RECORDER_SAMPLE_RATE,
    )

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "avfoundation",
        "-i",
        input_device,
        "-t",
        str(duration_seconds),
        "-ac",
        "1",
        "-ar",
        str(RECORDER_SAMPLE_RATE),
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RecorderError(f"ffmpeg not found at {FFMPEG_BIN!r}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "unknown ffmpeg error"
        device_details = list_audio_devices()
        raise RecorderError(
            "Microphone recording failed with ffmpeg. "
            f"Input={input_device!r}, duration={duration_seconds}s, error={stderr}\n"
            f"Available device info:\n{device_details}"
        ) from exc

    return output_path


def _record_question_adaptive(output_path: Path, duration_seconds: float) -> Path:
    if _NUMPY_IMPORT_ERROR is not None:
        raise RecorderError(f"numpy import failed: {_NUMPY_IMPORT_ERROR}")
    if _SOUNDDEVICE_IMPORT_ERROR is not None:
        raise RecorderError(f"sounddevice import failed: {_SOUNDDEVICE_IMPORT_ERROR}")

    selected_device = _resolve_sounddevice_input_device()
    chunk_frames = max(256, RECORDER_CHUNK_FRAMES)
    max_frames = max(chunk_frames, int(duration_seconds * RECORDER_SAMPLE_RATE))
    min_frames = max(chunk_frames, int(RECORDER_MIN_SECONDS * RECORDER_SAMPLE_RATE))
    silence_stop_frames = max(chunk_frames, int(RECORDER_SILENCE_STOP_SECONDS * RECORDER_SAMPLE_RATE))
    pre_roll_chunks = max(1, math.ceil((RECORDER_PRE_ROLL_SECONDS * RECORDER_SAMPLE_RATE) / chunk_frames))
    wall_clock_timeout = max(float(duration_seconds), RECORDER_MIN_SECONDS) + RECORDER_WALL_CLOCK_GRACE_SECONDS
    wall_clock_deadline = time.monotonic() + wall_clock_timeout

    logger.info(
        "recording started: mode=adaptive device=%s max_seconds=%.2f silence_stop=%.2fs threshold=%.4f sample_rate=%s",
        selected_device if selected_device is not None else "default",
        duration_seconds,
        RECORDER_SILENCE_STOP_SECONDS,
        RECORDER_SPEECH_THRESHOLD,
        RECORDER_SAMPLE_RATE,
    )

    all_chunks = []
    captured_chunks = []
    pre_roll = deque(maxlen=pre_roll_chunks)
    total_frames = 0
    captured_frames = 0
    silence_frames = 0
    speech_started = False
    hit_max_duration = False
    hit_wall_clock_timeout = False

    try:
        with sd.InputStream(
            device=selected_device,
            channels=1,
            samplerate=RECORDER_SAMPLE_RATE,
            dtype="int16",
            blocksize=chunk_frames,
        ) as stream:
            while total_frames < max_frames:
                if time.monotonic() >= wall_clock_deadline:
                    hit_wall_clock_timeout = True
                    logger.warning(
                        "recording timeout: wall_clock_seconds=%.2f captured_seconds=%.2f speech_started=%s",
                        wall_clock_timeout,
                        total_frames / RECORDER_SAMPLE_RATE,
                        speech_started,
                    )
                    break

                read_available = getattr(stream, "read_available", chunk_frames)
                if isinstance(read_available, int) and read_available < chunk_frames:
                    time.sleep(0.01)
                    continue

                audio_block, overflowed = stream.read(chunk_frames)
                if overflowed:
                    logger.warning("recording stream overflowed")

                chunk = np.asarray(audio_block[:, 0], dtype=np.int16).copy()
                total_frames += len(chunk)
                all_chunks.append(chunk)
                pre_roll.append(chunk)

                normalized = chunk.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(np.square(normalized))))

                if not speech_started:
                    if rms >= RECORDER_SPEECH_THRESHOLD:
                        speech_started = True
                        logger.info(
                            "speech detected: rms=%.4f threshold=%.4f pre_roll_chunks=%s",
                            rms,
                            RECORDER_SPEECH_THRESHOLD,
                            len(pre_roll),
                        )
                        buffered_chunks = list(pre_roll)
                        captured_chunks.extend(buffered_chunks)
                        captured_frames += sum(len(buffered_chunk) for buffered_chunk in buffered_chunks)
                        silence_frames = 0
                    continue

                captured_chunks.append(chunk)
                captured_frames += len(chunk)

                if rms >= RECORDER_SPEECH_THRESHOLD:
                    silence_frames = 0
                else:
                    silence_frames += len(chunk)

                if captured_frames >= min_frames and silence_frames >= silence_stop_frames:
                    logger.info(
                        "recording auto-stopped after silence: captured_seconds=%.2f silence_seconds=%.2f",
                        captured_frames / RECORDER_SAMPLE_RATE,
                        silence_frames / RECORDER_SAMPLE_RATE,
                    )
                    break
            else:
                hit_max_duration = True
    except Exception as exc:
        raise RecorderError(
            "Adaptive microphone recording failed. "
            f"Selected device={selected_device!r}. Available devices:\n{list_audio_devices()}\nError: {exc}"
        ) from exc

    if hit_max_duration:
        logger.warning(
            "recording timeout: max_duration_seconds=%.2f captured_seconds=%.2f speech_started=%s",
            duration_seconds,
            total_frames / RECORDER_SAMPLE_RATE,
            speech_started,
        )

    frames_to_write = captured_chunks if speech_started else all_chunks
    if not frames_to_write:
        raise RecorderError("Adaptive recording finished without any captured audio frames.")

    _write_wav_file(output_path, frames_to_write)
    actual_seconds = sum(len(frame) for frame in frames_to_write) / RECORDER_SAMPLE_RATE
    logger.info(
        "recording ended: mode=adaptive actual_seconds=%.2f speech_started=%s",
        actual_seconds,
        speech_started,
    )
    logger.info(
        "recording stream released: mode=adaptive device=%s wall_clock_timeout=%s max_duration=%s",
        selected_device if selected_device is not None else "default",
        hit_wall_clock_timeout,
        hit_max_duration,
    )
    return output_path


def record_question(
    output_path: Path = QUESTION_AUDIO,
    duration_seconds: float = RECORDER_MAX_SECONDS,
    input_device: str = RECORDER_INPUT,
) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    recorder_mode = RECORDER_MODE
    if recorder_mode == "adaptive":
        try:
            _record_question_adaptive(output_path, duration_seconds)
        except RecorderError:
            raise
        except Exception as exc:
            raise RecorderError(f"Adaptive microphone recording failed: {exc}") from exc
    else:
        _record_question_ffmpeg(output_path, duration_seconds, input_device)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RecorderError(f"Recording finished but {output_path} was not created.")

    logger.info("file saved: %s", output_path)
    return output_path
