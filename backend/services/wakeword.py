import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - handled at runtime
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - handled at runtime
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

try:
    import openwakeword
    from openwakeword.model import Model
    from openwakeword.utils import download_models
except Exception as exc:  # pragma: no cover - handled at runtime
    openwakeword = None
    Model = None
    download_models = None
    _OPENWAKEWORD_IMPORT_ERROR = exc
else:
    _OPENWAKEWORD_IMPORT_ERROR = None

WAKEWORD_ENABLED = os.getenv("WAKEWORD_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "alexa").strip() or "alexa"
WAKEWORD_INPUT_DEVICE = os.getenv("WAKEWORD_INPUT_DEVICE", "").strip()
WAKEWORD_SAMPLE_RATE = int(os.getenv("WAKEWORD_SAMPLE_RATE", "16000"))
WAKEWORD_CHUNK_SIZE = int(os.getenv("WAKEWORD_CHUNK_SIZE", "1280"))
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_DEBOUNCE_SECONDS = float(os.getenv("WAKEWORD_DEBOUNCE_SECONDS", "4.0"))
WAKEWORD_INFERENCE_FRAMEWORK = os.getenv("WAKEWORD_INFERENCE_FRAMEWORK", "auto").strip().lower() or "auto"
WAKEWORD_VAD_THRESHOLD = float(os.getenv("WAKEWORD_VAD_THRESHOLD", "0"))

logger = logging.getLogger("science_teacher.wakeword")


class WakeWordError(RuntimeError):
    pass


@dataclass(frozen=True)
class WakeWordEvent:
    model_name: str
    label: str
    score: float
    timestamp: float


def ensure_wakeword_dependencies() -> None:
    if _NUMPY_IMPORT_ERROR is not None:
        raise WakeWordError(f"numpy import failed: {_NUMPY_IMPORT_ERROR}")
    if _SOUNDDEVICE_IMPORT_ERROR is not None:
        raise WakeWordError(f"sounddevice import failed: {_SOUNDDEVICE_IMPORT_ERROR}")
    if _OPENWAKEWORD_IMPORT_ERROR is not None:
        raise WakeWordError(f"openwakeword import failed: {_OPENWAKEWORD_IMPORT_ERROR}")


def _normalize_model_name(model_name: str) -> str:
    return str(model_name).strip().lower().replace(" ", "_")


def resolve_wakeword_model(model_name: str = WAKEWORD_MODEL) -> str:
    ensure_wakeword_dependencies()

    requested = str(model_name).strip()
    if not requested:
        requested = WAKEWORD_MODEL

    if Path(requested).exists():
        return str(Path(requested).resolve())

    normalized = _normalize_model_name(requested)
    if normalized in openwakeword.MODELS:
        return normalized

    available = ", ".join(sorted(openwakeword.MODELS.keys()))
    raise WakeWordError(f"Unknown wake word model {requested!r}. Available built-in models: {available}")


def resolve_input_device(input_device: str = WAKEWORD_INPUT_DEVICE):
    if input_device in {"", None}:
        return None
    try:
        return int(input_device)
    except (TypeError, ValueError):
        return input_device


def list_input_devices() -> str:
    ensure_wakeword_dependencies()
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


def describe_input_device(input_device: str = WAKEWORD_INPUT_DEVICE) -> str:
    ensure_wakeword_dependencies()
    selected_device = resolve_input_device(input_device)
    try:
        info = sd.query_devices(selected_device, "input")
    except Exception as exc:
        label = "default" if selected_device is None else repr(selected_device)
        return f"Unable to resolve input device {label}: {exc}"

    return (
        f"{info['name']} "
        f"(inputs={int(info.get('max_input_channels', 0))}, "
        f"default_samplerate={info.get('default_samplerate')})"
    )


def build_wakeword_model(
    model_name: str = WAKEWORD_MODEL,
    inference_framework: str = WAKEWORD_INFERENCE_FRAMEWORK,
):
    ensure_wakeword_dependencies()
    selected_model = resolve_wakeword_model(model_name)
    selected_framework = resolve_inference_framework(selected_model, inference_framework)
    logger.info("wake word model selected: model=%s framework=%s", selected_model, selected_framework)
    return Model(
        wakeword_models=[selected_model],
        inference_framework=selected_framework,
        vad_threshold=WAKEWORD_VAD_THRESHOLD,
    ), selected_model


def detect_wakeword_in_audio_file(
    audio_path: Path,
    model_name: str = WAKEWORD_MODEL,
    threshold: float = WAKEWORD_THRESHOLD,
    inference_framework: str = WAKEWORD_INFERENCE_FRAMEWORK,
) -> Optional[WakeWordEvent]:
    ensure_wakeword_dependencies()

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise WakeWordError(f"Wake word audio file not found: {audio_path}")

    model, selected_model = build_wakeword_model(model_name=model_name, inference_framework=inference_framework)
    frame_predictions = model.predict_clip(str(audio_path), padding=1)

    best_label = ""
    best_score = 0.0
    for frame in frame_predictions:
        if not frame:
            continue
        label, score = max(frame.items(), key=lambda item: float(item[1]))
        score = float(score)
        if score > best_score:
            best_label = str(label)
            best_score = score

    model.reset()
    if best_score < threshold:
        return None

    return WakeWordEvent(
        model_name=selected_model,
        label=best_label or selected_model,
        score=best_score,
        timestamp=time.time(),
    )


class WakeWordListener:
    def __init__(
        self,
        on_wake: Callable[[WakeWordEvent], None],
        model_name: str = WAKEWORD_MODEL,
        input_device: str = WAKEWORD_INPUT_DEVICE,
        sample_rate: int = WAKEWORD_SAMPLE_RATE,
        chunk_size: int = WAKEWORD_CHUNK_SIZE,
        threshold: float = WAKEWORD_THRESHOLD,
        debounce_seconds: float = WAKEWORD_DEBOUNCE_SECONDS,
        inference_framework: str = WAKEWORD_INFERENCE_FRAMEWORK,
    ) -> None:
        self.on_wake = on_wake
        self.sample_rate = int(sample_rate)
        self.chunk_size = int(chunk_size)
        self.threshold = float(threshold)
        self.debounce_seconds = float(debounce_seconds)
        self.inference_framework = str(inference_framework).strip().lower() or "auto"
        self.selected_model = resolve_wakeword_model(model_name)
        self.input_device = resolve_input_device(input_device)

        self._model = None
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._last_detection_at = 0.0
        self._chunks_seen = 0
        self._last_debug_at = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run, name="wakeword-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._model = None

    def pause(self) -> None:
        self._pause_event.set()
        if self._model is not None:
            self._model.reset()

    def resume(self) -> None:
        if self._model is not None:
            self._model.reset()
        self._pause_event.clear()

    def _run(self) -> None:
        logger.info(
            "wake listener started: model=%s device=%s sample_rate=%s chunk_size=%s threshold=%.2f debounce=%.2fs",
            self.selected_model,
            self.input_device if self.input_device is not None else "default",
            self.sample_rate,
            self.chunk_size,
            self.threshold,
            self.debounce_seconds,
        )
        logger.info("wake listener input device detail: %s", describe_input_device(self.input_device))

        try:
            self._model, self.selected_model = build_wakeword_model(
                model_name=self.selected_model,
                inference_framework=self.inference_framework,
            )

            with sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=self.sample_rate,
                dtype="int16",
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            ):
                while not self._stop_event.wait(0.2):
                    pass
        except Exception as exc:
            device_details = list_input_devices() if sd is not None else "sounddevice unavailable"
            logger.exception("wake listener error")
            logger.error("wake listener error detail: %s\nAvailable input devices:\n%s", exc, device_details)
        finally:
            if self._model is not None:
                self._model.reset()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if self._stop_event.is_set() or self._pause_event.is_set() or self._model is None:
            return

        if status:
            logger.warning("wake listener audio status: %s", status)

        try:
            self._chunks_seen += 1
            audio_chunk = np.asarray(indata[:, 0], dtype=np.int16).copy()
            predictions = self._model.predict(
                audio_chunk,
                threshold={self.selected_model: self.threshold},
                debounce_time=self.debounce_seconds,
            )
            label, score = self._pick_detection(predictions)
            now = time.time()

            if logger.isEnabledFor(logging.DEBUG) and now - self._last_debug_at >= 5.0:
                self._last_debug_at = now
                logger.debug(
                    "wake listener heartbeat: chunks=%s top_label=%s top_score=%.3f threshold=%.3f",
                    self._chunks_seen,
                    label or self.selected_model,
                    score,
                    self.threshold,
                )

            if score < self.threshold:
                return

            if now - self._last_detection_at < self.debounce_seconds:
                return

            self._last_detection_at = now
            event = WakeWordEvent(
                model_name=self.selected_model,
                label=label or self.selected_model,
                score=score,
                timestamp=now,
            )
            logger.info(
                "wake word detected: model=%s label=%s score=%.3f",
                event.model_name,
                event.label,
                event.score,
            )
            self.on_wake(event)
        except Exception:
            logger.exception("wake listener error")

    @staticmethod
    def _pick_detection(predictions: dict) -> tuple[str, float]:
        if not predictions:
            return "", 0.0
        label, score = max(predictions.items(), key=lambda item: float(item[1]))
        return str(label), float(score)


def get_model_file_paths(model_name: str) -> tuple[Path, Path]:
    ensure_wakeword_dependencies()

    normalized = _normalize_model_name(model_name)
    if Path(model_name).exists():
        model_path = Path(model_name).resolve()
        if model_path.suffix == ".onnx":
            return model_path, model_path.with_suffix(".tflite")
        return model_path.with_suffix(".onnx"), model_path

    if normalized not in openwakeword.MODELS:
        raise WakeWordError(f"Unknown wake word model {model_name!r}")

    tflite_path = Path(openwakeword.MODELS[normalized]["model_path"]).resolve()
    onnx_path = tflite_path.with_suffix(".onnx")
    return onnx_path, tflite_path


def ensure_model_files(model_name: str) -> tuple[Path, Path]:
    onnx_path, tflite_path = get_model_file_paths(model_name)
    if onnx_path.exists() or tflite_path.exists():
        return onnx_path, tflite_path

    logger.info("wake word model files missing; downloading resources for %s", model_name)
    try:
        download_models([_normalize_model_name(model_name)])
    except Exception as exc:
        raise WakeWordError(f"Failed to download wake word model assets for {model_name!r}: {exc}") from exc

    return onnx_path, tflite_path


def resolve_inference_framework(model_name: str, requested_framework: str) -> str:
    ensure_wakeword_dependencies()
    onnx_path, tflite_path = ensure_model_files(model_name)
    requested = str(requested_framework).strip().lower() or "auto"

    if requested not in {"auto", "onnx", "tflite"}:
        raise WakeWordError("WAKEWORD_INFERENCE_FRAMEWORK must be one of: auto, onnx, tflite")

    if requested in {"auto", "onnx"} and onnx_path.exists():
        return "onnx"

    if requested in {"auto", "tflite"}:
        try:
            import tflite_runtime.interpreter  # noqa: F401
        except Exception:
            if requested == "tflite":
                raise WakeWordError("tflite_runtime is not installed for WAKEWORD_INFERENCE_FRAMEWORK=tflite")
        else:
            if tflite_path.exists():
                return "tflite"

    if requested == "onnx":
        raise WakeWordError(f"ONNX wake word model file is missing for {model_name!r}: {onnx_path}")
    if requested == "tflite":
        raise WakeWordError(f"TFLite wake word model file is missing for {model_name!r}: {tflite_path}")

    raise WakeWordError(
        f"No usable wake word model assets found for {model_name!r}. "
        "ONNX assets are preferred unless tflite_runtime is installed."
    )
