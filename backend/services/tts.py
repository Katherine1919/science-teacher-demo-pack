import asyncio
import logging
import os
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[2]
LATEST_DIR = ROOT / "latest"
ANSWER_AUDIO = LATEST_DIR / "answer.mp3"
WAKEWORD_ACK_AUDIO = LATEST_DIR / "wake_ack.mp3"

TTS_VOICE_ZH = os.getenv("TTS_VOICE_ZH", "zh-CN-XiaoxiaoNeural")
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-US-AvaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")

logger = logging.getLogger("science_teacher.tts")


class TTSError(RuntimeError):
    pass


def pick_voice(language: str) -> str:
    return TTS_VOICE_ZH if str(language).lower().startswith("zh") else TTS_VOICE_EN


async def _save_edge_tts_audio(text: str, output_path: Path, voice: str) -> None:
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=TTS_RATE,
        volume=TTS_VOLUME,
    )
    await communicator.save(str(output_path))


def generate_tts_audio(
    text: str,
    output_path: Path,
    language: str = "zh",
) -> Path:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise TTSError("No answer text provided for TTS")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    voice = pick_voice(language)
    logger.info("TTS started: voice=%s output=%s", voice, output_path)

    try:
        asyncio.run(_save_edge_tts_audio(cleaned_text, output_path, voice))
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        raise TTSError(f"Edge TTS generation failed: {exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TTSError(f"TTS finished but {output_path} was not created.")

    logger.info("TTS done")
    logger.info("file saved: %s", output_path)
    return output_path


def generate_answer_audio(
    text: str,
    output_path: Path = ANSWER_AUDIO,
    language: str = "zh",
) -> Path:
    return generate_tts_audio(text=text, output_path=output_path, language=language)
