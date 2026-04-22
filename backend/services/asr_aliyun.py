import json
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
LATEST_DIR = ROOT / "latest"
QUESTION_AUDIO = LATEST_DIR / "question.wav"

DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
ALIYUN_ASR_MODEL = os.getenv("ALIYUN_ASR_MODEL", "fun-asr")
ALIYUN_ASR_LANGUAGE = os.getenv("ALIYUN_ASR_LANGUAGE", "zh").strip()
ALIYUN_ASR_TIMEOUT_SECONDS = float(os.getenv("ALIYUN_ASR_TIMEOUT_SECONDS", "300"))
ALIYUN_ASR_POLL_INTERVAL_SECONDS = max(0.2, float(os.getenv("ALIYUN_ASR_POLL_INTERVAL_SECONDS", "0.25")))

logger = logging.getLogger("science_teacher.asr_aliyun")


class ASRError(RuntimeError):
    pass


class ASRNoSpeechError(ASRError):
    pass


def get_dashscope_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ASRError("DASHSCOPE_API_KEY is not set")
    return api_key


def build_headers(api_key: str, *, include_json: bool = True) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    if include_json:
        headers["Content-Type"] = "application/json"
    return headers


def get_upload_policy(client: httpx.Client, api_key: str, model_name: str) -> dict:
    response = client.get(
        f"{DASHSCOPE_BASE_URL}/uploads",
        headers=build_headers(api_key),
        params={"action": "getPolicy", "model": model_name},
    )
    response.raise_for_status()
    payload = response.json()
    return payload["data"]


def upload_file_to_temporary_oss(client: httpx.Client, policy_data: dict, file_path: Path) -> str:
    file_path = Path(file_path).resolve()
    suffix = file_path.suffix or ".bin"
    object_name = f"{uuid.uuid4().hex}{suffix}"
    object_key = f"{policy_data['upload_dir']}/{object_name}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    with file_path.open("rb") as audio_file:
        files = {
            "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
            "Signature": (None, policy_data["signature"]),
            "policy": (None, policy_data["policy"]),
            "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
            "key": (None, object_key),
            "success_action_status": (None, "200"),
            "file": (file_path.name, audio_file, content_type),
        }
        response = client.post(policy_data["upload_host"], files=files)
    response.raise_for_status()
    return f"oss://{object_key}"


def submit_transcription_task(
    client: httpx.Client,
    api_key: str,
    file_url: str,
    model_name: str,
    language_hint: str = "",
) -> str:
    payload = {
        "model": model_name,
        "input": {"file_urls": [file_url]},
        "parameters": {"channel_id": [0]},
    }
    if language_hint:
        payload["parameters"]["language_hints"] = [language_hint]

    headers = build_headers(api_key)
    headers["X-DashScope-Async"] = "enable"
    headers["X-DashScope-OssResourceResolve"] = "enable"

    logger.info("request sent")
    response = client.post(
        f"{DASHSCOPE_BASE_URL}/services/audio/asr/transcription",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()
    body = response.json()
    logger.info("response received")

    task_id = body.get("output", {}).get("task_id")
    if not task_id:
        raise ASRError(f"Aliyun ASR task_id missing in response: {body}")
    return task_id


def fetch_task_result(client: httpx.Client, api_key: str, task_id: str) -> dict:
    url = f"{DASHSCOPE_BASE_URL}/tasks/{task_id}"
    headers = build_headers(api_key, include_json=False)

    # The official docs label this endpoint as POST, but the cURL sample omits -X POST.
    # Try GET first to match the sample and fall back to POST if the server rejects it.
    response = client.get(url, headers=headers)
    if response.status_code == 405:
        response = client.post(url, headers=headers)
    response.raise_for_status()
    return response.json()


def wait_for_task_result(client: httpx.Client, api_key: str, task_id: str) -> dict:
    deadline = time.time() + ALIYUN_ASR_TIMEOUT_SECONDS

    while time.time() < deadline:
        body = fetch_task_result(client, api_key, task_id)
        output = body.get("output", {})
        task_status = output.get("task_status")
        if task_status in {"SUCCEEDED", "FAILED"}:
            logger.info("response received")
            return body
        time.sleep(ALIYUN_ASR_POLL_INTERVAL_SECONDS)

    raise ASRError(f"Aliyun ASR timed out waiting for task {task_id}")


def get_transcription_url(task_body: dict) -> str:
    output = task_body.get("output", {})
    task_status = output.get("task_status")
    if task_status != "SUCCEEDED":
        task_code = str(output.get("code", "")).strip()
        task_message = str(output.get("message", "")).strip()
        if task_code == "ASR_RESPONSE_HAVE_NO_WORDS" or task_message == "ASR_RESPONSE_HAVE_NO_WORDS":
            raise ASRNoSpeechError(f"Aliyun ASR detected no speech: {json.dumps(task_body, ensure_ascii=False)}")
        raise ASRError(f"Aliyun ASR task failed: {json.dumps(task_body, ensure_ascii=False)}")

    results = output.get("results", [])
    for result in results:
        if result.get("subtask_status") == "SUCCEEDED" and result.get("transcription_url"):
            return result["transcription_url"]

    raise ASRError(f"Aliyun ASR returned no successful transcription result: {json.dumps(task_body, ensure_ascii=False)}")


def download_transcript_json(client: httpx.Client, transcription_url: str) -> dict:
    response = client.get(transcription_url)
    response.raise_for_status()
    return response.json()


def extract_text_from_transcript(transcript_json: dict) -> str:
    parts = []
    for item in transcript_json.get("transcripts", []):
        text = str(item.get("text", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def transcribe_question(audio_path: Path = QUESTION_AUDIO) -> str:
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise ASRError(f"Audio file not found: {audio_path}")

    logger.info("ASR started: %s", audio_path)
    api_key = get_dashscope_api_key()

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, read=60.0)) as client:
            policy_data = get_upload_policy(client, api_key, ALIYUN_ASR_MODEL)
            file_url = upload_file_to_temporary_oss(client, policy_data, audio_path)
            task_id = submit_transcription_task(
                client,
                api_key,
                file_url,
                ALIYUN_ASR_MODEL,
                ALIYUN_ASR_LANGUAGE,
            )
            task_body = wait_for_task_result(client, api_key, task_id)
            transcript_json = download_transcript_json(client, get_transcription_url(task_body))
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip() if exc.response is not None else str(exc)
        logger.exception("ASR failed")
        raise ASRError(f"Aliyun ASR HTTP error: {detail}") from exc
    except ASRError:
        logger.exception("ASR failed")
        raise
    except Exception as exc:
        logger.exception("ASR failed")
        raise ASRError(f"Aliyun ASR failed: {exc}") from exc

    text = extract_text_from_transcript(transcript_json)
    if not text:
        logger.error("ASR failed")
        raise ASRNoSpeechError(f"Aliyun ASR returned empty text: {json.dumps(transcript_json, ensure_ascii=False)}")

    logger.info("recognized text: %s", text)
    return text
