import json
import os
import re
from typing import Dict, List

try:
    from .openai_client import build_openai_client, get_default_chat_model, get_llm_provider
    from .prompts import (
        SCIENCE_TEACHER_SYSTEM_PROMPT,
        QUESTION_CLASSIFIER_PROMPT,
        ANSWER_FORMATTER_PROMPT,
        SAFETY_GUARD_PROMPT,
    )
    from .retriever import retrieve_context
except ImportError:
    from openai_client import build_openai_client, get_default_chat_model, get_llm_provider
    from prompts import (
        SCIENCE_TEACHER_SYSTEM_PROMPT,
        QUESTION_CLASSIFIER_PROMPT,
        ANSWER_FORMATTER_PROMPT,
        SAFETY_GUARD_PROMPT,
    )
    from retriever import retrieve_context

CHAT_MODEL = get_default_chat_model()
LLM_PROVIDER = get_llm_provider()
USE_LLM_CLASSIFIER = os.getenv("USE_LLM_CLASSIFIER", "").strip().lower() in {"1", "true", "yes", "on"}
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "140"))
SAFETY_MAX_TOKENS = int(os.getenv("SAFETY_MAX_TOKENS", "120"))
CONTEXT_TOP_K = max(1, int(os.getenv("CONTEXT_TOP_K", "2")))
CONTEXT_CHUNK_MAX_CHARS = max(120, int(os.getenv("CONTEXT_CHUNK_MAX_CHARS", "240")))
CONTEXT_TEXT_MAX_CHARS = max(CONTEXT_CHUNK_MAX_CHARS, int(os.getenv("CONTEXT_TEXT_MAX_CHARS", "640")))
SPOKEN_ANSWER_INCLUDE_FOLLOW_UP = os.getenv("SPOKEN_ANSWER_INCLUDE_FOLLOW_UP", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SPOKEN_ANSWER_MAX_ITEMS = max(1, int(os.getenv("SPOKEN_ANSWER_MAX_ITEMS", "2")))
SPOKEN_ANSWER_MAX_CHARS = max(48, int(os.getenv("SPOKEN_ANSWER_MAX_CHARS", "110")))
FAST_LOCAL_ANSWER_ENABLED = os.getenv("FAST_LOCAL_ANSWER_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

QUESTION_MARKER_KEYWORDS_ZH = [
    "为什么",
    "什么是",
    "怎么",
    "如何",
    "原理",
    "形成",
    "会不会",
    "是不是",
    "吗",
    "呢",
]

QUESTION_MARKER_KEYWORDS_EN = [
    "why",
    "what",
    "how",
    "when",
    "where",
    "can ",
    "does ",
    "do ",
    "?",
]

FAST_SOCIAL_RESPONSES = {
    "zh": {
        "answer_short": "你好，我是你的科学老师，很高兴和你一起探索科学问题。",
        "answer_full": [
            "你可以直接问我科学现象、天气变化、光影、植物或者小实验。",
            "问题越具体，我就越容易回答得清楚。",
        ],
        "follow_up": "你现在最想知道哪个科学问题呢？",
    },
    "en": {
        "answer_short": "Hello, I am your science teacher and I am ready to explore science with you.",
        "answer_full": [
            "You can ask me about weather, light and shadow, plants, animals, or simple experiments.",
            "The more specific your question is, the easier it is for me to answer clearly.",
        ],
        "follow_up": "What science question would you like to ask first?",
    },
}


class AgentError(RuntimeError):
    pass


def get_openai_client():
    return build_openai_client(AgentError)


def get_chat_request_kwargs(*, temperature: float, max_tokens: int | None = None) -> Dict:
    # MiniMax OpenAI-compatible docs use temperatures in (0.0, 1.0].
    normalized_temperature = max(0.1, float(temperature)) if LLM_PROVIDER == "minimax" else float(temperature)
    kwargs: Dict = {"temperature": normalized_temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if LLM_PROVIDER == "minimax":
        kwargs["extra_body"] = {"reasoning_split": True}
    return kwargs


def strip_reasoning_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def normalize_json_text(text: str) -> str:
    cleaned = strip_reasoning_tags(text).strip()

    for _ in range(3):
        updated = re.sub(r"^\s*json\b[\s:]*", "", cleaned, count=1, flags=re.IGNORECASE).strip()
        updated = re.sub(r"^\s*```(?:json)?\s*", "", updated, count=1, flags=re.IGNORECASE).strip()
        updated = re.sub(r"\s*```\s*$", "", updated, count=1, flags=re.IGNORECASE).strip()
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned


def extract_json_object(text: str) -> Dict:
    cleaned = normalize_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", cleaned):
        start = match.start()
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    extracted_fields = extract_schema_fields(cleaned)
    if extracted_fields:
        return extracted_fields

    raise AgentError(f"LLM did not return valid JSON: {cleaned}")


def extract_schema_fields(text: str) -> Dict:
    data: Dict = {}

    for key in ["question_type", "language", "answer_short", "follow_up", "show_image", "safety_note"]:
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
        if match:
            data[key] = json.loads(f'"{match.group(1)}"')

    array_match = re.search(r'"answer_full"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if array_match:
        array_text = f'[{array_match.group(1)}]'
        try:
            data["answer_full"] = json.loads(array_text)
        except json.JSONDecodeError:
            data["answer_full"] = [
                json.loads(f'"{item}"')
                for item in re.findall(r'"((?:\\.|[^"\\])*)"', array_match.group(1), flags=re.DOTALL)
            ]

    return data


def sanitize_show_image(value: str, max_chars: int = 32) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    if not cleaned:
        return ""

    cleaned = re.split(r"[。！？!?\n,，;；]", cleaned, maxsplit=1)[0].strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip(" ,，。")
    return cleaned


def normalize_similarity_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def dedupe_answer_full(answer_short: str, answer_full: List[str]) -> List[str]:
    short_norm = normalize_similarity_text(answer_short)
    short_prefix = short_norm[:18]
    filtered: List[str] = []
    seen: set[str] = set()

    for item in answer_full:
        cleaned = str(item).strip()
        if not cleaned:
            continue

        item_norm = normalize_similarity_text(cleaned)
        if not item_norm or item_norm in seen:
            continue

        if short_norm:
            if item_norm == short_norm:
                continue
            if len(short_prefix) >= 8 and item_norm.startswith(short_prefix):
                continue
            if len(short_norm) >= 12 and item_norm in short_norm:
                continue

        filtered.append(cleaned)
        seen.add(item_norm)

    return filtered


def normalize_formatted_answer(data: Dict, *, question_type: str, language: str) -> Dict:
    answer_short = str(data.get("answer_short", "")).strip()
    answer_full = dedupe_answer_full(
        answer_short,
        [
            str(item).strip()
            for item in data.get("answer_full", [])
            if str(item).strip()
        ],
    )[:3]
    if not answer_full and answer_short:
        answer_full = [answer_short]

    return {
        "question_type": str(data.get("question_type", question_type)).strip() or question_type,
        "language": str(data.get("language", language)).strip() or language,
        "answer_short": answer_short,
        "answer_full": answer_full,
        "follow_up": str(data.get("follow_up", "")).strip(),
        "show_image": sanitize_show_image(data.get("show_image", "")),
        "safety_note": str(data.get("safety_note", "")).strip(),
    }


def detect_language(text: str) -> str:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def normalize_whitespace_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def looks_like_science_question(question: str) -> bool:
    q = normalize_whitespace_text(question).lower()
    if not q:
        return False

    if any(marker in q for marker in QUESTION_MARKER_KEYWORDS_ZH):
        return True
    if any(marker in q for marker in QUESTION_MARKER_KEYWORDS_EN):
        return True
    return q.endswith(("?", "？"))


def heuristic_classify_question(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "unknown"

    social_keywords = [
        "你好", "您好", "谢谢", "再见", "你是谁", "你叫什么", "早上好", "晚上好",
        "hello", "hi", "hey", "thanks", "thank you", "who are you",
    ]
    unsafe_keywords = [
        "爆炸", "炸", "烧", "火", "点燃", "酒精灯", "打火机", "电人", "触电", "硫酸", "刀",
        "explode", "fire", "burn", "acid", "knife", "shock",
    ]
    experiment_keywords = [
        "实验", "怎么做", "步骤", "材料", "观察", "动手", "试一试",
        "experiment", "materials", "steps", "observe",
    ]
    curriculum_keywords = [
        "课本", "教材", "单元", "第几课", "年级", "课堂", "老师要求",
        "textbook", "curriculum", "lesson", "grade",
    ]

    is_question = looks_like_science_question(q)

    if any(keyword in q for keyword in unsafe_keywords):
        return "unsafe"
    if any(keyword in q for keyword in experiment_keywords):
        return "experiment"
    if any(keyword in q for keyword in curriculum_keywords):
        return "curriculum"
    if any(keyword in q for keyword in social_keywords) and not is_question:
        return "social"
    return "concept"


def classify_question(question: str) -> str:
    heuristic_label = heuristic_classify_question(question)
    if heuristic_label != "unknown":
        return heuristic_label
    if not USE_LLM_CLASSIFIER:
        return heuristic_label

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": QUESTION_CLASSIFIER_PROMPT},
            {"role": "user", "content": question},
        ],
        **get_chat_request_kwargs(temperature=0, max_tokens=20),
    )
    label = strip_reasoning_tags(resp.choices[0].message.content or "").strip().lower()
    allowed = {"concept", "curriculum", "experiment", "social", "unsafe", "unknown"}
    return label if label in allowed else heuristic_label


def build_context_text(chunks: List[Dict]) -> str:
    if not chunks:
        return ""

    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        text = normalize_whitespace_text(chunk.get("text", ""))
        if not text:
            continue
        if len(text) > CONTEXT_CHUNK_MAX_CHARS:
            text = text[:CONTEXT_CHUNK_MAX_CHARS].rstrip(" ,，。") + "..."
        parts.append(f"[知识片段 {idx}] {text}")

    context_text = "\n".join(parts).strip()
    if len(context_text) > CONTEXT_TEXT_MAX_CHARS:
        context_text = context_text[:CONTEXT_TEXT_MAX_CHARS].rstrip(" ,，。") + "..."
    return context_text


def format_answer_with_llm(question: str, question_type: str, language: str, context_text: str) -> Dict:
    client = get_openai_client()
    extra_rules = ""
    if question_type == "unsafe":
        extra_rules = "\n请严格遵守安全守则，不要给危险步骤。"

    system_prompt = SCIENCE_TEACHER_SYSTEM_PROMPT + "\n\n" + ANSWER_FORMATTER_PROMPT + extra_rules
    user_content = {
        "question": question,
        "question_type": question_type,
        "language": language,
        "retrieved_context": context_text,
        "required_output_schema": {
            "question_type": "string",
            "language": "zh or en",
            "answer_short": "string",
            "answer_full": ["string", "string", "optional third string"],
            "follow_up": "string",
            "show_image": "short image keyword only",
            "safety_note": "string",
        },
    }

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        **get_chat_request_kwargs(temperature=0, max_tokens=ANSWER_MAX_TOKENS),
    )

    raw = resp.choices[0].message.content or ""
    data = extract_json_object(raw)
    return normalize_formatted_answer(data, question_type=question_type, language=language)


def build_spoken_answer(answer_short: str, answer_full: List[str], follow_up: str = "") -> str:
    parts = []
    seen: set[str] = set()
    char_count = 0

    def append_part(raw_text: str) -> bool:
        nonlocal char_count

        if len(parts) >= SPOKEN_ANSWER_MAX_ITEMS:
            return False

        spoken_item = normalize_whitespace_text(raw_text)
        spoken_norm = normalize_similarity_text(spoken_item)
        if not spoken_item or not spoken_norm or spoken_norm in seen:
            return True

        separator_chars = 1 if parts else 0
        remaining_chars = SPOKEN_ANSWER_MAX_CHARS - char_count - separator_chars
        if remaining_chars <= 0:
            return False

        if len(spoken_item) > remaining_chars:
            if parts:
                return False
            spoken_item = spoken_item[:remaining_chars].rstrip(" ,，。；;：:、")
            if not spoken_item:
                return False
            if len(spoken_item) < len(normalize_whitespace_text(raw_text)):
                spoken_item = spoken_item + "…"

        parts.append(spoken_item)
        seen.add(spoken_norm)
        char_count += len(spoken_item) + separator_chars
        return True

    for item in [answer_short, *answer_full]:
        if not append_part(item):
            break

    if SPOKEN_ANSWER_INCLUDE_FOLLOW_UP:
        follow_up_text = normalize_whitespace_text(follow_up)
        follow_up_norm = normalize_similarity_text(follow_up_text)
        if follow_up_text and follow_up_norm and follow_up_norm not in seen:
            append_part(follow_up_text)

    return "\n".join(parts).strip()


def split_context_sentences(text: str) -> List[str]:
    cleaned = normalize_whitespace_text(text)
    if not cleaned:
        return []

    cleaned = re.sub(r"#+\s*", "", cleaned)
    cleaned = re.sub(r"[*`_>\-]+", " ", cleaned)
    raw_parts = re.split(r"(?<=[。！？!?])\s+|\n+", cleaned)
    sentences: List[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        sentence = normalize_whitespace_text(part)
        sentence = re.sub(r"^[一二三四五六七八九0-9]+年级\s+\S+\s*", "", sentence).strip()
        if len(sentence) < 8:
            continue
        norm = normalize_similarity_text(sentence)
        if not norm or norm in seen:
            continue
        if sentence.lower().startswith(("core ", "grade ", "unit ")):
            continue
        sentences.append(sentence)
        seen.add(norm)

    return sentences


def pick_follow_up(question_type: str, language: str) -> str:
    if str(language).lower().startswith("zh"):
        if question_type == "experiment":
            return "你愿意在老师或家长陪同下继续观察这个现象吗？"
        return "你还想再观察一个生活中的例子吗？"

    if question_type == "experiment":
        return "Would you like to observe this with a teacher or parent?"
    return "Would you like to find one more example in daily life?"


def retrieval_is_confident(context_chunks: List[Dict]) -> bool:
    return bool(context_chunks and context_chunks[0].get("retrieval_confident"))


def build_social_response(language: str) -> Dict:
    lang_key = "zh" if str(language).lower().startswith("zh") else "en"
    result = dict(FAST_SOCIAL_RESPONSES[lang_key])
    result["question_type"] = "social"
    result["language"] = lang_key
    result["show_image"] = ""
    result["safety_note"] = ""
    return result


def build_fast_local_answer(question_type: str, language: str, context_chunks: List[Dict]) -> Dict | None:
    if not context_chunks or not retrieval_is_confident(context_chunks):
        return None

    candidate_sentences: List[str] = []
    for chunk in context_chunks:
        candidate_sentences.extend(split_context_sentences(chunk.get("text", "")))

    if not candidate_sentences:
        return None

    answer_short = candidate_sentences[0]
    answer_full = dedupe_answer_full(answer_short, candidate_sentences[1:4])[:3]
    if not answer_full:
        answer_full = [answer_short]

    show_image = ""
    for chunk in context_chunks:
        tags = [str(tag).strip() for tag in chunk.get("tags", []) if str(tag).strip()]
        for tag in tags:
            if tag.lower() in {"core", "extended"}:
                continue
            if tag.lower().startswith("grade"):
                continue
            if "_" in tag:
                continue
            if len(tag) > 32:
                continue
            show_image = tag
            break
        if show_image:
            break

    return {
        "question_type": question_type,
        "language": language,
        "answer_short": answer_short,
        "answer_full": answer_full,
        "follow_up": pick_follow_up(question_type, language),
        "show_image": sanitize_show_image(show_image),
        "safety_note": "",
        "fast_answer_confident": True,
    }


def build_answer_preview(user_question: str, grade: str = "auto", language: str = "auto", mode: str = "science") -> Dict | None:
    if language == "auto":
        language = detect_language(user_question)

    question_type = heuristic_classify_question(user_question)

    if question_type == "social":
        result = build_social_response(language)
        result["retrieved_chunks"] = []
    elif question_type in {"concept", "curriculum", "experiment"}:
        context_chunks = retrieve_context(user_question, grade=grade, top_k=CONTEXT_TOP_K)
        result = build_fast_local_answer(question_type, language, context_chunks)
        if result is None:
            return None
        result["retrieved_chunks"] = context_chunks
    else:
        return None

    result["mode"] = mode
    result["grade"] = grade
    result["original_question"] = user_question
    result["spoken_answer"] = build_spoken_answer(
        result.get("answer_short", ""),
        result.get("answer_full", []),
        result.get("follow_up", ""),
    )
    result["preview"] = True
    return result


def run_agent(user_question: str, grade: str = "auto", language: str = "auto", mode: str = "science") -> Dict:
    if language == "auto":
        language = detect_language(user_question)

    question_type = classify_question(user_question)

    if question_type == "social":
        result = build_social_response(language)
        result["retrieved_chunks"] = []
        result["mode"] = mode
        result["grade"] = grade
        result["original_question"] = user_question
        result["spoken_answer"] = build_spoken_answer(
            result.get("answer_short", ""),
            result.get("answer_full", []),
            result.get("follow_up", ""),
        )
        return result

    if question_type == "unsafe":
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SAFETY_GUARD_PROMPT},
                {"role": "user", "content": user_question},
            ],
            **get_chat_request_kwargs(temperature=0.2, max_tokens=SAFETY_MAX_TOKENS),
        )
        safe_text = strip_reasoning_tags(resp.choices[0].message.content or "").strip()
        return {
            "question_type": "unsafe",
            "language": language,
            "answer_short": safe_text,
            "answer_full": [safe_text],
            "follow_up": "你想不想试一个更安全的科学小实验？" if language == "zh" else "Would you like to try a safer science activity instead?",
            "show_image": "",
            "safety_note": safe_text,
        }

    context_chunks = []
    if question_type in {"concept", "curriculum", "experiment"}:
        context_chunks = retrieve_context(user_question, grade=grade, top_k=CONTEXT_TOP_K)
        if not retrieval_is_confident(context_chunks):
            context_chunks = []

    if FAST_LOCAL_ANSWER_ENABLED and question_type in {"concept", "curriculum", "experiment"}:
        local_result = build_fast_local_answer(question_type, language, context_chunks)
        if local_result is not None:
            local_result["retrieved_chunks"] = context_chunks
            local_result["mode"] = mode
            local_result["grade"] = grade
            local_result["original_question"] = user_question
            local_result["spoken_answer"] = build_spoken_answer(
                local_result.get("answer_short", ""),
                local_result.get("answer_full", []),
                local_result.get("follow_up", ""),
            )
            return local_result

    context_text = build_context_text(context_chunks)
    result = format_answer_with_llm(
        question=user_question,
        question_type=question_type,
        language=language,
        context_text=context_text,
    )
    result["retrieved_chunks"] = context_chunks
    result["mode"] = mode
    result["grade"] = grade
    result["original_question"] = user_question
    result["spoken_answer"] = build_spoken_answer(
        result.get("answer_short", ""),
        result.get("answer_full", []),
        result.get("follow_up", ""),
    )
    return result
