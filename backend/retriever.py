import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
KB_PROCESSED_PATH = ROOT / "kb" / "processed" / "chunks.jsonl"

GENERIC_QUERY_PHRASES = (
    "是怎么形成的",
    "为什么会有",
    "为什么会",
    "是什么原因",
    "什么原因",
    "什么是",
    "是什么",
    "怎么形成的",
    "怎么形成",
    "怎么来的",
    "告诉我",
    "解释一下",
    "请解释",
    "请问",
    "为什么",
    "如何",
    "怎么",
    "原理",
    "形成",
    "原因",
    "作用",
    "过程",
    "现象",
    "吗",
    "呢",
)
GENERIC_QUERY_TERMS = {
    "什么",
    "为什么",
    "怎么",
    "如何",
    "形成",
    "原因",
    "原理",
    "作用",
    "过程",
    "现象",
    "东西",
}
MIN_RETRIEVAL_SCORE = 3
CONFIDENT_RETRIEVAL_SCORE = 7


@lru_cache(maxsize=1)
def load_chunks() -> List[Dict]:
    if not KB_PROCESSED_PATH.exists():
        return []

    chunks = []
    with KB_PROCESSED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def normalize_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def strip_query_boilerplate(question: str) -> str:
    cleaned = normalize_text(question)
    stripped = cleaned
    for phrase in GENERIC_QUERY_PHRASES:
        stripped = stripped.replace(normalize_text(phrase), "")
    return stripped.strip()


def extract_query_terms(question: str) -> List[str]:
    raw = normalize_text(question)
    core = strip_query_boilerplate(question)
    source = core or raw
    candidates: List[str] = []

    if core and core not in GENERIC_QUERY_TERMS:
        candidates.append(core)

    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in source)
    if has_cjk:
        if len(source) == 1 and source not in GENERIC_QUERY_TERMS:
            candidates.append(source)
        max_size = min(4, len(source))
        for size in range(max_size, 1, -1):
            for idx in range(len(source) - size + 1):
                token = source[idx : idx + size]
                if token in GENERIC_QUERY_TERMS:
                    continue
                candidates.append(token)
    else:
        for token in re.split(r"[^a-z0-9]+", source):
            if len(token) >= 3 and token not in GENERIC_QUERY_TERMS:
                candidates.append(token)

    seen = set()
    result = []
    for token in candidates:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result[:10]


def score_chunk(question: str, chunk: Dict) -> tuple[int, List[str], bool]:
    terms = extract_query_terms(question)
    if not terms:
        return 0, [], False

    text_norm = normalize_text(chunk.get("text", ""))
    topic_norm = normalize_text(chunk.get("topic", ""))
    tag_norms = {normalize_text(tag) for tag in chunk.get("tags", []) if normalize_text(tag)}

    score = 0
    hits: List[str] = []
    strong_match = False
    primary_term = terms[0]

    for idx, term in enumerate(terms):
        term_score = 0
        term_is_strong = False

        if term in tag_norms:
            term_score = 8
            term_is_strong = True
        elif term and term == topic_norm:
            term_score = 7
            term_is_strong = True
        elif term and term in topic_norm:
            term_score = 6
            term_is_strong = True
        elif term and term in text_norm:
            if len(term) >= 3:
                term_score = 5
            elif len(term) == 2:
                term_score = 3
            else:
                term_score = 2
            term_is_strong = idx == 0 and len(term) >= 2

        if term_score <= 0:
            continue

        if idx == 0:
            term_score += 2

        score += term_score
        hits.append(term)
        strong_match = strong_match or term_is_strong

    unique_hits = list(dict.fromkeys(hits))
    confident = strong_match and score >= CONFIDENT_RETRIEVAL_SCORE
    return score, unique_hits, confident


def retrieve_context(question: str, grade: str = "auto", top_k: int = 3) -> List[Dict]:
    chunks = load_chunks()
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        chunk_grade = str(chunk.get("grade", ""))
        score, hits, confident = score_chunk(question, chunk)

        if grade != "auto" and grade and chunk_grade == str(grade):
            score += 1

        if score >= MIN_RETRIEVAL_SCORE:
            enriched_chunk = dict(chunk)
            enriched_chunk["retrieval_score"] = score
            enriched_chunk["retrieval_hits"] = hits
            enriched_chunk["retrieval_confident"] = confident
            scored.append((score, int(confident), enriched_chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:top_k]]
