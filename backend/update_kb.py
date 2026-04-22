# backend/update_kb.py

import json
import re
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "kb" / "raw"
PROCESSED_DIR = ROOT / "kb" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def guess_metadata(path: Path) -> Dict:
    filename = path.stem.lower()
    grade = ""
    match = re.search(r"grade(\d+)", filename)
    if match:
        grade = match.group(1)
    topic = path.stem
    unit = path.parent.name
    source = str(path)
    return {"grade": grade, "unit": unit, "topic": topic, "source": source}


def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= chunk_size:
            current += ("\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def make_tags(text: str, topic: str, unit: str) -> List[str]:
    tags = set()
    if topic:
        tags.add(topic.lower())
    if unit:
        tags.add(unit.lower())
    for kw in ["水循环", "蒸发", "凝结", "降水", "影子", "植物", "阳光", "电路", "天气", "重力", "引力", "下落"]:
        if kw in text:
            tags.add(kw)
    return sorted(tags)


def collect_source_files() -> List[Path]:
    files = []
    if not RAW_DIR.exists():
        return files
    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def build_chunks() -> List[Dict]:
    all_chunks = []
    source_files = collect_source_files()
    for path in source_files:
        raw = clean_text(read_text_file(path))
        if not raw:
            continue
        meta = guess_metadata(path)
        split_chunks = split_text_into_chunks(raw, chunk_size=500)
        for idx, chunk_text in enumerate(split_chunks, start=1):
            all_chunks.append({
                "id": f"{path.stem}_{idx:03d}",
                "source": meta["source"],
                "grade": meta["grade"],
                "unit": meta["unit"],
                "topic": meta["topic"],
                "tags": make_tags(chunk_text, meta["topic"], meta["unit"]),
                "text": chunk_text,
                "is_experiment": "实验" in chunk_text or "experiment" in chunk_text.lower(),
                "is_safety_related": "安全" in chunk_text or "danger" in chunk_text.lower(),
            })
    return all_chunks


def save_chunks(chunks: List[Dict]) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main():
    chunks = build_chunks()
    save_chunks(chunks)
    print(f"✅ knowledge base updated: {len(chunks)} chunks written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
