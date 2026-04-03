from __future__ import annotations

import re
from typing import Iterable


def _normalize_paragraphs(page_text: str) -> list[str]:
    raw_parts = re.split(r"\n\s*\n+", page_text)
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in raw_parts]
    return [part for part in paragraphs if len(part) > 20]


def split_into_sections(page_text: str) -> list[str]:
    paragraphs = _normalize_paragraphs(page_text)
    if not paragraphs and page_text.strip():
        paragraphs = [re.sub(r"\s+", " ", page_text).strip()]

    if not paragraphs:
        return []

    sections: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(" ".join(buffer).strip())
            buffer.clear()

    for paragraph in paragraphs:
        if len(paragraph) < 120 and buffer:
            buffer.append(paragraph)
            continue
        if buffer:
            flush()
        buffer.append(paragraph)

    flush()
    return sections


def batched_sections(section_texts: Iterable[str], batch_size: int = 20) -> list[list[str]]:
    batch: list[str] = []
    result: list[list[str]] = []
    for item in section_texts:
        batch.append(item)
        if len(batch) >= batch_size:
            result.append(batch)
            batch = []
    if batch:
        result.append(batch)
    return result
