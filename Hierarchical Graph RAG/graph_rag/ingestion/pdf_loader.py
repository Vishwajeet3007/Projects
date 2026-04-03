from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from graph_rag.models import PageRecord


logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | Path) -> tuple[str, str, list[PageRecord]]:
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    doc_id = path.stem.lower().replace(" ", "_")
    pages: list[PageRecord] = []

    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        page_id = f"{doc_id}-page-{index}"
        pages.append(PageRecord(page_id=page_id, page_number=index, text=text))

    logger.info("Loaded PDF '%s' with %s pages", path.name, len(pages))
    return doc_id, path.name, pages
