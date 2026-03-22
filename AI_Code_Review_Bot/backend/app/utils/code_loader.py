"""Helpers for normalizing code inputs."""

from __future__ import annotations

from fastapi import UploadFile

from app.schemas.review import CodeFile, SourceBundle


async def bundle_from_upload(upload_file: UploadFile, language: str | None = None) -> SourceBundle:
    """Build a source bundle from an uploaded file."""

    content = (await upload_file.read()).decode("utf-8", errors="ignore")
    return SourceBundle(
        source_type="raw_code",
        identifier=upload_file.filename or "uploaded_file",
        files=[
            CodeFile(
                path=upload_file.filename or "uploaded_file",
                content=content,
                language=language,
            )
        ],
        metadata={},
    )



def bundle_from_raw_code(code: str, filename: str, language: str) -> SourceBundle:
    """Build a source bundle from a raw code snippet."""

    return SourceBundle(
        source_type="raw_code",
        identifier=filename,
        files=[CodeFile(path=filename, content=code, language=language)],
        metadata={},
    )
