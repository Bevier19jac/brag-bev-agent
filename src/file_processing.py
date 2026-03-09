from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SOURCE_DIR = Path("data/source")
PROCESSED_DIR = Path("data/raw")
SUPPORTED_SOURCE_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}


@dataclass
class FileProcessingResult:
    source_path: Path
    output_path: Path | None
    status: str
    message: str


@dataclass
class IngestionSummary:
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    processed_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    error_files: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def ensure_data_directories() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    value = value.strip() or "document"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def processed_filename_for(file_path: Path) -> str:
    try:
        relative_path = file_path.relative_to(SOURCE_DIR)
        pieces = list(relative_path.with_suffix("").parts)
    except ValueError:
        pieces = list(file_path.with_suffix("").parts)

    safe_base = "__".join(sanitize_name(piece) for piece in pieces if piece)
    extension_label = file_path.suffix.lower().lstrip(".") or "file"
    return f"{safe_base}__{extension_label}.txt"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    raw = file_path.read_bytes()

    if suffix in {".txt", ".md"}:
        return raw.decode("utf-8", errors="replace")

    if suffix == ".json":
        decoded = raw.decode("utf-8", errors="replace")
        try:
            return json.dumps(json.loads(decoded), indent=2, ensure_ascii=True)
        except json.JSONDecodeError:
            return decoded

    if suffix == ".csv":
        decoded = raw.decode("utf-8", errors="replace")
        reader = csv.reader(StringIO(decoded))
        return "\n".join(" | ".join(row) for row in reader)

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        page_text = []
        for page in reader.pages:
            page_text.append(page.extract_text() or "")
        return "\n\n".join(page_text)

    if suffix == ".docx":
        document = DocxDocument(BytesIO(raw))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


def process_source_file(source_path: Path, processed_dir: Path) -> FileProcessingResult:
    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_SOURCE_EXTENSIONS:
        return FileProcessingResult(
            source_path=source_path,
            output_path=None,
            status="skipped",
            message=f"Skipped unsupported file type: {source_path.name}",
        )

    try:
        extracted_text = extract_text_from_file(source_path)
        normalized = normalize_text(extracted_text)
    except Exception as exc:
        return FileProcessingResult(
            source_path=source_path,
            output_path=None,
            status="error",
            message=f"Failed to process {source_path.name}: {exc}",
        )

    if not normalized:
        return FileProcessingResult(
            source_path=source_path,
            output_path=None,
            status="skipped",
            message=f"Skipped empty or unreadable file: {source_path.name}",
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / processed_filename_for(source_path)
    try:
        relative_path = source_path.relative_to(SOURCE_DIR)
    except ValueError:
        relative_path = source_path

    metadata_header = "\n".join(
        [
            f"Source filename: {source_path.name}",
            f"Source relative path: {relative_path.as_posix()}",
            f"Source type: {suffix}",
            f"Processed at: {datetime.now(UTC).isoformat()}",
            "",
        ]
    )
    output_path.write_text(metadata_header + normalized + "\n", encoding="utf-8")

    return FileProcessingResult(
        source_path=source_path,
        output_path=output_path,
        status="processed",
        message=f"Processed {source_path.name} -> {output_path.name}",
    )


def ingest_source_directory(source_dir: Path, processed_dir: Path) -> IngestionSummary:
    summary = IngestionSummary()

    if not source_dir.exists():
        summary.messages.append(f"Source directory not found: {source_dir}")
        return summary

    source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if not source_files:
        summary.messages.append(f"No files found in source directory: {source_dir}")
        return summary

    for source_path in source_files:
        result = process_source_file(source_path, processed_dir)
        summary.messages.append(result.message)

        if result.status == "processed":
            summary.processed += 1
            summary.processed_files.append(result.output_path.name if result.output_path else source_path.name)
        elif result.status == "skipped":
            summary.skipped += 1
            summary.skipped_files.append(source_path.name)
        else:
            summary.errors += 1
            summary.error_files.append(source_path.name)

    return summary


def count_supported_source_files(source_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    return sum(
        1
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
    )


def get_processed_file_count(processed_dir: Path) -> int:
    if not processed_dir.exists():
        return 0
    return sum(1 for path in processed_dir.rglob("*.txt") if path.is_file())
