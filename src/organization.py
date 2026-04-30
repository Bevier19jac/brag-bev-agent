from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class OrganizedFile:
    file_name: str
    file_type: str
    source_path: str
    created_date: str
    modified_date: str
    preview: str
    category: str
    confidence: float
    reason: str
    suggested_file_name: str
    suggested_destination_folder: str
    duplicate_group: str


CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Legal / IP / Patent / NDA", ["nda", "patent", "assignment", "security agreement", "operating agreement", "docusign", "provisional"]),
    ("Investor / Business / Pitch / Financial", ["investor", "pitch", "business plan", "financial", "invoice", "bank", "capitalization", "term sheet"]),
    ("Vendor / Partner Communications", ["vendor", "partner", "email", "thread", "communications", "chat"]),
    ("Product Development / CAD / Engineering", ["cad", "engineering", "design", "sketch", "template", "mechanism", "batchcards"]),
    ("R&D / Testing / Lab Results", ["test", "report", "coverage", "labs", "are labs", "deposition", "run sheet"]),
    ("Formulation / Aerosol / Chemistry / Regulatory", ["formulation", "aerosol", "chemical", "ingredients", "spray", "reg", "pto"]),
    ("Images / Specs / Renderings", ["pic", "image", "render", "spec", "specsheet", "photo"]),
    ("Operations / Admin / Internal Planning", ["checklist", "admin", "internal", "planning", "workflow", "status"]),
]


CATEGORY_FOLDERS = {
    "Legal / IP / Patent / NDA": "organized/legal_ip",
    "Investor / Business / Pitch / Financial": "organized/investor_business",
    "Vendor / Partner Communications": "organized/vendor_partner_comms",
    "Product Development / CAD / Engineering": "organized/product_development",
    "R&D / Testing / Lab Results": "organized/rd_testing",
    "Formulation / Aerosol / Chemistry / Regulatory": "organized/formulation_chemistry",
    "Images / Specs / Renderings": "organized/images_specs",
    "Operations / Admin / Internal Planning": "organized/operations_admin",
    "Uncategorized / Needs Review": "organized/needs_review",
}


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return cleaned[:120] or "document"


def _extract_preview(path: Path, max_chars: int = 280) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars]


def _categorize(file_name: str, preview: str, source_path: str) -> tuple[str, float, str]:
    haystack = f"{file_name} {source_path} {preview}".lower()
    best = ("Uncategorized / Needs Review", 0.2, "No strong keyword match")
    for category, keywords in CATEGORY_RULES:
        matches = [kw for kw in keywords if kw in haystack]
        if matches:
            score = min(0.55 + 0.1 * len(matches), 0.98)
            reason = f"Matched keywords: {', '.join(matches[:4])}"
            if score > best[1]:
                best = (category, score, reason)
    return best


def _iso_or_blank(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.utcfromtimestamp(ts).isoformat() + "Z"


def build_organized_index(processed_dir: Path) -> list[OrganizedFile]:
    files = sorted(path for path in processed_dir.rglob("*.txt") if path.is_file())
    rows: list[OrganizedFile] = []

    for path in files:
        stat = path.stat()
        preview = _extract_preview(path)
        category, confidence, reason = _categorize(path.name, preview, str(path))
        extension = path.name.split("__")[-1].replace(".txt", "") if "__" in path.name else path.suffix.lstrip(".")
        cleaned_name = f"{_slugify(path.stem)}.{extension}"
        rows.append(
            OrganizedFile(
                file_name=path.name,
                file_type=extension,
                source_path=str(path),
                created_date=_iso_or_blank(getattr(stat, "st_ctime", None)),
                modified_date=_iso_or_blank(getattr(stat, "st_mtime", None)),
                preview=preview,
                category=category,
                confidence=round(confidence, 2),
                reason=reason,
                suggested_file_name=cleaned_name,
                suggested_destination_folder=CATEGORY_FOLDERS.get(category, CATEGORY_FOLDERS["Uncategorized / Needs Review"]),
                duplicate_group="",
            )
        )

    _apply_duplicate_groups(rows)
    return rows


def _apply_duplicate_groups(rows: list[OrganizedFile]) -> None:
    group_id = 1
    used = set()
    for i, row in enumerate(rows):
        if i in used:
            continue
        current_group = []
        for j in range(i + 1, len(rows)):
            if j in used:
                continue
            name_sim = SequenceMatcher(None, row.file_name.lower(), rows[j].file_name.lower()).ratio()
            preview_sim = SequenceMatcher(None, row.preview[:180], rows[j].preview[:180]).ratio() if row.preview and rows[j].preview else 0.0
            if name_sim >= 0.9 or preview_sim >= 0.92:
                current_group.append(j)
        if current_group:
            tag = f"DUP-{group_id:03d}"
            row.duplicate_group = tag
            for idx in current_group:
                rows[idx].duplicate_group = tag
                used.add(idx)
            used.add(i)
            group_id += 1


def to_csv(rows: list[OrganizedFile]) -> str:
    import csv
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
    if rows:
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return buffer.getvalue()


def to_json(rows: list[OrganizedFile]) -> str:
    return json.dumps([asdict(row) for row in rows], indent=2)
