from __future__ import annotations

from pathlib import Path
import re

from benchmark_builder.models import MarkdownSection
from benchmark_builder.utils.id_utils import stable_id


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def classify_block_type(content: str) -> str:
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return "paragraph"

    has_table = any("<table" in line.lower() or "|" in line for line in lines)
    has_list = any(line.lstrip().startswith(("-", "*", "+")) or re.match(r"^\d+\.", line.strip()) for line in lines)
    if has_table and has_list:
        return "mixed"
    if has_table:
        return "table"
    if has_list:
        return "list"
    return "paragraph"


def clean_doc_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            return match.group(2).strip()
    return path.stem


def parse_markdown_file(path: Path) -> list[MarkdownSection]:
    text = path.read_text(encoding="utf-8")
    doc_title = clean_doc_title(path, text)
    doc_id = stable_id(str(path.resolve()), prefix="doc_")

    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    section_counter = 0

    def flush_section() -> None:
        nonlocal current_lines, current_path, section_counter
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        section_counter += 1
        section_id = stable_id(doc_id, "/".join(current_path), str(section_counter), prefix="sec_")
        sections.append(
            MarkdownSection(
                doc_id=doc_id,
                file_path=str(path),
                doc_title=doc_title,
                section_id=section_id,
                block_id=section_id,
                section_path=current_path[:],
                content=content,
                block_type=classify_block_type(content),
                char_len=len(content),
            )
        )
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
            current_path = [item[1] for item in heading_stack]
            continue

        if line.strip().startswith("!["):
            continue
        current_lines.append(line)

    flush_section()
    if not sections and text.strip():
        section_id = stable_id(doc_id, "full", prefix="sec_")
        sections.append(
            MarkdownSection(
                doc_id=doc_id,
                file_path=str(path),
                doc_title=doc_title,
                section_id=section_id,
                block_id=section_id,
                section_path=[doc_title],
                content=text.strip(),
                block_type=classify_block_type(text),
                char_len=len(text.strip()),
            )
        )
    return sections
