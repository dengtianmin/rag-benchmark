from __future__ import annotations

from pathlib import Path
import logging

from tqdm import tqdm

from benchmark_builder.config import Settings
from benchmark_builder.models import MarkdownSection
from benchmark_builder.utils.io_utils import write_jsonl
from benchmark_builder.utils.markdown_parser import parse_markdown_file

logger = logging.getLogger(__name__)


def collect_markdown_files(input_dir: Path, max_files: int | None = None) -> list[Path]:
    files = sorted(input_dir.rglob("*.md"))
    return files[:max_files] if max_files else files


def run_parse_markdown(settings: Settings) -> list[MarkdownSection]:
    files = collect_markdown_files(settings.input_dir, settings.max_files)
    sections: list[MarkdownSection] = []
    for path in tqdm(files, desc="Parsing markdown"):
        try:
            sections.extend(parse_markdown_file(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse markdown file %s: %s", path, exc)

    output_path = settings.artifacts_dir / "markdown_sections.jsonl"
    write_jsonl(output_path, sections)
    logger.info("Parsed %s sections from %s markdown files", len(sections), len(files))
    return sections
