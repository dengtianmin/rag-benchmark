from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import logging

from pydantic import ValidationError
from tqdm import tqdm

from benchmark_builder.clients.deepseek_client import DeepSeekClient
from benchmark_builder.config import Settings
from benchmark_builder.models import KnowledgeExtractionPayload, KnowledgeExtractionResult, MarkdownSection
from benchmark_builder.prompts.knowledge_extraction_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from benchmark_builder.utils.io_utils import append_jsonl, load_jsonl_as_models, write_jsonl

logger = logging.getLogger(__name__)


def _load_sections(path: Path) -> list[MarkdownSection]:
    return load_jsonl_as_models(path, MarkdownSection)


def _render_prompt(section: MarkdownSection) -> str:
    return USER_PROMPT_TEMPLATE.format(
        doc_title=section.doc_title,
        section_path=" > ".join(section.section_path),
        content=section.content,
    )


def _safe_parse_payload(raw_text: str) -> tuple[KnowledgeExtractionPayload, str | None]:
    try:
        payload = json.loads(raw_text)
        return KnowledgeExtractionPayload.model_validate(payload), None
    except (json.JSONDecodeError, ValidationError) as exc:
        return KnowledgeExtractionPayload(), str(exc)


def _extract_one(client: DeepSeekClient, section: MarkdownSection, dry_run: bool) -> KnowledgeExtractionResult:
    if dry_run or not client.enabled:
        return KnowledgeExtractionResult(
            doc_id=section.doc_id,
            section_id=section.section_id,
            section_path=section.section_path,
            content=section.content,
            extraction=KnowledgeExtractionPayload(),
            raw_response_text="",
            parse_error=None,
            success=True,
        )

    response = client.chat_completion(system_prompt=SYSTEM_PROMPT, user_prompt=_render_prompt(section))
    extraction, parse_error = _safe_parse_payload(response["content"])
    return KnowledgeExtractionResult(
        doc_id=section.doc_id,
        section_id=section.section_id,
        section_path=section.section_path,
        content=section.content,
        extraction=extraction,
        raw_response_text=response["content"],
        parse_error=parse_error,
        success=parse_error is None,
    )


def run_extract_knowledge(settings: Settings) -> list[KnowledgeExtractionResult]:
    sections_path = settings.artifacts_dir / "markdown_sections.jsonl"
    sections = _load_sections(sections_path)
    output_path = settings.artifacts_dir / "knowledge_extraction.jsonl"

    completed: dict[str, KnowledgeExtractionResult] = {}
    if settings.resume and output_path.exists():
        for item in load_jsonl_as_models(output_path, KnowledgeExtractionResult):
            completed[item.section_id] = item

    pending = [section for section in sections if section.section_id not in completed]
    client = DeepSeekClient(settings)

    results = list(completed.values())
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as executor:
            futures = {
                executor.submit(_extract_one, client, section, settings.dry_run): section.section_id for section in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting knowledge"):
                try:
                    result = future.result()
                    results.append(result)
                    append_jsonl(output_path, [result])
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Knowledge extraction failed: %s", exc)

    results = sorted(results, key=lambda item: item.section_id)
    write_jsonl(output_path, results)
    logger.info("Saved %s extraction results", len(results))
    return results
