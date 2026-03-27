from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from benchmark_builder.clients.deepseek_client import LLMClient
from benchmark_builder.config import Settings
from benchmark_builder.models import KnowledgeExtractionResult, MarkdownSection, QACandidate
from benchmark_builder.prompts.qa_generation_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from benchmark_builder.utils.id_utils import stable_id
from benchmark_builder.utils.io_utils import append_jsonl, load_jsonl_as_models, write_jsonl
from benchmark_builder.utils.text_utils import is_quote_relevant, normalize_for_similarity, similarity

logger = logging.getLogger(__name__)


class QACandidatePayload(BaseModel):
    question: str
    answer_short: str
    answer_long: str
    question_type: str
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requires_text_compensation: bool = False
    evidence: list[dict] = Field(default_factory=list)
    source_scope: str = "single_section"


class QAGenerationEnvelope(BaseModel):
    items: list[QACandidatePayload] = Field(default_factory=list)


class QASectionProgress(BaseModel):
    section_id: str
    status: str
    qa_count: int = 0


def _load_inputs(settings: Settings) -> tuple[dict[str, MarkdownSection], list[KnowledgeExtractionResult]]:
    sections = {
        item.section_id: item
        for item in load_jsonl_as_models(settings.artifacts_dir / "markdown_sections.jsonl", MarkdownSection)
    }
    extractions = load_jsonl_as_models(settings.artifacts_dir / "knowledge_extraction.jsonl", KnowledgeExtractionResult)
    return sections, extractions


def _render_prompt(section: MarkdownSection, extraction: KnowledgeExtractionResult, settings: Settings) -> str:
    return USER_PROMPT_TEMPLATE.format(
        max_per_section=settings.qa_generation.max_per_section,
        type_quota=settings.qa_generation.type_quota.model_dump_json(),
        allow_cross_section=str(settings.qa_generation.allow_cross_section).lower(),
        allow_cross_document=str(settings.qa_generation.allow_cross_document).lower(),
        require_strong_constraints=str(settings.qa_generation.require_strong_constraints).lower(),
        require_evidence=str(settings.qa_generation.require_evidence).lower(),
        doc_id=section.doc_id,
        section_id=section.section_id,
        doc_title=section.doc_title,
        section_path=" > ".join(section.section_path),
        content=section.content,
        knowledge_json=extraction.extraction.model_dump_json(indent=2),
    )


def _parse_candidates(raw_text: str) -> tuple[list[QACandidatePayload], str | None]:
    try:
        data = json.loads(raw_text)
        envelope = QAGenerationEnvelope.model_validate(data)
        return envelope.items, None
    except (json.JSONDecodeError, ValidationError) as exc:
        return [], str(exc)


def _basic_quality_filter(candidate: QACandidate, settings: Settings) -> bool:
    if not candidate.question or not candidate.answer_short or not candidate.answer_long:
        return False
    if settings.qa_generation.require_evidence and not candidate.evidence:
        return False
    if candidate.question_type == "multi_evidence" and len(candidate.evidence) < 2:
        return False
    if candidate.question_type == "relation" and not candidate.relations:
        return False
    if candidate.question_type == "explanation" and len(candidate.answer_long) < settings.qa_generation.explanation_min_length:
        return False
    if any(not is_quote_relevant(candidate.answer_long, item.quote) for item in candidate.evidence):
        return False
    return True


def _deduplicate(candidates: list[QACandidate], settings: Settings) -> list[QACandidate]:
    result: list[QACandidate] = []
    seen_exact: set[str] = set()
    for candidate in candidates:
        normalized = normalize_for_similarity(candidate.question)
        if normalized in seen_exact:
            continue
        if any(similarity(candidate.question, existing.question) >= settings.qa_generation.similarity_threshold for existing in result):
            continue
        seen_exact.add(normalized)
        result.append(candidate)
    return result


def _generate_one(
    client: LLMClient,
    section: MarkdownSection,
    extraction: KnowledgeExtractionResult,
    settings: Settings,
) -> tuple[list[QACandidate], bool]:
    if settings.dry_run or not client.enabled:
        return [], True

    response = client.chat_completion(system_prompt=SYSTEM_PROMPT, user_prompt=_render_prompt(section, extraction, settings))
    payloads, parse_error = _parse_candidates(response["content"])
    if parse_error:
        logger.warning("Failed to parse QA payload for section %s: %s", section.section_id, parse_error)
        return [], False

    candidates: list[QACandidate] = []
    for item in payloads[: settings.qa_generation.max_per_section]:
        try:
            candidate = QACandidate.model_validate(
                {
                    **item.model_dump(),
                    "qid": stable_id(section.doc_id, section.section_id, item.question, prefix="qa_"),
                    "doc_id": section.doc_id,
                    "section_id": section.section_id,
                }
            )
        except ValidationError as exc:
            logger.warning("Invalid QA candidate for section %s: %s", section.section_id, exc)
            continue
        if _basic_quality_filter(candidate, settings):
            candidates.append(candidate)
    return candidates, True


def run_generate_qa(settings: Settings) -> list[QACandidate]:
    sections, extractions = _load_inputs(settings)
    output_path = settings.artifacts_dir / "qa_candidates.jsonl"
    progress_path = settings.artifacts_dir / "qa_generation_progress.jsonl"
    existing: dict[str, QACandidate] = {}
    completed_sections: set[str] = set()
    if settings.resume and output_path.exists():
        for item in load_jsonl_as_models(output_path, QACandidate):
            existing[item.qid] = item
    if settings.resume and progress_path.exists():
        for item in load_jsonl_as_models(progress_path, QASectionProgress):
            if item.status == "completed":
                completed_sections.add(item.section_id)
    else:
        for item in existing.values():
            completed_sections.add(item.section_id)

    client = LLMClient(settings)
    pending = [
        item
        for item in extractions
        if item.section_id in sections and (not settings.resume or item.section_id not in completed_sections)
    ]

    generated: list[QACandidate] = list(existing.values())
    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as executor:
        futures = {
            executor.submit(_generate_one, client, sections[item.section_id], item, settings): item.section_id for item in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating QA"):
            try:
                section_id = futures[future]
                section_candidates, completed = future.result()
                generated.extend(section_candidates)
                if section_candidates:
                    append_jsonl(output_path, section_candidates)
                if completed:
                    append_jsonl(
                        progress_path,
                        [QASectionProgress(section_id=section_id, status="completed", qa_count=len(section_candidates))],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("QA generation failed: %s", exc)

    generated = _deduplicate(generated, settings)
    write_jsonl(output_path, generated)
    logger.info("Saved %s QA candidates", len(generated))
    return generated
