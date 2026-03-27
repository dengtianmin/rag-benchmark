from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging

from pydantic import ValidationError
from tqdm import tqdm

from benchmark_builder.clients.deepseek_client import LLMClient
from benchmark_builder.config import Settings
from benchmark_builder.models import MarkdownSection, QACandidate, QAValidationResult
from benchmark_builder.prompts.qa_validation_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from benchmark_builder.utils.io_utils import append_jsonl, load_jsonl_as_models, write_jsonl

logger = logging.getLogger(__name__)


def _render_prompt(candidate: QACandidate, section: MarkdownSection) -> str:
    return USER_PROMPT_TEMPLATE.format(
        qa_json=candidate.model_dump_json(indent=2),
        content=section.content,
    )


def _validate_one(client: LLMClient, candidate: QACandidate, section: MarkdownSection, settings: Settings) -> QAValidationResult:
    if settings.dry_run or not client.enabled:
        return QAValidationResult(
            qid=candidate.qid,
            is_valid=True,
            support_score=1.0,
            completeness_score=1.0,
            type_consistent=True,
            is_nontrivial=True,
            has_hallucination=False,
            reject_reasons=[],
        )

    response = client.chat_completion(system_prompt=SYSTEM_PROMPT, user_prompt=_render_prompt(candidate, section))
    try:
        data = json.loads(response["content"])
        result = QAValidationResult.model_validate({**data, "raw_response_text": response["content"]})
    except (json.JSONDecodeError, ValidationError) as exc:
        return QAValidationResult(
            qid=candidate.qid,
            is_valid=False,
            reject_reasons=["validation_parse_failed"],
            raw_response_text=response["content"],
            parse_error=str(exc),
        )

    thresholds_ok = (
        result.support_score >= settings.validation.support_score_threshold
        and result.completeness_score >= settings.validation.completeness_score_threshold
        and result.type_consistent
        and result.is_nontrivial
        and not result.has_hallucination
    )
    result.is_valid = bool(result.is_valid and thresholds_ok)
    if not result.is_valid and not result.reject_reasons:
        result.reject_reasons = ["below_threshold"]
    return result


def run_validate_qa(settings: Settings) -> list[QAValidationResult]:
    candidates = load_jsonl_as_models(settings.artifacts_dir / "qa_candidates.jsonl", QACandidate)
    sections = {
        item.section_id: item
        for item in load_jsonl_as_models(settings.artifacts_dir / "markdown_sections.jsonl", MarkdownSection)
    }
    output_path = settings.artifacts_dir / "qa_validation.jsonl"
    existing: dict[str, QAValidationResult] = {}
    if settings.resume and output_path.exists():
        for item in load_jsonl_as_models(output_path, QAValidationResult):
            existing[item.qid] = item

    client = LLMClient(settings)
    results: list[QAValidationResult] = list(existing.values())
    with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as executor:
        futures = {
            executor.submit(_validate_one, client, candidate, sections[candidate.section_id], settings): candidate.qid
            for candidate in candidates
            if candidate.section_id in sections and candidate.qid not in existing
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating QA"):
            try:
                result = future.result()
                results.append(result)
                append_jsonl(output_path, [result])
            except Exception as exc:  # noqa: BLE001
                logger.exception("QA validation failed: %s", exc)

    results = sorted(results, key=lambda item: item.qid)
    write_jsonl(output_path, results)
    logger.info("Saved %s validation results", len(results))
    return results
