from __future__ import annotations

from pathlib import Path

import pytest

from core.schema import AnswerResult, BenchmarkSample, EvidenceItem
from dataio.loaders import load_benchmark_samples
from dataio.normalizers import normalize_benchmark_sample


def test_normalize_benchmark_sample_maps_qid_and_doc_ids() -> None:
    sample = normalize_benchmark_sample(
        {
            "qid": "qa_1",
            "question": "产品地址是什么？",
            "answer_short": "北京",
            "question_type": "fact",
            "entities": ["北京总部"],
            "relations": [],
            "constraints": [],
            "evidence": [{"doc_id": "doc_1", "section_id": "sec_1", "quote": "北京总部"}],
            "source_scope": "single_section",
            "requires_text_compensation": False,
            "doc_id": "doc_1",
            "section_id": "sec_1",
        }
    )
    assert sample.question_id == "qa_1"
    assert sample.source_id == "doc_1"
    assert sample.evidence[0].source_id == "doc_1"
    assert sample.evidence[0].section_id == "sec_1"


def test_benchmark_sample_requires_one_answer_field() -> None:
    with pytest.raises(ValueError):
        BenchmarkSample(
            question_id="qa_2",
            question="empty answer",
            answer_short="",
            answer_long="",
            question_type="fact",
            source_scope="single_section",
        )


def test_evidence_item_requires_source_and_section() -> None:
    with pytest.raises(ValueError):
        EvidenceItem(source_id="", section_id="sec_1")
    with pytest.raises(ValueError):
        EvidenceItem(source_id="doc_1", section_id="")


def test_answer_result_requires_content() -> None:
    with pytest.raises(ValueError):
        AnswerResult(question_id="qa_1", answer_text="")


def test_load_benchmark_samples_from_demo_dataset() -> None:
    dataset_path = Path("outputs/two_file_demo/benchmark_dataset.jsonl")
    samples = load_benchmark_samples(dataset_path)
    assert samples
    assert samples[0].question_id
    assert samples[0].question
    assert samples[0].answer_short or samples[0].answer_long
