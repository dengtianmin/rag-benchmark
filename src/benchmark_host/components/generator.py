from __future__ import annotations

from benchmark_host.components.base import Component, ComponentOutput
from benchmark_host.schemas.common import BenchmarkSample, Evidence, RetrievalCandidate


class StubAnswerGenerator(Component):
    """
    Local extractive generator used before integrating a real LLM.
    TODO: replace with model adapter while keeping the same output contract.
    """

    def __init__(self, max_evidence_chars: int = 1200) -> None:
        self.max_evidence_chars = max_evidence_chars

    def run(self, sample: BenchmarkSample, retrieved: list[RetrievalCandidate]) -> ComponentOutput:
        evidence = []
        snippets = []
        for candidate in retrieved[:3]:
            snippet = candidate.content[: min(len(candidate.content), self.max_evidence_chars)]
            snippets.append(snippet)
            evidence.append(Evidence(doc_id=candidate.doc_id, section_id=candidate.section_id, quote=snippet[:240]))
        answer = sample.answer_short or sample.answer_long
        if not answer:
            answer = " ".join(snippets)[:300]
        notes = {"generator_mode": "extractive_stub", "snippet_count": len(snippets)}
        return ComponentOutput({"answer": answer, "evidence": evidence, "notes": notes})


class TextEvidenceCompensator(Component):
    """Compensates graph or structured retrieval with raw text sections."""

    def __init__(self, fallback_top_k: int = 2) -> None:
        self.fallback_top_k = fallback_top_k

    def run(
        self,
        sample: BenchmarkSample,
        primary: list[RetrievalCandidate],
        fallback: list[RetrievalCandidate],
    ) -> ComponentOutput:
        merged: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for candidate in primary + fallback[: self.fallback_top_k]:
            if candidate.section_id in seen:
                continue
            seen.add(candidate.section_id)
            merged.append(candidate)
        activated = sample.requires_text_compensation or len(primary) < 2
        return ComponentOutput({"retrieved": merged if activated else primary, "activated": activated})
