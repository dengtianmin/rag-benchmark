from __future__ import annotations

from benchmark_host.components.base import Component, ComponentOutput
from benchmark_host.schemas.common import BenchmarkSample


class HeuristicRewriteComponent(Component):
    """Rewrite query into a retrieval-oriented form using benchmark fields."""

    def run(self, sample: BenchmarkSample) -> ComponentOutput:
        skeleton_parts = [sample.question]
        if sample.entities:
            skeleton_parts.append("实体: " + ", ".join(sample.entities))
        if sample.relations:
            skeleton_parts.append("关系: " + ", ".join(sample.relations))
        if sample.constraints:
            skeleton_parts.append("约束: " + ", ".join(sample.constraints))
        rewritten = " | ".join(skeleton_parts)
        return ComponentOutput({"query": rewritten, "skeleton": {"entities": sample.entities, "relations": sample.relations}})


class SkeletonExtractionComponent(Component):
    """Chapter 4 skeleton extraction stub with explicit TODO for future LLM parser."""

    def run(self, sample: BenchmarkSample) -> ComponentOutput:
        skeleton = {
            "entities": sample.entities,
            "relations": sample.relations,
            "constraints": sample.constraints,
            "question_type": sample.question_type,
            "todo": "Replace heuristic skeleton extraction with model-based parser.",
        }
        retrieval_query = " ".join(sample.entities + sample.relations + sample.constraints) or sample.question
        return ComponentOutput({"skeleton": skeleton, "query": retrieval_query})
