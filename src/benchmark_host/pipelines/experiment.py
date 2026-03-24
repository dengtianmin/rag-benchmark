from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from benchmark_host.baselines.graph_rag import GraphEnhancedRAGBaseline
from benchmark_host.baselines.kbqa import KBQABaseline
from benchmark_host.baselines.ours_ch4 import OursCh4Baseline
from benchmark_host.baselines.rewrite_rag import RewriteRAGBaseline
from benchmark_host.baselines.traditional import TraditionalRAGBaseline
from benchmark_host.components.generator import StubAnswerGenerator, TextEvidenceCompensator
from benchmark_host.components.indexes import CorpusIndex
from benchmark_host.components.retrievers import GraphRetriever, RelationDrivenRetriever, TraditionalRetriever
from benchmark_host.components.rewriters import HeuristicRewriteComponent, SkeletonExtractionComponent
from benchmark_host.config.models import ExperimentConfig
from benchmark_host.services.dataset_loader import DatasetLoader
from benchmark_host.services.evaluator import Evaluator
from benchmark_host.utils.jsonl import write_jsonl


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.loader = DatasetLoader()
        self.evaluator = Evaluator()

    def build_systems(self, index: CorpusIndex) -> dict[str, object]:
        traditional = TraditionalRetriever(index, self.config.retriever.top_k)
        relation = RelationDrivenRetriever(index, self.config.retriever.top_k, self.config.retriever.expand_k)
        graph = GraphRetriever(
            index,
            self.config.retriever.top_k,
            self.config.retriever.expand_k,
            self.config.retriever.neighbor_hops,
        )
        generator = StubAnswerGenerator(self.config.generator.max_evidence_chars)
        systems = {
            "traditional_rag": TraditionalRAGBaseline(traditional, generator),
            "rewrite_rag": RewriteRAGBaseline(HeuristicRewriteComponent(), traditional, generator),
            "graph_enhanced_rag": GraphEnhancedRAGBaseline(traditional, graph, generator),
            "kbqa_baseline": KBQABaseline(SkeletonExtractionComponent(), relation, generator),
            "ours_ch4": OursCh4Baseline(
                SkeletonExtractionComponent(),
                relation,
                traditional,
                TextEvidenceCompensator(),
                generator,
            ),
        }
        return {name: systems[name] for name in self.config.systems}

    def run(self) -> dict[str, str]:
        bundle = self.loader.load(
            benchmark_dataset_path=str(self.config.paths.benchmark_dataset),
            markdown_sections_path=str(self.config.paths.markdown_sections),
            knowledge_extraction_path=str(self.config.paths.knowledge_extraction),
            limit=self.config.limit,
        )
        index = CorpusIndex(bundle.sections, bundle.extractions)
        systems = self.build_systems(index)
        self.config.paths.output_dir.mkdir(parents=True, exist_ok=True)

        result_paths: dict[str, str] = {}
        leaderboard_rows = []
        for system_name, system in systems.items():
            predictions = []
            evaluations = []
            for sample in bundle.samples:
                prediction = system.predict(sample)
                predictions.append(prediction.to_dict())
                evaluations.append(asdict(self.evaluator.evaluate(sample, prediction)))
            pred_path = self.config.paths.output_dir / f"{system_name}.predictions.jsonl"
            eval_path = self.config.paths.output_dir / f"{system_name}.metrics.csv"
            write_jsonl(pred_path, predictions)
            frame = pd.DataFrame(evaluations)
            summary = frame[["exact_match", "token_f1", "evidence_hit", "retrieval_hit"]].mean().to_frame("value")
            summary.to_csv(eval_path)
            leaderboard_rows.append(
                {
                    "system_name": system_name,
                    "exact_match": summary.loc["exact_match", "value"],
                    "token_f1": summary.loc["token_f1", "value"],
                    "evidence_hit": summary.loc["evidence_hit", "value"],
                    "retrieval_hit": summary.loc["retrieval_hit", "value"],
                }
            )
            result_paths[f"{system_name}_predictions"] = str(pred_path)
            result_paths[f"{system_name}_metrics"] = str(eval_path)
        leaderboard_path = self.config.paths.output_dir / "leaderboard.csv"
        pd.DataFrame(leaderboard_rows).sort_values(["token_f1", "retrieval_hit"], ascending=False).to_csv(
            leaderboard_path, index=False
        )
        result_paths["leaderboard"] = str(leaderboard_path)
        return result_paths
