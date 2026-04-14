from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clients.chat_llm_client import ChatLLMClient
from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.retrieval_metrics import aggregate_retrieval_metric_dicts, evaluate_retrieval_ids
from evaluation.rewrite_diagnostics import aggregate_rewrite_diagnostics, evaluate_rewrite_diagnostics
from modules.graph_expander import GraphIndex
from modules.query_rewriters import QueryRewriteResult, RetrievalLabQueryRewriter
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractionResult, SkeletonExtractor
from pipelines.base import PublicIndex
from retrievers.text_retriever import TextRetrieverQuery, build_text_retriever
from runtime_config import build_trace_metadata, load_runtime_settings


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _combo_name(rewrite_mode: str, retrieval_mode: str, scorer_mode: str) -> str:
    return f"{rewrite_mode}__{retrieval_mode}__{scorer_mode}"


def _print_matrix(rows: list[dict], top_k: int) -> None:
    if not rows:
        print("No comparison rows generated.")
        return
    headers = [
        "combo",
        f"precision@{top_k}",
        f"recall@{top_k}",
        f"hit@{top_k}",
        "mrr",
        "semantic_drift_rate",
    ]
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "combo": row["combo"],
                f"precision@{top_k}": f'{row[f"precision@{top_k}"]:.4f}',
                f"recall@{top_k}": f'{row[f"recall@{top_k}"]:.4f}',
                f"hit@{top_k}": f'{row[f"hit@{top_k}"]:.4f}',
                "mrr": f'{row["mrr"]:.4f}',
                "semantic_drift_rate": f'{row["semantic_drift_rate"]:.4f}',
            }
        )
    widths = {
        header: max(len(header), max(len(str(item[header])) for item in display_rows))
        for header in headers
    }
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    divider = "-+-".join("-" * widths[header] for header in headers)
    print("\nResult Matrix")
    print(header_line)
    print(divider)
    for row in display_rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def _skeleton_dict(skeleton: SkeletonExtractionResult) -> dict:
    return {
        "mode": skeleton.mode,
        "entities": [item.model_dump() for item in skeleton.structured_entities],
        "relations": [item.model_dump() for item in skeleton.structured_relations],
        "constraints": [item.model_dump() for item in skeleton.structured_constraints],
        "rewrite_payload": skeleton.rewrite_payload.model_dump(),
        "details": skeleton.details,
    }


def _truncate_document(document, max_chars: int = 240) -> dict:
    return {
        "source_id": document.source_id,
        "section_id": document.section_id,
        "rank": document.rank,
        "score": float(document.score),
        "content_preview": document.content[:max_chars],
        "metadata": document.metadata,
    }


def _serialize_stage1_query(query: str | TextRetrieverQuery) -> str | dict[str, str]:
    if isinstance(query, TextRetrieverQuery):
        return {
            "lexical_query": query.lexical_query,
            "dense_query": query.dense_query,
        }
    return query


def _best_rank(ids: list[str], gold_ids: set[str]) -> int | None:
    ranks = [index for index, section_id in enumerate(ids, start=1) if section_id in gold_ids]
    return min(ranks) if ranks else None


def _extract_stage1_rows(
    *,
    scorer_mode: str,
    retrieval_query: str | TextRetrieverQuery,
    documents,
    retrieval_trace: dict,
) -> list[dict]:
    candidate_pool = retrieval_trace.get("candidate_pool", [])
    if candidate_pool and isinstance(candidate_pool[0], dict):
        return candidate_pool
    rows: list[dict] = []
    for document in documents:
        rows.append(
            {
                "section_id": document.section_id,
                "stage1_rank": document.rank,
                "stage1_score": float(document.score),
                "lexical_score": float(document.metadata.get("lexical_score", document.score)),
                "dense_score": float(document.metadata.get("dense_score", document.score)),
                "hybrid_score": float(document.metadata.get("hybrid_score", document.score)),
                "hybrid_sources": list(document.metadata.get("hybrid_sources", [])),
                "retriever": document.metadata.get("retriever", scorer_mode),
                "branch_trace": (
                    document.metadata.get("hybrid_branch_trace", {})
                    if isinstance(document.metadata.get("hybrid_branch_trace"), dict)
                    else {}
                ),
            }
        )
    if not rows and isinstance(retrieval_query, TextRetrieverQuery):
        rows.append(
            {
                "section_id": "",
                "stage1_rank": 0,
                "stage1_score": 0.0,
                "lexical_score": 0.0,
                "dense_score": 0.0,
                "hybrid_score": 0.0,
                "hybrid_sources": [],
                "retriever": scorer_mode,
                "branch_trace": {
                    "lexical_query": retrieval_query.lexical_query,
                    "dense_query": retrieval_query.dense_query,
                    "lexical_candidate_ids": [],
                    "dense_candidate_ids": [],
                    "lexical_candidate_scores": {},
                    "dense_candidate_scores": {},
                },
            }
        )
    return [row for row in rows if row.get("section_id")]


def _hybrid_branch_trace(stage1_rows: list[dict], retrieval_query: str | TextRetrieverQuery) -> dict:
    branch_trace: dict = {}
    for row in stage1_rows:
        candidate_trace = row.get("branch_trace")
        if isinstance(candidate_trace, dict) and candidate_trace:
            branch_trace = candidate_trace
            break
    if not branch_trace and isinstance(retrieval_query, TextRetrieverQuery):
        branch_trace = {
            "lexical_query": retrieval_query.lexical_query,
            "dense_query": retrieval_query.dense_query,
            "lexical_candidate_ids": [],
            "dense_candidate_ids": [],
            "lexical_candidate_scores": {},
            "dense_candidate_scores": {},
        }
    return branch_trace


def _gold_trace_payload(
    *,
    gold_section_ids: set[str],
    stage1_rows: list[dict],
    final_topk_ids: list[str],
    retrieval_trace: dict,
) -> dict:
    stage1_ids = [row["section_id"] for row in stage1_rows]
    gold_stage1_hits = [section_id for section_id in stage1_ids if section_id in gold_section_ids]
    gold_in_stage1_pool = bool(gold_stage1_hits)
    gold_best_rank_stage1 = _best_rank(stage1_ids, gold_section_ids)
    gold_in_final_topk = bool(gold_section_ids & set(final_topk_ids))
    gold_best_rank_final = _best_rank(final_topk_ids, gold_section_ids)

    kept_ids = list(retrieval_trace.get("kept_section_ids", final_topk_ids))
    filtered_ids = list(retrieval_trace.get("filtered_out_section_ids", []))
    filtered_reason_map = {
        section_id: reasons
        for section_id, reasons in (retrieval_trace.get("filtered_out_reason_map", {}) or {}).items()
    }
    gold_filtered_ids = [section_id for section_id in gold_section_ids if section_id in filtered_ids]
    gold_filtered_reason = filtered_reason_map.get(gold_filtered_ids[0], []) if gold_filtered_ids else []

    if gold_in_final_topk:
        miss_type = "in_final_topk"
        miss_reason_summary = "gold document reached final top-k"
    elif not gold_in_stage1_pool:
        miss_type = "not_in_stage1_pool"
        miss_reason_summary = "gold document missing from stage1 candidate pool"
    elif gold_filtered_ids:
        miss_type = "in_stage1_but_filtered"
        miss_reason_summary = "gold document entered stage1 but was filtered before final ranking"
    elif any(section_id in kept_ids for section_id in gold_section_ids):
        miss_type = "in_final_candidates_but_rank_too_low"
        miss_reason_summary = "gold document survived filtering but did not enter final top-k"
    else:
        miss_type = "unknown"
        miss_reason_summary = "gold miss could not be attributed to a known stage"

    all_gold_breakdowns: list[dict] = []
    breakdown_sources = [
        retrieval_trace.get("skeleton_aware_scores", {}) or {},
        retrieval_trace.get("filtered_out", {}) or {},
    ]
    for source in breakdown_sources:
        for section_id in gold_section_ids:
            if section_id in source:
                all_gold_breakdowns.append(
                    {
                        "section_id": section_id,
                        "kept": section_id not in filtered_ids,
                        "breakdown": source[section_id],
                    }
                )
    representative_gold = None
    if all_gold_breakdowns:
        representative_gold = max(
            all_gold_breakdowns,
            key=lambda item: float(item["breakdown"].get("final_score", 0.0)),
        )
    return {
        "stage1_candidate_ids": stage1_ids,
        "stage1_candidate_count": len(stage1_ids),
        "stage1_candidate_scores": {
            row["section_id"]: float(row.get("stage1_score", 0.0))
            for row in stage1_rows
        },
        "gold_in_stage1_pool": gold_in_stage1_pool,
        "gold_best_rank_stage1": gold_best_rank_stage1,
        "gold_stage1_hits": gold_stage1_hits,
        "final_topk_ids": final_topk_ids,
        "gold_in_final_topk": gold_in_final_topk,
        "gold_best_rank_final": gold_best_rank_final,
        "final_topk_count": len(final_topk_ids),
        "miss_type": miss_type,
        "miss_reason_summary": miss_reason_summary,
        "kept_section_ids": kept_ids,
        "filtered_out_section_ids": filtered_ids,
        "filtered_out_reason_map": filtered_reason_map,
        "recovered_section_ids": list(retrieval_trace.get("recovered_section_ids", [])),
        "gold_filtered_out": bool(gold_filtered_ids),
        "gold_filtered_reason": gold_filtered_reason,
        "representative_gold_section_id": representative_gold["section_id"] if representative_gold else None,
        "gold_semantic_score": (
            float(representative_gold["breakdown"].get("semantic_score", 0.0))
            if representative_gold
            else None
        ),
        "gold_entity_alignment_score": (
            float(representative_gold["breakdown"].get("entity_alignment_score", 0.0))
            if representative_gold
            else None
        ),
        "gold_relation_score": (
            float(representative_gold["breakdown"].get("relation_alignment_score", 0.0))
            if representative_gold
            else None
        ),
        "gold_constraint_score": (
            float(representative_gold["breakdown"].get("constraint_satisfaction_score", 0.0))
            if representative_gold
            else None
        ),
        "gold_fusion_score": (
            float(representative_gold["breakdown"].get("final_score", 0.0))
            if representative_gold
            else None
        ),
        "gold_filter_reasons": (
            list(representative_gold["breakdown"].get("filter_reasons", []))
            if representative_gold
            else []
        ),
        "gold_kept_or_filtered": (
            "kept" if representative_gold and representative_gold["kept"] else "filtered"
            if representative_gold
            else "missing"
        ),
    }


def _hybrid_source_payload(
    *,
    retrieval_mode: str,
    gold_section_ids: set[str],
    stage1_rows: list[dict],
    final_documents,
    retrieval_query: str | TextRetrieverQuery,
) -> dict:
    if retrieval_mode != "hybrid":
        return {
            "lexical_stage1_candidate_ids": [],
            "dense_stage1_candidate_ids": [],
            "gold_in_lexical_pool": False,
            "gold_in_dense_pool": False,
            "gold_only_in_lexical": False,
            "gold_only_in_dense": False,
            "gold_in_both_pools": False,
            "final_topk_source_breakdown": {"lexical": 0, "dense": 0, "both": 0, "unknown": 0},
        }
    branch_trace = _hybrid_branch_trace(stage1_rows, retrieval_query)
    lexical_ids = list(branch_trace.get("lexical_candidate_ids", []))
    dense_ids = list(branch_trace.get("dense_candidate_ids", []))
    gold_in_lexical = bool(gold_section_ids & set(lexical_ids))
    gold_in_dense = bool(gold_section_ids & set(dense_ids))
    source_breakdown = {"lexical": 0, "dense": 0, "both": 0, "unknown": 0}
    for document in final_documents:
        sources = list(document.metadata.get("hybrid_sources", []))
        if "lexical" in sources and "dense" in sources:
            source_breakdown["both"] += 1
        elif "lexical" in sources:
            source_breakdown["lexical"] += 1
        elif "dense" in sources:
            source_breakdown["dense"] += 1
        else:
            source_breakdown["unknown"] += 1
    return {
        "lexical_stage1_candidate_ids": lexical_ids,
        "dense_stage1_candidate_ids": dense_ids,
        "gold_in_lexical_pool": gold_in_lexical,
        "gold_in_dense_pool": gold_in_dense,
        "gold_only_in_lexical": gold_in_lexical and not gold_in_dense,
        "gold_only_in_dense": gold_in_dense and not gold_in_lexical,
        "gold_in_both_pools": gold_in_lexical and gold_in_dense,
        "final_topk_source_breakdown": source_breakdown,
    }


def _process_sample(
    *,
    sample,
    skeleton_mode: str,
    rewrite_mode: str,
    scorer_mode: str,
    top_k: int,
    trace_metadata: dict,
    skeleton_extractor: SkeletonExtractor,
    query_rewriter: RetrievalLabQueryRewriter,
    text_retriever,
    relation_driven: RelationDrivenRetriever,
) -> tuple[dict, dict[str, float], dict[str, float | bool]]:
    original_skeleton = skeleton_extractor.extract(sample, mode=skeleton_mode)
    rewrite_result: QueryRewriteResult = query_rewriter.rewrite(sample, original_skeleton, mode=rewrite_mode)  # type: ignore[arg-type]
    rewritten_sample = sample.model_copy(update={"question": rewrite_result.rewritten_query})
    rewritten_skeleton = skeleton_extractor.extract(rewritten_sample, mode="stub_predicted")
    rewrite_diagnostics = evaluate_rewrite_diagnostics(original_skeleton, rewritten_skeleton)
    retrieval_query = rewrite_result.query_for_retrieval(trace_metadata["retrieval_mode"])

    if scorer_mode == "plain":
        documents = text_retriever.retrieve(retrieval_query, top_k=top_k)
        retrieval_trace = {
            "mode": "plain",
            "stage1_query": _serialize_stage1_query(retrieval_query),
            "query_source": "rewrite_mode",
        }
    elif scorer_mode == "relation_driven":
        retrieve_result = relation_driven.retrieve(
            sample,
            original_skeleton,
            top_k=top_k,
            use_relation_driven=True,
            use_skeleton_rewrite=False,
            query_override=retrieval_query,
        )
        documents = retrieve_result.documents
        retrieval_trace = retrieve_result.details
    else:
        raise ValueError(f"Unsupported scorer mode: {scorer_mode}")

    retrieved_section_ids = [item.section_id for item in documents]
    gold_section_ids = {item.section_id for item in sample.evidence}
    stage1_rows = _extract_stage1_rows(
        scorer_mode=scorer_mode,
        retrieval_query=retrieval_query,
        documents=documents,
        retrieval_trace=retrieval_trace,
    )
    retrieval_metrics = evaluate_retrieval_ids(gold_section_ids, retrieved_section_ids, top_k)
    gold_trace = _gold_trace_payload(
        gold_section_ids=gold_section_ids,
        stage1_rows=stage1_rows,
        final_topk_ids=retrieved_section_ids,
        retrieval_trace=retrieval_trace,
    )
    hybrid_trace = _hybrid_source_payload(
        retrieval_mode=trace_metadata["retrieval_mode"],
        gold_section_ids=gold_section_ids,
        stage1_rows=stage1_rows,
        final_documents=documents,
        retrieval_query=retrieval_query,
    )
    stage1_query_payload = _serialize_stage1_query(retrieval_query)

    row = {
        "sample_id": sample.question_id,
        "question": sample.question,
        "original_question": sample.question,
        "rewrite_mode": rewrite_mode,
        "retrieval_mode": trace_metadata["retrieval_mode"],
        "scorer_mode": scorer_mode,
        "rewritten_query": rewrite_result.rewritten_query,
        "lexical_query": rewrite_result.lexical_query,
        "dense_query": rewrite_result.dense_query,
        "structured_rewrite": rewrite_result.structured_rewrite(),
        "original_skeleton": _skeleton_dict(original_skeleton),
        "rewritten_skeleton": _skeleton_dict(rewritten_skeleton),
        "rewrite_diagnostics": rewrite_diagnostics,
        "gold_evidence_section_ids": sorted(gold_section_ids),
        "stage1_query": stage1_query_payload,
        "stage1_query_source": retrieval_trace.get("query_source", "rewrite_mode"),
        "retrieved_section_ids": retrieved_section_ids,
        "retrieved_documents": [_truncate_document(item) for item in documents],
        "retrieval_metrics": retrieval_metrics,
        **gold_trace,
        **hybrid_trace,
        "trace_metadata": {
            **trace_metadata,
            "skeleton_mode": skeleton_mode,
            "rewrite_details": rewrite_result.details,
            "retrieval_trace": retrieval_trace,
        },
    }
    if scorer_mode == "relation_driven":
        row["dense_scores"] = {
            item.section_id: float(item.metadata.get("semantic_raw", item.metadata.get("dense_score", item.score)))
            for item in documents
        }
        row["relation_scores"] = {
            item.section_id: float(item.metadata.get("relation_alignment_score", 0.0))
            for item in documents
        }
        row["constraint_scores"] = {
            item.section_id: float(item.metadata.get("constraint_satisfaction_score", 0.0))
            for item in documents
        }
        row["fusion_score"] = {
            item.section_id: float(item.metadata.get("fusion_score", item.score))
            for item in documents
        }
        row["reasons"] = {
            item.section_id: item.metadata.get("filter_reasons", [])
            for item in documents
        }
    return row, retrieval_metrics, rewrite_diagnostics


def _build_llm_client(max_tokens: int, temperature: float) -> ChatLLMClient:
    settings = load_runtime_settings()
    return ChatLLMClient(
        api_key=settings.generator.api_key,
        base_url=settings.generator.base_url,
        model=settings.generator.model,
        timeout=settings.generator.timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )


def run_retrieval_lab(
    *,
    dataset: str | Path,
    sections: str | Path,
    knowledge: str | Path,
    top_k: int,
    limit: int | None,
    max_workers: int,
    skeleton_mode: str,
    rewrite_modes: list[str],
    retrieval_modes: list[str],
    scorer_modes: list[str],
    output_dir: str | Path,
    overwrite: bool = False,
    llm_rewrite_max_tokens: int = 64,
    llm_rewrite_temperature: float = 0.0,
    enable_debug_trace: bool = True,
    record_stage1_pool: bool = True,
    record_hybrid_branch_trace: bool = True,
) -> Path:
    del enable_debug_trace, record_stage1_pool, record_hybrid_branch_trace
    samples = load_benchmark_samples(dataset)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No samples found for retrieval lab.")

    output_path = Path(output_dir)
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    combos_dir = output_path / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

    text_index = PublicIndex.from_markdown_sections(sections)
    graph_index = GraphIndex.build(load_graph_section_records(knowledge))
    skeleton_llm_client = _build_llm_client(llm_rewrite_max_tokens, llm_rewrite_temperature)
    skeleton_extractor = SkeletonExtractor(
        graph_index,
        llm_client=skeleton_llm_client,
        llm_max_tokens=llm_rewrite_max_tokens,
        llm_temperature=llm_rewrite_temperature,
    )

    llm_rewrite_mode_names = {"llm", "sparse_llm", "dense_llm", "hybrid_llm"}
    llm_client = skeleton_llm_client if any(mode in llm_rewrite_mode_names for mode in rewrite_modes) else None
    query_rewriter = RetrievalLabQueryRewriter(
        llm_client=llm_client,
        llm_max_tokens=llm_rewrite_max_tokens,
        llm_temperature=llm_rewrite_temperature,
    )

    comparison_rows: list[dict] = []
    summary_payload: dict[str, dict] = {}
    combos = [
        (rewrite_mode, retrieval_mode, scorer_mode)
        for retrieval_mode in retrieval_modes
        for rewrite_mode in rewrite_modes
        for scorer_mode in scorer_modes
    ]

    for rewrite_mode, retrieval_mode, scorer_mode in tqdm(combos, desc="Retrieval Lab combos"):
        settings = load_runtime_settings()
        settings.retrieval.mode = retrieval_mode
        trace_metadata = build_trace_metadata(settings)
        text_retriever = build_text_retriever(index=text_index, settings=settings)
        relation_driven = RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever)
        combo = _combo_name(rewrite_mode, retrieval_mode, scorer_mode)
        print(f"\n[combo] {combo}")
        combo_dir = combos_dir / combo
        combo_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows: list[dict | None] = [None] * len(samples)
        retrieval_metric_rows: list[dict[str, float] | None] = [None] * len(samples)
        rewrite_metric_rows: list[dict[str, float | bool] | None] = [None] * len(samples)

        worker_count = max(1, max_workers)
        if worker_count == 1:
            for index, sample in enumerate(tqdm(samples, desc=f"samples:{combo}", leave=False)):
                row, retrieval_metrics, rewrite_diagnostics = _process_sample(
                    sample=sample,
                    skeleton_mode=skeleton_mode,
                    rewrite_mode=rewrite_mode,
                    scorer_mode=scorer_mode,
                    top_k=top_k,
                    trace_metadata=trace_metadata,
                    skeleton_extractor=skeleton_extractor,
                    query_rewriter=query_rewriter,
                    text_retriever=text_retriever,
                    relation_driven=relation_driven,
                )
                prediction_rows[index] = row
                retrieval_metric_rows[index] = retrieval_metrics
                rewrite_metric_rows[index] = rewrite_diagnostics
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _process_sample,
                        sample=sample,
                        skeleton_mode=skeleton_mode,
                        rewrite_mode=rewrite_mode,
                        scorer_mode=scorer_mode,
                        top_k=top_k,
                        trace_metadata=trace_metadata,
                        skeleton_extractor=skeleton_extractor,
                        query_rewriter=query_rewriter,
                        text_retriever=text_retriever,
                        relation_driven=relation_driven,
                    ): index
                    for index, sample in enumerate(samples)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"samples:{combo}", leave=False):
                    index = futures[future]
                    row, retrieval_metrics, rewrite_diagnostics = future.result()
                    prediction_rows[index] = row
                    retrieval_metric_rows[index] = retrieval_metrics
                    rewrite_metric_rows[index] = rewrite_diagnostics

        prediction_rows = [row for row in prediction_rows if row is not None]
        retrieval_metric_rows = [row for row in retrieval_metric_rows if row is not None]
        rewrite_metric_rows = [row for row in rewrite_metric_rows if row is not None]

        aggregate_retrieval = aggregate_retrieval_metric_dicts(retrieval_metric_rows, top_k)
        aggregate_diagnostics = aggregate_rewrite_diagnostics(rewrite_metric_rows)
        combo_metrics = {
            "combo": combo,
            "rewrite_mode": rewrite_mode,
            "retrieval_mode": retrieval_mode,
            "scorer_mode": scorer_mode,
            "sample_count": len(prediction_rows),
            "top_k": top_k,
            "retrieval": aggregate_retrieval,
            "rewrite_diagnostics": aggregate_diagnostics,
        }
        with (combo_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (combo_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(combo_metrics, handle, ensure_ascii=False, indent=2)

        summary_payload[combo] = combo_metrics
        comparison_row = {
            "combo": combo,
            "rewrite_mode": rewrite_mode,
            "retrieval_mode": retrieval_mode,
            "scorer_mode": scorer_mode,
            f"precision@{top_k}": aggregate_retrieval[f"precision@{top_k}"],
            f"recall@{top_k}": aggregate_retrieval[f"recall@{top_k}"],
            f"hit@{top_k}": aggregate_retrieval[f"hit@{top_k}"],
            "mrr": aggregate_retrieval["mrr"],
            "semantic_drift_rate": aggregate_diagnostics["semantic_drift_rate"],
            "entity_retention_recall": aggregate_diagnostics["entity_retention_recall"],
            "relation_retention_recall": aggregate_diagnostics["relation_retention_recall"],
            "constraint_retention_recall": aggregate_diagnostics["constraint_retention_recall"],
        }
        comparison_rows.append(comparison_row)
        print(
            "  "
            f'precision@{top_k}={comparison_row[f"precision@{top_k}"]:.4f} '
            f'recall@{top_k}={comparison_row[f"recall@{top_k}"]:.4f} '
            f'hit@{top_k}={comparison_row[f"hit@{top_k}"]:.4f} '
            f'mrr={comparison_row["mrr"]:.4f} '
            f'semantic_drift_rate={comparison_row["semantic_drift_rate"]:.4f}'
        )

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    with (output_path / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison_rows, handle, ensure_ascii=False, indent=2)
    matrix_columns = [
        "combo",
        "rewrite_mode",
        "retrieval_mode",
        "scorer_mode",
        f"precision@{top_k}",
        f"recall@{top_k}",
        f"hit@{top_k}",
        "mrr",
        "semantic_drift_rate",
        "entity_retention_recall",
        "relation_retention_recall",
        "constraint_retention_recall",
    ]
    with (output_path / "matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_columns)
        writer.writeheader()
        writer.writerows(comparison_rows)
    _print_matrix(comparison_rows, top_k)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only Ours lab with rewrite diagnosis.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted", "llm_predicted"], default="oracle")
    parser.add_argument("--rewrite-modes", default="original,splicing,sparse_llm,dense_llm,hybrid_llm")
    parser.add_argument("--retrieval-modes", default="lexical,dense,hybrid")
    parser.add_argument("--scorer-modes", default="plain,relation_driven")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/ours_rewrite_comparison"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-rewrite-max-tokens", type=int, default=64)
    parser.add_argument("--llm-rewrite-temperature", type=float, default=0.0)
    parser.add_argument("--enable-debug-trace", action="store_true", default=True)
    parser.add_argument("--record-stage1-pool", action="store_true", default=True)
    parser.add_argument("--record-hybrid-branch-trace", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_retrieval_lab(
        dataset=args.dataset,
        sections=args.sections,
        knowledge=args.knowledge,
        top_k=args.top_k,
        limit=args.limit,
        max_workers=args.max_workers,
        skeleton_mode=args.skeleton_mode,
        rewrite_modes=_parse_csv_arg(args.rewrite_modes),
        retrieval_modes=_parse_csv_arg(args.retrieval_modes),
        scorer_modes=_parse_csv_arg(args.scorer_modes),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        llm_rewrite_max_tokens=args.llm_rewrite_max_tokens,
        llm_rewrite_temperature=args.llm_rewrite_temperature,
        enable_debug_trace=args.enable_debug_trace,
        record_stage1_pool=args.record_stage1_pool,
        record_hybrid_branch_trace=args.record_hybrid_branch_trace,
    )
    print(f"outputs: {output_path}")


if __name__ == "__main__":
    main()
