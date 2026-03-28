from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import orjson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_builder.logging_utils import setup_logging
from dataio.loaders import iter_jsonl
from embedders.zhipu_embedder import ZhipuEmbedder
from retrievers.qdrant_store import QdrantStore
from runtime_config import RuntimeSettings, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline Qdrant vector index from markdown sections JSONL.")
    parser.add_argument(
        "--sections",
        type=Path,
        default=Path("artifacts/two_file_demo/markdown_sections.jsonl"),
        help="Path to markdown_sections.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default_config.json"),
        help="Path to runtime config JSON file.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/qdrant_index_manifest.json"),
        help="Path to output manifest JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of sections to ingest.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the target collection before upserting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read sections and print stats without embedding or writing to Qdrant.",
    )
    return parser.parse_args()


def build_embedding_text(section: dict) -> str:
    doc_title = str(section.get("doc_title", "")).strip()
    section_path = [str(item).strip() for item in section.get("section_path", []) if str(item).strip()]
    content = str(section.get("content", "")).strip()

    parts = []
    if doc_title:
        parts.append(f"doc_title: {doc_title}")
    if section_path:
        parts.append(f"section_path: {' / '.join(section_path)}")
    if content:
        parts.append(f"content: {content}")
    return "\n".join(parts)


def load_sections(path: Path, *, limit: int | None = None) -> list[dict]:
    records = []
    for row in iter_jsonl(path):
        records.append(
            {
                "source_id": str(row.get("doc_id", "")).strip(),
                "section_id": str(row.get("section_id", "")).strip(),
                "doc_title": str(row.get("doc_title", "")).strip(),
                "section_path": [str(item) for item in row.get("section_path", [])],
                "block_type": str(row.get("block_type", "")).strip(),
                "char_len": int(row.get("char_len", 0) or 0),
                "content": str(row.get("content", "")),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def build_manifest(
    *,
    settings: RuntimeSettings,
    sections_path: Path,
    manifest_path: Path,
    indexed_count: int,
    dry_run: bool,
) -> dict:
    return {
        "sections_path": str(sections_path),
        "manifest_path": str(manifest_path),
        "collection_name": settings.retrieval.qdrant_collection,
        "embedding_model": settings.embedding.model,
        "vector_dimension": settings.embedding.dimension,
        "indexed_count": indexed_count,
        "backend": settings.retrieval.vector_db_backend,
        "qdrant_use_local": settings.retrieval.qdrant_use_local,
        "qdrant_target": str(settings.retrieval.qdrant_path)
        if settings.retrieval.qdrant_use_local
        else settings.retrieval.qdrant_url,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
    }


def print_summary(manifest: dict) -> None:
    print(f"sections_path: {manifest['sections_path']}")
    print(f"collection_name: {manifest['collection_name']}")
    print(f"embedding_model: {manifest['embedding_model']}")
    print(f"vector_dimension: {manifest['vector_dimension']}")
    print(f"indexed_count: {manifest['indexed_count']}")
    print(f"qdrant_target: {manifest['qdrant_target']}")
    print(f"dry_run: {manifest['dry_run']}")
    print(f"manifest: {manifest['manifest_path']}")


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings(args.config)
    setup_logging(PROJECT_ROOT / "logs", settings.log_level)

    if not args.sections.exists():
        raise FileNotFoundError(f"Sections file not found: {args.sections}")

    sections = load_sections(args.sections, limit=args.limit)
    embedding_texts = [build_embedding_text(section) for section in sections]
    non_empty_texts = sum(1 for text in embedding_texts if text.strip())

    if args.dry_run:
        manifest = build_manifest(
            settings=settings,
            sections_path=args.sections,
            manifest_path=args.manifest_path,
            indexed_count=len(sections),
            dry_run=True,
        )
        write_manifest(args.manifest_path, manifest)
        print_summary(manifest)
        return

    embedder = ZhipuEmbedder(
        api_key=settings.embedding.api_key,
        model=settings.embedding.model,
        dimensions=settings.embedding.dimension,
        batch_size=settings.embedding.batch_size,
    )
    store = QdrantStore(
        collection_name=settings.retrieval.qdrant_collection,
        vector_size=settings.embedding.dimension,
        url=settings.retrieval.qdrant_url,
        path=settings.retrieval.qdrant_path,
        use_local=settings.retrieval.qdrant_use_local,
        embedder=embedder,
    )

    try:
        if args.recreate:
            store.recreate_collection()
        else:
            store.ensure_collection()

        vectors = embedder.embed_documents(embedding_texts)
        store.upsert_sections(sections, vectors)
    finally:
        store.close()

    manifest = build_manifest(
        settings=settings,
        sections_path=args.sections,
        manifest_path=args.manifest_path,
        indexed_count=len(sections),
        dry_run=False,
    )
    write_manifest(args.manifest_path, manifest)

    print_summary(manifest)
    print(f"non_empty_embedding_texts: {non_empty_texts}")


if __name__ == "__main__":
    main()
