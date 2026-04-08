from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


def _read_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


class EmbeddingConfig(BaseModel):
    provider: str = "zhipuai"
    api_key: str | None = None
    model: str = "embedding-3"
    dimension: int = 1024
    batch_size: int = 32


class RetrievalConfig(BaseModel):
    backend: str = "qdrant"
    mode: str = "lexical"
    top_k: int = 5
    candidate_multiplier: int = 3
    vector_db_backend: str = "qdrant"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "benchmark_sections"
    qdrant_path: Path = Path("artifacts/qdrant")
    qdrant_use_local: bool = False

    @model_validator(mode="after")
    def expand_paths(self) -> "RetrievalConfig":
        self.qdrant_path = self.qdrant_path.expanduser()
        return self


class RerankConfig(BaseModel):
    enabled: bool = True
    use_rerank: bool = True
    backend: str = "tei"
    model_path: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    use_fp16: bool = False
    query_max_length: int = 256
    passage_max_length: int = 512
    tei_url: str = "http://127.0.0.1:8080"
    timeout: float = 30.0
    api_key: str | None = None
    max_retries: int = 0
    top_n: int | None = None
    allow_mock_fallback: bool = False

    @model_validator(mode="after")
    def normalize_teir_url(self) -> "RerankConfig":
        self.tei_url = self.tei_url.rstrip("/")
        return self


class GeneratorConfig(BaseModel):
    backend: str = "mock"
    provider: str = "openai_compatible"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    timeout: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0
    json_mode: bool = True

    @model_validator(mode="after")
    def normalize_base_url(self) -> "GeneratorConfig":
        self.base_url = self.base_url.rstrip("/")
        return self


class RuntimeSettings(BaseModel):
    log_level: str = "INFO"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)


def build_trace_metadata(settings: RuntimeSettings) -> dict[str, Any]:
    return {
        "embedding_model": settings.embedding.model,
        "embedding_dim": settings.embedding.dimension,
        "qdrant_collection": settings.retrieval.qdrant_collection,
        "retrieval_mode": settings.retrieval.mode,
        "use_rerank": settings.rerank.enabled,
        "rerank_backend": settings.rerank.backend,
        "rerank_url": settings.rerank.tei_url if settings.rerank.backend == "tei" else None,
        "rerank_top_n": settings.rerank.top_n,
        "generator_backend": settings.generator.backend,
    }


def load_runtime_settings(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RuntimeSettings:
    load_dotenv()
    path = Path(config_path) if config_path else Path("config/default_config.json")
    raw = _read_json_config(path)
    if overrides:
        raw.update({key: value for key, value in overrides.items() if value is not None})

    embedding_raw = dict(raw.get("embedding", {}))
    embedding_raw["provider"] = os.getenv("EMBEDDING_PROVIDER", embedding_raw.get("provider", "zhipuai"))
    embedding_raw["api_key"] = os.getenv("ZHIPUAI_API_KEY", embedding_raw.get("api_key"))
    embedding_raw["model"] = os.getenv("EMBEDDING_MODEL", embedding_raw.get("model", "embedding-3"))
    embedding_raw["dimension"] = _env_int("EMBEDDING_DIM", int(embedding_raw.get("dimension", 1024)))
    embedding_raw["batch_size"] = _env_int("EMBEDDING_BATCH_SIZE", int(embedding_raw.get("batch_size", 32)))

    retrieval_raw = dict(raw.get("retrieval", {}))
    retrieval_raw["backend"] = os.getenv("RETRIEVAL_BACKEND", retrieval_raw.get("backend", "qdrant"))
    retrieval_raw["mode"] = os.getenv("RETRIEVAL_MODE", retrieval_raw.get("mode", "lexical"))
    retrieval_raw["top_k"] = _env_int("RETRIEVAL_TOP_K", int(retrieval_raw.get("top_k", 5)))
    retrieval_raw["candidate_multiplier"] = _env_int(
        "RETRIEVAL_CANDIDATE_MULTIPLIER",
        int(retrieval_raw.get("candidate_multiplier", 3)),
    )
    retrieval_raw["vector_db_backend"] = os.getenv(
        "VECTOR_DB_BACKEND",
        retrieval_raw.get("vector_db_backend", retrieval_raw.get("backend", "qdrant")),
    )
    retrieval_raw["qdrant_url"] = os.getenv("QDRANT_URL", retrieval_raw.get("qdrant_url", "http://127.0.0.1:6333"))
    retrieval_raw["qdrant_collection"] = os.getenv(
        "QDRANT_COLLECTION",
        retrieval_raw.get("qdrant_collection", "benchmark_sections"),
    )
    retrieval_raw["qdrant_path"] = os.getenv("QDRANT_PATH", retrieval_raw.get("qdrant_path", "artifacts/qdrant"))
    retrieval_raw["qdrant_use_local"] = _env_bool(
        "QDRANT_USE_LOCAL",
        bool(retrieval_raw.get("qdrant_use_local", False)),
    )

    rerank_raw = dict(raw.get("rerank", {}))
    rerank_flag = _env_bool(
        "USE_RERANK",
        bool(rerank_raw.get("use_rerank", rerank_raw.get("enabled", True))),
    )
    rerank_raw["enabled"] = _env_bool("RERANK_ENABLED", bool(rerank_raw.get("enabled", rerank_flag))) and rerank_flag
    rerank_raw["use_rerank"] = rerank_flag
    rerank_raw["backend"] = os.getenv("RERANK_BACKEND", rerank_raw.get("backend", "tei"))
    rerank_raw["model_path"] = os.getenv(
        "RERANKER_MODEL_PATH",
        rerank_raw.get("model_path", "BAAI/bge-reranker-v2-m3"),
    )
    rerank_raw["device"] = os.getenv("RERANKER_DEVICE", rerank_raw.get("device", "cpu"))
    rerank_raw["use_fp16"] = _env_bool("RERANKER_USE_FP16", bool(rerank_raw.get("use_fp16", False)))
    rerank_raw["query_max_length"] = _env_int(
        "RERANKER_QUERY_MAX_LENGTH",
        int(rerank_raw.get("query_max_length", 256)),
    )
    rerank_raw["passage_max_length"] = _env_int(
        "RERANKER_PASSAGE_MAX_LENGTH",
        int(rerank_raw.get("passage_max_length", 512)),
    )
    rerank_raw["tei_url"] = os.getenv("TEI_RERANK_URL", rerank_raw.get("tei_url", "http://127.0.0.1:8080"))
    rerank_raw["timeout"] = _env_float("TEI_RERANK_TIMEOUT", float(rerank_raw.get("timeout", 30)))
    rerank_raw["api_key"] = os.getenv("TEI_RERANK_API_KEY", rerank_raw.get("api_key"))
    rerank_raw["max_retries"] = _env_int("TEI_RERANK_MAX_RETRIES", int(rerank_raw.get("max_retries", 0)))
    rerank_raw["top_n"] = _env_optional_int("RERANK_TOP_N", rerank_raw.get("top_n"))
    rerank_raw["allow_mock_fallback"] = _env_bool(
        "RERANK_ALLOW_MOCK_FALLBACK",
        bool(rerank_raw.get("allow_mock_fallback", False)),
    )

    generator_raw = dict(raw.get("generator", {}))
    generator_provider = os.getenv("GENERATOR_PROVIDER", generator_raw.get("provider", "openai_compatible")).strip().lower()
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", generator_raw.get("dashscope_api_key"))
    dashscope_base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        generator_raw.get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    dashscope_model = os.getenv("DASHSCOPE_MODEL", generator_raw.get("dashscope_model", "qwen2.5-7b-instruct-1m"))
    if generator_provider == "dashscope":
        default_generator_api_key = dashscope_api_key
        default_generator_base_url = dashscope_base_url
        default_generator_model = dashscope_model
    else:
        default_generator_api_key = generator_raw.get("api_key")
        default_generator_base_url = generator_raw.get("base_url", "https://api.openai.com/v1")
        default_generator_model = generator_raw.get("model", "")

    generator_raw["backend"] = os.getenv("GENERATOR_BACKEND", generator_raw.get("backend", "mock"))
    generator_raw["provider"] = generator_provider
    generator_raw["api_key"] = os.getenv("GENERATOR_API_KEY", default_generator_api_key)
    generator_raw["base_url"] = os.getenv(
        "GENERATOR_BASE_URL",
        default_generator_base_url,
    )
    generator_raw["model"] = os.getenv("GENERATOR_MODEL", default_generator_model)
    generator_raw["timeout"] = _env_float("GENERATOR_TIMEOUT", float(generator_raw.get("timeout", 60)))
    generator_raw["max_tokens"] = _env_int("GENERATOR_MAX_TOKENS", int(generator_raw.get("max_tokens", 512)))
    generator_raw["temperature"] = _env_float(
        "GENERATOR_TEMPERATURE",
        float(generator_raw.get("temperature", 0.0)),
    )
    generator_raw["json_mode"] = _env_bool(
        "GENERATOR_JSON_MODE",
        bool(generator_raw.get("json_mode", True)),
    )

    payload = {
        "log_level": os.getenv("BENCHMARK_LOG_LEVEL", raw.get("log_level", "INFO")),
        "embedding": embedding_raw,
        "retrieval": retrieval_raw,
        "rerank": rerank_raw,
        "generator": generator_raw,
    }
    return RuntimeSettings.model_validate(payload)
