from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class LLMOptions(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 2048
    json_mode: bool = True


class QATypeQuota(BaseModel):
    fact: int = 1
    relation: int = 1
    multi_evidence: int = 1
    explanation: int = 1


class QAGenerationConfig(BaseModel):
    max_per_section: int = 4
    type_quota: QATypeQuota = Field(default_factory=QATypeQuota)
    allow_cross_section: bool = False
    allow_cross_document: bool = False
    require_strong_constraints: bool = True
    require_evidence: bool = True
    explanation_min_length: int = 40
    similarity_threshold: float = 0.9


class ValidationConfig(BaseModel):
    support_score_threshold: float = 0.8
    completeness_score_threshold: float = 0.75


class Settings(BaseModel):
    input_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")
    log_level: str = "INFO"
    max_files: int | None = None
    concurrency: int = 4
    request_timeout: int = 120
    retry_attempts: int = 3
    retry_backoff_min: int = 1
    retry_backoff_max: int = 20
    resume: bool = True
    dry_run: bool = False
    llm: LLMOptions = Field(default_factory=LLMOptions)
    qa_generation: QAGenerationConfig = Field(default_factory=QAGenerationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    llm_provider: str = "openai_compatible"
    llm_api_style: str = "openai"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_disable_auth: bool = False
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen2.5-7b-instruct-1m"

    @model_validator(mode="after")
    def expand_paths(self) -> "Settings":
        self.input_dir = self.input_dir.expanduser()
        self.artifacts_dir = self.artifacts_dir.expanduser()
        self.outputs_dir = self.outputs_dir.expanduser()
        self.logs_dir = self.logs_dir.expanduser()
        return self

    def ensure_directories(self) -> None:
        for path in (self.artifacts_dir, self.outputs_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


def _read_json_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_settings(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    load_dotenv()
    path = Path(config_path) if config_path else Path("config/default_config.json")
    raw = _read_json_config(path)
    if overrides:
        raw.update({key: value for key, value in overrides.items() if value is not None})

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", raw.get("deepseek_api_key"))
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", raw.get("deepseek_base_url", "https://api.deepseek.com"))
    deepseek_model = os.getenv("DEEPSEEK_MODEL", raw.get("deepseek_model", "deepseek-chat"))
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", raw.get("dashscope_api_key"))
    dashscope_base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        raw.get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    dashscope_model = os.getenv("DASHSCOPE_MODEL", raw.get("dashscope_model", "qwen2.5-7b-instruct-1m"))
    llm_provider = os.getenv("LLM_PROVIDER", raw.get("llm_provider", "openai_compatible")).strip().lower()

    raw["deepseek_api_key"] = deepseek_api_key
    raw["deepseek_base_url"] = deepseek_base_url
    raw["deepseek_model"] = deepseek_model
    raw["dashscope_api_key"] = dashscope_api_key
    raw["dashscope_base_url"] = dashscope_base_url
    raw["dashscope_model"] = dashscope_model

    if llm_provider == "dashscope":
        default_api_key = dashscope_api_key
        default_base_url = dashscope_base_url
        default_model = dashscope_model
    else:
        default_api_key = deepseek_api_key
        default_base_url = deepseek_base_url
        default_model = deepseek_model

    raw["llm_provider"] = llm_provider
    raw["llm_api_style"] = os.getenv("LLM_API_STYLE", raw.get("llm_api_style", "openai"))
    raw["llm_api_key"] = os.getenv("LLM_API_KEY", raw.get("llm_api_key", default_api_key))
    raw["llm_base_url"] = os.getenv("LLM_BASE_URL", raw.get("llm_base_url", default_base_url))
    raw["llm_model"] = os.getenv("LLM_MODEL", raw.get("llm_model", default_model))
    raw["llm_disable_auth"] = os.getenv(
        "LLM_DISABLE_AUTH",
        str(raw.get("llm_disable_auth", False)),
    ).lower() in {"1", "true", "yes", "on"}
    raw["log_level"] = os.getenv("BENCHMARK_LOG_LEVEL", raw.get("log_level", "INFO"))

    settings = Settings.model_validate(raw)
    settings.ensure_directories()
    return settings
