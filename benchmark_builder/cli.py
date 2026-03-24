from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from benchmark_builder.config import load_settings
from benchmark_builder.logging_utils import setup_logging
from benchmark_builder.pipelines.build_dataset import run_build_dataset
from benchmark_builder.pipelines.extract_knowledge import run_extract_knowledge
from benchmark_builder.pipelines.generate_qa import run_generate_qa
from benchmark_builder.pipelines.parse_markdown import run_parse_markdown
from benchmark_builder.pipelines.validate_qa import run_validate_qa

app = typer.Typer(help="Benchmark dataset builder for product-document QA.")


def _settings(
    config_path: Optional[Path],
    input_dir: Optional[Path],
    artifacts_dir: Optional[Path],
    output_dir: Optional[Path],
    max_files: Optional[int],
    concurrency: Optional[int],
    resume: bool,
    dry_run: bool,
):
    settings = load_settings(
        config_path=config_path,
        overrides={
            "input_dir": input_dir,
            "artifacts_dir": artifacts_dir,
            "outputs_dir": output_dir,
            "max_files": max_files,
            "concurrency": concurrency,
            "resume": resume,
            "dry_run": dry_run,
        },
    )
    setup_logging(settings.logs_dir, settings.log_level)
    return settings


@app.command("parse-md")
def parse_md(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_parse_markdown(settings)


@app.command("extract-knowledge")
def extract_knowledge(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_extract_knowledge(settings)


@app.command("generate-qa")
def generate_qa(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_generate_qa(settings)


@app.command("validate-qa")
def validate_qa(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_validate_qa(settings)


@app.command("build-dataset")
def build_dataset(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_build_dataset(settings)


@app.command("run-all")
def run_all(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts-dir"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    max_files: Optional[int] = typer.Option(None, "--max-files"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    settings = _settings(config_path, input_dir, artifacts_dir, output_dir, max_files, concurrency, resume, dry_run)
    run_parse_markdown(settings)
    run_extract_knowledge(settings)
    run_generate_qa(settings)
    run_validate_qa(settings)
    run_build_dataset(settings)
