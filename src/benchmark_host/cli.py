from __future__ import annotations

from pathlib import Path

import typer

from benchmark_host.config.models import ExperimentConfig
from benchmark_host.pipelines.experiment import ExperimentRunner

app = typer.Typer(help="Unified experiment host for enterprise benchmark baselines.")


@app.callback()
def main() -> None:
    """CLI root."""


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
) -> None:
    experiment_config = ExperimentConfig.load(config)
    outputs = ExperimentRunner(experiment_config).run()
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
