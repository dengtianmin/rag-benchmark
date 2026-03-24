from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(logs_dir: Path, log_level: str = "INFO") -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "benchmark_builder.log"

    root = logging.getLogger()
    root.setLevel(log_level.upper())
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
