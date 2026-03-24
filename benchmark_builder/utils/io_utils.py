from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypeVar
import json
import logging

import orjson
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def append_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for record in records:
            data = record.model_dump() if isinstance(record, BaseModel) else record
            handle.write(orjson.dumps(data))
            handle.write(b"\n")


def write_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            data = record.model_dump() if isinstance(record, BaseModel) else record
            handle.write(orjson.dumps(data))
            handle.write(b"\n")


def load_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return iter(())

    def _iterator() -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skip invalid JSONL line %s in %s", line_number, path)

    return _iterator()


def load_jsonl_as_models(path: Path, model_cls: type[T]) -> list[T]:
    return [model_cls.model_validate(item) for item in load_jsonl(path)]
