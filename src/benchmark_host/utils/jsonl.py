from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import orjson

T = TypeVar("T")


def read_jsonl(path: str | Path, mapper: Callable[[dict[str, Any]], T] | None = None) -> list[T] | list[dict[str, Any]]:
    items: list[Any] = []
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = orjson.loads(raw_line)
            items.append(mapper(record) if mapper else record)
    return items


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for row in rows:
            if is_dataclass(row):
                payload = asdict(row)
            else:
                payload = row
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
