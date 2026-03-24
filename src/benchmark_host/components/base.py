from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentOutput:
    data: dict[str, Any] = field(default_factory=dict)


class Component:
    """Haystack-like lightweight component contract without external dependency."""

    def run(self, **kwargs: Any) -> ComponentOutput:
        raise NotImplementedError
