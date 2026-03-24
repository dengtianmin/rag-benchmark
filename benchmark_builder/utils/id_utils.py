from __future__ import annotations

import hashlib


def stable_id(*parts: str, prefix: str = "") -> str:
    joined = "||".join(part.strip() for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest
