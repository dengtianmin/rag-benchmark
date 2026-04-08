from __future__ import annotations

from html import escape
from pathlib import Path


def write_html_report(markdown_path: Path, html_path: Path) -> None:
    markdown = markdown_path.read_text(encoding="utf-8")
    html = (
        "<html><head><meta charset='utf-8'><title>Rewrite Lab Report</title>"
        "<style>body{font-family:sans-serif;max-width:1000px;margin:40px auto;line-height:1.6;}pre{white-space:pre-wrap;background:#f5f5f5;padding:16px;border-radius:8px;}</style>"
        "</head><body><pre>"
        + escape(markdown)
        + "</pre></body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
