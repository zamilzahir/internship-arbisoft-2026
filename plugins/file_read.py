"""
plugins/file_read.py
---------------------
A file-I/O plugin: gives the agent read access to .txt and .pdf files on
disk. Deliberately read-only and restricted to those two extensions -
an agent with unrestricted file I/O is a security footgun, so we fail
loudly on anything else rather than silently trying to "handle" it.
"""

from __future__ import annotations
from pathlib import Path
from hooks import logged_tool

MAX_CHARS = 20_000  # guard against blowing the context window on huge files


def _read_txt(path: Path) -> str:
    return path.read_text(errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so txt-only users don't need pypdf installed

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"--- page {i + 1} ---\n{text}")
    return "\n".join(pages)


@logged_tool
def read_file(path: str) -> str:
    """Read a .txt or .pdf file and return its text content (truncated
    if very long). This is the function the agent's tool-use loop calls."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".txt":
        content = _read_txt(p)
    elif suffix == ".pdf":
        content = _read_pdf(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r} (only .txt and .pdf are supported)")

    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + f"\n... [truncated, {len(content)} chars total]"
    return content
