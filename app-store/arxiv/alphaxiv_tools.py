"""AlphaXiv tool wrappers.

Thin layer mapping Truffle-facing tool signatures (with _alphaxiv suffix)
to :class:`AlphaXivClient` convenience methods.

All functions return ``{"status": "success"|"error", ...}`` dicts that
match the pattern used by the v1 tools in ``arxiv_tools.py``.
"""

from __future__ import annotations

import re
from typing import Any

from alphaxiv_client import AlphaXivClient

# Pattern for extracting arXiv IDs from AlphaXiv's formatted text responses.
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def extract_arxiv_ids(text: str) -> list[str]:
    """Extract unique arXiv paper IDs from free-form text.

    Returns IDs without version suffix, deduplicated, in order of first
    appearance.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for m in _ARXIV_ID_RE.finditer(text):
        pid = m.group(1)
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


async def search_papers_alphaxiv(
    client: AlphaXivClient,
    query: str,
    mode: str = "agentic",
) -> dict[str, Any]:
    """Search papers via AlphaXiv.

    *mode* selects the underlying search tool:
    - ``"agentic"``  → ``agentic_paper_retrieval`` (default, multi-turn)
    - ``"semantic"`` → ``embedding_similarity_search``
    - ``"keyword"``  → ``full_text_papers_search``
    """
    if mode == "semantic":
        return await client.search_semantic(query)
    if mode == "keyword":
        return await client.search_keyword(query)
    return await client.search_agentic(query)


async def read_paper_alphaxiv(
    client: AlphaXivClient,
    paper_id: str,
    full_text: bool = False,
) -> dict[str, Any]:
    """Read paper content via AlphaXiv's ``get_paper_content``."""
    url = f"https://arxiv.org/abs/{paper_id}"
    return await client.get_paper_content(url, full_text=full_text)


async def answer_queries_alphaxiv(
    client: AlphaXivClient,
    paper_id: str,
    query: str,
) -> dict[str, Any]:
    """Answer a question about a paper's PDF via AlphaXiv AI."""
    url = f"https://arxiv.org/abs/{paper_id}"
    return await client.answer_pdf_query(url, query)


async def read_github_alphaxiv(
    client: AlphaXivClient,
    github_url: str,
    path: str = "/",
) -> dict[str, Any]:
    """Read files from a paper's GitHub repo via AlphaXiv."""
    return await client.read_github(github_url, path)
