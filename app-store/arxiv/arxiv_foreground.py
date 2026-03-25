from __future__ import annotations

import logging
import os
from typing import Any

from app_runtime.mcp import create_mcp_server, run_mcp_server

from arxiv_tools import (
    build_deep_analysis_prompt,
    delete_paper,
    download_paper,
    list_papers,
    read_paper,
    search_papers,
)
from arxiv_common import (
    get_effective_interests,
    load_user_config,
    save_user_config,
    parse_research_interests,
    get_open_questions as _get_open_questions,
    add_open_question as _add_open_question,
    resolve_open_question as _resolve_open_question,
)

try:
    from mcp.types import Icon
except ImportError:
    class Icon:
        def __init__(self, **kwargs: Any) -> None:
            pass

# Guarded alphaxiv imports — missing deps must not break v1 tools.
try:
    from arxiv_config import is_alphaxiv_configured
    from alphaxiv_client import AlphaXivClient
    from alphaxiv_tools import (
        search_papers_alphaxiv,
        read_paper_alphaxiv,
        answer_queries_alphaxiv,
        read_github_alphaxiv,
    )
    _ALPHAXIV_AVAILABLE = True
except ImportError:
    _ALPHAXIV_AVAILABLE = False

logger = logging.getLogger("arxiv.foreground")
logger.setLevel(logging.INFO)

mcp = create_mcp_server("arxiv")


@mcp.tool(
    "search_papers",
    description=(
        "Search arXiv papers with optional date/category filters. "
        "Use quoted phrases for exact matches (for example: \"multi-agent systems\") "
        "and categories for precision (for example: cs.AI, cs.LG, cs.CL)."
    ),
    icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/magnifying-glass.svg")],
)
async def tool_search_papers(
    query: str,
    max_results: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    return await search_papers(
        query=query,
        max_results=max_results,
        date_from=date_from,
        date_to=date_to,
        categories=categories,
        sort_by=sort_by,
    )


@mcp.tool(
    "download_paper",
    description=(
        "Download an arXiv paper by ID and convert it to markdown for local reading. "
        "Use check_status=true to poll conversion progress."
    ),
    icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/download-simple.svg")],
)
async def tool_download_paper(
    paper_id: str,
    check_status: bool = False,
) -> dict[str, Any]:
    return await download_paper(paper_id=paper_id, check_status=check_status)


@mcp.tool(
    "list_papers",
    description="List papers currently downloaded to local storage.",
    icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/list.svg")],
)
async def tool_list_papers() -> dict[str, Any]:
    return await list_papers()


@mcp.tool(
    "delete_paper",
    description="Delete a downloaded paper from local storage by arXiv ID.",
    icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/trash.svg")],
)
async def tool_delete_paper(paper_id: str) -> dict[str, Any]:
    return await delete_paper(paper_id=paper_id)


@mcp.tool(
    "read_paper",
    description="Read full markdown content for a downloaded paper by arXiv ID.",
    icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/book-open-text.svg")],
)
async def tool_read_paper(paper_id: str) -> dict[str, Any]:
    return await read_paper(paper_id=paper_id)


@mcp.prompt(
    "deep-paper-analysis",
    description="Generate a structured deep analysis instruction set for a paper ID.",
)
async def prompt_deep_paper_analysis(paper_id: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_deep_analysis_prompt(paper_id)}]


# ---------------------------------------------------------------------------
# Research interests & open questions
# ---------------------------------------------------------------------------


@mcp.tool(
    "get_research_interests",
    description="View your current research interests that drive paper recommendations.",
)
async def tool_get_research_interests() -> dict[str, Any]:
    interests = get_effective_interests(os.getenv("ARXIV_RESEARCH_INTERESTS", ""))
    return {"status": "success", "interests": interests, "count": len(interests)}


@mcp.tool(
    "set_research_interests",
    description="Set your research interests (comma-separated). Replaces existing interests. The background worker will use these for paper discovery.",
)
async def tool_set_research_interests(interests: str) -> dict[str, Any]:
    parsed = parse_research_interests(interests)
    if not parsed:
        return {"status": "error", "message": "Provide at least one interest (comma-separated)."}
    config = load_user_config()
    config["interests"] = parsed
    save_user_config(config)
    return {"status": "success", "interests": parsed, "message": f"Set {len(parsed)} interest(s). Background worker will use these on next cycle."}


@mcp.tool(
    "add_open_question",
    description="Add a research question you want answered. The background worker will search for papers that address it.",
)
async def tool_add_open_question(question: str) -> dict[str, Any]:
    questions = _add_open_question(question)
    open_count = sum(1 for q in questions if q.get("status") == "open")
    return {"status": "success", "message": f"Question added. You have {open_count} open question(s).", "question": question}


@mcp.tool(
    "get_open_questions",
    description="View your open research questions and their status (open/answered).",
)
async def tool_get_open_questions() -> dict[str, Any]:
    questions = _get_open_questions()
    return {"status": "success", "questions": questions, "total": len(questions), "open": sum(1 for q in questions if q.get("status") == "open")}


@mcp.tool(
    "resolve_open_question",
    description="Mark an open question as answered. Provide the question index (0-based), optionally a paper ID that answered it, and optional notes.",
)
async def tool_resolve_open_question(
    question_index: int,
    paper_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_open_question(question_index, paper_id=paper_id, notes=notes)
    if resolved is None:
        return {"status": "error", "message": f"Invalid question index: {question_index}"}
    return {"status": "success", "resolved": resolved, "message": "Question marked as answered."}


# ---------------------------------------------------------------------------
# AlphaXiv tools (registered only when credentials are available)
# ---------------------------------------------------------------------------


def _alphaxiv_error_message(exc: Exception) -> str:
    """One-line error string for AlphaXiv tool failures."""
    return f"AlphaXiv error: {type(exc).__name__}: {exc}"


if _ALPHAXIV_AVAILABLE and is_alphaxiv_configured():

    @mcp.tool(
        "search_papers_alphaxiv",
        description=(
            "Search academic papers via AlphaXiv's enhanced search. "
            "Supports agentic (multi-turn), semantic (embedding), and keyword modes."
        ),
        icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/magnifying-glass.svg")],
    )
    async def tool_search_papers_alphaxiv(
        query: str,
        mode: str = "agentic",
    ) -> dict[str, Any]:
        try:
            async with AlphaXivClient() as client:
                return await search_papers_alphaxiv(client, query, mode)
        except Exception as exc:
            return {"status": "error", "message": _alphaxiv_error_message(exc)}

    @mcp.tool(
        "read_paper_alphaxiv",
        description=(
            "Read paper content via AlphaXiv. Retrieves metadata and optionally "
            "full text for a given arXiv paper ID."
        ),
        icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/book-open-text.svg")],
    )
    async def tool_read_paper_alphaxiv(
        paper_id: str,
        full_text: bool = False,
    ) -> dict[str, Any]:
        try:
            async with AlphaXivClient() as client:
                return await read_paper_alphaxiv(client, paper_id, full_text)
        except Exception as exc:
            return {"status": "error", "message": _alphaxiv_error_message(exc)}

    @mcp.tool(
        "answer_queries_alphaxiv",
        description=(
            "Ask a question about a paper's PDF content. AlphaXiv AI answers "
            "using the full paper text."
        ),
        icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/chat-text.svg")],
    )
    async def tool_answer_queries_alphaxiv(
        paper_id: str,
        query: str,
    ) -> dict[str, Any]:
        try:
            async with AlphaXivClient() as client:
                return await answer_queries_alphaxiv(client, paper_id, query)
        except Exception as exc:
            return {"status": "error", "message": _alphaxiv_error_message(exc)}

    @mcp.tool(
        "read_github_alphaxiv",
        description=(
            "Read files from a paper's associated GitHub repository via AlphaXiv."
        ),
        icons=[Icon(src="https://raw.githubusercontent.com/phosphor-icons/core/main/assets/regular/github-logo.svg")],
    )
    async def tool_read_github_alphaxiv(
        github_url: str,
        path: str = "/",
    ) -> dict[str, Any]:
        try:
            async with AlphaXivClient() as client:
                return await read_github_alphaxiv(client, github_url, path)
        except Exception as exc:
            return {"status": "error", "message": _alphaxiv_error_message(exc)}

    logger.info("AlphaXiv tools registered (credentials found)")

elif _ALPHAXIV_AVAILABLE:
    logger.info("AlphaXiv tools skipped — no credentials configured")
else:
    logger.info("AlphaXiv tools skipped — alphaxiv dependencies not available")


if __name__ == "__main__":
    run_mcp_server(mcp, logger)
