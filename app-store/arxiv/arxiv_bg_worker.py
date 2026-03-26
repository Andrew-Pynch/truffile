from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta, timezone
import logging
from typing import Any

import arxiv

from arxiv_common import get_bg_state_path, get_effective_interests, get_open_questions
from arxiv_config import is_alphaxiv_configured, is_alphaxiv_enrich_enabled

try:
    from alphaxiv_client import AlphaXivClient
    from alphaxiv_tools import (
        answer_queries_alphaxiv,
        extract_arxiv_ids,
        search_papers_alphaxiv,
    )

    _ALPHAXIV_AVAILABLE = True
except ImportError:
    _ALPHAXIV_AVAILABLE = False

logger = logging.getLogger("arxiv.bg_worker")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Persistent event loop for async AlphaXiv calls (matches kalshi pattern)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):  # noqa: ANN001
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

MAX_ALPHAXIV_RESULTS = 3
MAX_TOTAL_RESULTS = 5


@dataclass
class ArxivRecommendation:
    interest: str
    paper_id: str
    title: str
    published: str
    abs_url: str
    summary: str
    source: str = "arxiv"
    enriched_summary: str = ""


@dataclass
class BgRunResult:
    content: str | None = None
    error: str | None = None


class ArxivBackgroundWorker:
    def __init__(self, interests_raw: str) -> None:
        self._interests_raw = interests_raw
        self._client = arxiv.Client()
        self._state_path = get_bg_state_path()

    @property
    def interests(self) -> list[str]:
        return get_effective_interests(self._interests_raw)

    @property
    def open_questions(self) -> list[str]:
        """Return open (unanswered) research questions as search queries."""
        return [
            q["question"] for q in get_open_questions()
            if q.get("status") == "open"
        ]

    def verify(self) -> tuple[bool, str]:
        interests = self.interests
        if not interests:
            return False, "No research interests configured. Provide at least one interest."
        return True, f"ArXiv background configured with {len(interests)} interest(s)."

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> BgRunResult:
        interests = self.interests
        questions = self.open_questions
        # Combine interests + open questions as search queries
        all_queries = interests + questions
        if not all_queries:
            return BgRunResult(error="no_interests")

        state = self._load_state()
        today_iso = date.today().isoformat()
        cutoff_iso = (date.today() - timedelta(days=7)).isoformat()
        raw_seen = state.get("seen_ids") or {}
        if isinstance(raw_seen, list):  # backwards compat: migrate flat list → dict
            seen_ids: dict[str, str] = {pid: today_iso for pid in raw_seen}
        else:
            seen_ids = dict(raw_seen)
        seen_ids = {pid: d for pid, d in seen_ids.items() if d >= cutoff_iso}
        recommendations: list[ArxivRecommendation] = []

        # --- Phase 1: AlphaXiv agentic search (primary) ---
        if _ALPHAXIV_AVAILABLE and is_alphaxiv_configured():
            alphaxiv_recs, alphaxiv_error = self._search_alphaxiv_interests(
                all_queries, seen_ids, max_results=MAX_ALPHAXIV_RESULTS,
            )
            if not alphaxiv_error and alphaxiv_recs:
                for rec in alphaxiv_recs:
                    seen_ids[rec.paper_id] = today_iso
                recommendations.extend(alphaxiv_recs)

        # --- Phase 2: v1 arXiv search (fallback, fills remaining slots) ---
        v1_max = MAX_TOTAL_RESULTS - len(recommendations)
        if v1_max > 0:
            for interest in all_queries:
                for paper in self._search_interest(interest, max_results=8):
                    paper_id = paper.get_short_id()
                    if not paper_id or paper_id in seen_ids:
                        continue
                    published_iso = paper.published.astimezone(timezone.utc).date().isoformat()
                    recommendations.append(
                        ArxivRecommendation(
                            interest=interest,
                            paper_id=paper_id,
                            title=paper.title.strip(),
                            published=published_iso,
                            abs_url=f"https://arxiv.org/abs/{paper_id}",
                            summary=" ".join((paper.summary or "").split())[:450],
                            source="arxiv",
                        )
                    )
                    seen_ids[paper_id] = today_iso
                    if len(recommendations) >= MAX_TOTAL_RESULTS:
                        break
                if len(recommendations) >= MAX_TOTAL_RESULTS:
                    break

        if not recommendations:
            return BgRunResult(content=None)

        state["seen_ids"] = seen_ids
        self._save_state(state)
        return BgRunResult(
            content=self._build_context(recommendations),
        )

    # ------------------------------------------------------------------
    # v1 arXiv search
    # ------------------------------------------------------------------

    def _search_interest(self, interest: str, *, max_results: int) -> list[arxiv.Result]:
        try:
            search = arxiv.Search(
                query=interest,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            return list(self._client.results(search))
        except Exception as exc:
            logger.warning("arXiv search failed for interest '%s': %s", interest, exc)
            return []

    # ------------------------------------------------------------------
    # AlphaXiv search (async, bridged via _run_async)
    # ------------------------------------------------------------------

    def _search_alphaxiv_interests(
        self,
        interests: list[str],
        seen_ids: dict[str, str],
        *,
        max_results: int,
    ) -> tuple[list[ArxivRecommendation], str | None]:
        """Sync bridge into the async alphaxiv search. Returns (recs, error_str)."""
        try:
            recs = _run_async(
                self._search_alphaxiv_interests_async(
                    interests, seen_ids, max_results=max_results,
                )
            )
            return recs, None
        except Exception as exc:
            logger.warning("AlphaXiv background search failed: %s", exc)
            return [], f"{type(exc).__name__}: {exc}"

    async def _search_alphaxiv_interests_async(
        self,
        interests: list[str],
        seen_ids: dict[str, str],
        *,
        max_results: int,
    ) -> list[ArxivRecommendation]:
        recommendations: list[ArxivRecommendation] = []
        enrich = is_alphaxiv_enrich_enabled()

        async with AlphaXivClient() as client:
            for interest in interests:
                if len(recommendations) >= max_results:
                    break

                result = await search_papers_alphaxiv(client, interest, mode="agentic")
                if result.get("status") != "success":
                    logger.warning(
                        "AlphaXiv search failed for '%s': %s",
                        interest,
                        result.get("message", "unknown"),
                    )
                    continue

                paper_ids = extract_arxiv_ids(result.get("content", ""))
                new_ids = [pid for pid in paper_ids if pid not in seen_ids]
                if not new_ids:
                    continue

                # Look up structured metadata via arxiv API
                metadata_map = await asyncio.to_thread(
                    self._fetch_arxiv_metadata, new_ids,
                )

                for pid in new_ids:
                    if len(recommendations) >= max_results:
                        break
                    meta = metadata_map.get(pid)
                    if meta is None:
                        continue

                    enriched = ""
                    if enrich:
                        enriched = await self._enrich_paper(client, pid, interest)

                    recommendations.append(
                        ArxivRecommendation(
                            interest=interest,
                            paper_id=pid,
                            title=meta["title"],
                            published=meta["published"],
                            abs_url=f"https://arxiv.org/abs/{pid}",
                            summary=meta["summary"],
                            source="alphaxiv",
                            enriched_summary=enriched,
                        )
                    )

        return recommendations

    def _fetch_arxiv_metadata(self, paper_ids: list[str]) -> dict[str, dict[str, str]]:
        """Batch-lookup title/published/summary for paper IDs via arxiv API."""
        result: dict[str, dict[str, str]] = {}
        try:
            search = arxiv.Search(id_list=paper_ids)
            for paper in self._client.results(search):
                pid = paper.get_short_id()
                if pid:
                    published_iso = paper.published.astimezone(timezone.utc).date().isoformat()
                    result[pid] = {
                        "title": paper.title.strip(),
                        "published": published_iso,
                        "summary": " ".join((paper.summary or "").split())[:450],
                    }
        except Exception as exc:
            logger.warning("arXiv metadata lookup failed: %s", exc)
        return result

    async def _enrich_paper(
        self, client: AlphaXivClient, paper_id: str, interest: str,
    ) -> str:
        """Ask alphaxiv to summarize a paper relative to the interest."""
        try:
            result = await answer_queries_alphaxiv(
                client,
                paper_id,
                f"Summarize this paper's key contributions related to: {interest}",
            )
            if result.get("status") == "success":
                return " ".join(result.get("content", "").split())[:600]
        except Exception as exc:
            logger.warning("AlphaXiv PDF enrichment failed for %s: %s", paper_id, exc)
        return ""

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        path = self._state_path
        try:
            if not path.exists():
                return {"seen_ids": {}}
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to load BG state: %s", exc)
        return {"seen_ids": {}}

    def _save_state(self, state: dict[str, Any]) -> None:
        path = self._state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save BG state: %s", exc)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, items: list[ArxivRecommendation]) -> str:
        lines: list[str] = [
            "New research papers matching your interests:",
        ]
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"{idx}. [{item.source}] {item.title} (arXiv:{item.paper_id}, published {item.published})"
            )
            lines.append(f"   Interest match: {item.interest}")
            lines.append(f"   URL: {item.abs_url}")
            if item.summary:
                lines.append(f"   Abstract snippet: {item.summary}")
            if item.enriched_summary:
                lines.append(f"   AI summary: {item.enriched_summary}")
        return "\n".join(lines)
