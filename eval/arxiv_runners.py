"""Test runners for each eval capability: search, extraction, bg worker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path setup — import arxiv app modules from app-store/arxiv/
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
_ARXIV_APP_DIR = _REPO_ROOT / "app-store" / "arxiv"

if str(_ARXIV_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_ARXIV_APP_DIR))

from arxiv_test_set import BG_WORKER_INTERESTS, PaperSpec, SearchQuery
from arxiv_metrics import (
    char_count_score,
    dedup_correctness,
    extract_paper_ids_from_content,
    keyword_presence,
    normalize_arxiv_id,
    precision_at_k,
    recall,
    recommendation_relevance,
    section_header_presence,
)

logger = logging.getLogger("arxiv_eval")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Enforce minimum delay between arxiv API calls."""

    def __init__(self, min_delay: float = 3.5):
        self._min_delay = min_delay
        self._last_call = 0.0

    async def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            await asyncio.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()

    def wait_sync(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def with_retries(max_retries: int = 3, base_delay: float = 5.0):
    """Decorator for async functions that retries on transient errors."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1, max_retries + 1, func.__name__, e, delay,
                        )
                        await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Download helper — polls until conversion completes
# ---------------------------------------------------------------------------

async def download_and_wait(
    paper_id: str, timeout: float = 180.0
) -> dict[str, Any]:
    """Download a paper and poll until conversion finishes or times out."""
    from arxiv_tools import download_paper

    result = await download_paper(paper_id=paper_id)
    if result.get("status") == "success":
        return result
    if result.get("status") == "error":
        return result

    deadline = time.monotonic() + timeout
    delay = 1.0
    while time.monotonic() < deadline:
        await asyncio.sleep(delay)
        status = await download_paper(paper_id=paper_id, check_status=True)
        if status.get("status") in ("success", "error"):
            return status
        delay = min(delay * 1.5, 10.0)

    return {"status": "error", "message": f"Timeout after {timeout}s waiting for {paper_id}"}


# ---------------------------------------------------------------------------
# Search eval
# ---------------------------------------------------------------------------

async def run_search_eval(
    test_papers: list[PaperSpec],
    search_queries: list[SearchQuery],
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Evaluate search quality: known-item retrieval + topic queries."""
    from arxiv_tools import search_papers

    known_item_results: list[dict[str, Any]] = []
    topic_query_results: list[dict[str, Any]] = []

    # --- Known-item retrieval: search by title ---
    for paper in test_papers:
        await rate_limiter.wait()
        query = f'ti:"{paper.title}"'
        logger.info("Known-item search: %s", paper.arxiv_id)

        t0 = time.monotonic()
        try:
            response = await search_papers(query=query, max_results=5)
        except Exception as exc:
            known_item_results.append({
                "paper_id": paper.arxiv_id,
                "query": query,
                "status": "error",
                "error": str(exc),
                "latency_s": time.monotonic() - t0,
            })
            continue
        t1 = time.monotonic()

        if response.get("status") != "success":
            known_item_results.append({
                "paper_id": paper.arxiv_id,
                "query": query,
                "status": "error",
                "error": response.get("message", "unknown"),
                "latency_s": t1 - t0,
            })
            continue

        retrieved_ids = [p["id"] for p in response.get("papers", [])]
        found = any(
            normalize_arxiv_id(rid) == normalize_arxiv_id(paper.arxiv_id)
            for rid in retrieved_ids
        )
        known_item_results.append({
            "paper_id": paper.arxiv_id,
            "query": query,
            "status": "success",
            "found": found,
            "precision_at_5": precision_at_k(retrieved_ids, {paper.arxiv_id}, k=5),
            "num_results": len(retrieved_ids),
            "latency_s": t1 - t0,
        })

    # --- Topic queries ---
    for sq in search_queries:
        await rate_limiter.wait()
        logger.info("Topic query: %s", sq.description)

        t0 = time.monotonic()
        try:
            response = await search_papers(
                query=sq.query,
                max_results=10,
                categories=sq.categories,
            )
        except Exception as exc:
            topic_query_results.append({
                "query": sq.query,
                "description": sq.description,
                "status": "error",
                "error": str(exc),
                "latency_s": time.monotonic() - t0,
            })
            continue
        t1 = time.monotonic()

        if response.get("status") != "success":
            topic_query_results.append({
                "query": sq.query,
                "description": sq.description,
                "status": "error",
                "error": response.get("message", "unknown"),
                "latency_s": t1 - t0,
            })
            continue

        retrieved_ids = [p["id"] for p in response.get("papers", [])]
        relevant_set = set(sq.expected_paper_ids)
        topic_query_results.append({
            "query": sq.query,
            "description": sq.description,
            "status": "success",
            "precision_at_5": precision_at_k(retrieved_ids, relevant_set, k=5),
            "recall": recall(retrieved_ids, relevant_set),
            "num_results": len(retrieved_ids),
            "latency_s": t1 - t0,
        })

    # --- Aggregates ---
    successful_known = [r for r in known_item_results if r.get("status") == "success"]
    successful_topic = [r for r in topic_query_results if r.get("status") == "success"]

    aggregate = {
        "known_item_hit_rate": (
            sum(1 for r in successful_known if r.get("found")) / len(successful_known)
            if successful_known else 0.0
        ),
        "known_item_mean_latency_s": (
            sum(r["latency_s"] for r in successful_known) / len(successful_known)
            if successful_known else 0.0
        ),
        "topic_mean_precision_at_5": (
            sum(r["precision_at_5"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "topic_mean_recall": (
            sum(r["recall"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "topic_mean_latency_s": (
            sum(r["latency_s"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "total_queries": len(known_item_results) + len(topic_query_results),
        "total_errors": (
            len(known_item_results) - len(successful_known)
            + len(topic_query_results) - len(successful_topic)
        ),
    }

    return {
        "known_item": known_item_results,
        "topic_queries": topic_query_results,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Extraction eval
# ---------------------------------------------------------------------------

async def run_extraction_eval(
    test_papers: list[PaperSpec],
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Evaluate content extraction: download + convert + read + measure."""
    from arxiv_tools import read_paper

    per_paper: list[dict[str, Any]] = []

    for paper in test_papers:
        await rate_limiter.wait()
        logger.info("Extraction: downloading %s", paper.arxiv_id)

        t0 = time.monotonic()
        dl_result = await download_and_wait(paper.arxiv_id, timeout=180.0)
        t1 = time.monotonic()

        if dl_result.get("status") != "success":
            per_paper.append({
                "paper_id": paper.arxiv_id,
                "paper_type": paper.paper_type,
                "download_status": dl_result.get("status", "error"),
                "error": dl_result.get("message"),
                "download_latency_s": t1 - t0,
            })
            continue

        read_result = await read_paper(paper_id=paper.arxiv_id)
        t2 = time.monotonic()

        if read_result.get("status") != "success":
            per_paper.append({
                "paper_id": paper.arxiv_id,
                "paper_type": paper.paper_type,
                "download_status": "success",
                "read_status": "error",
                "error": read_result.get("message"),
                "download_latency_s": t1 - t0,
            })
            continue

        content = read_result.get("content", "")
        per_paper.append({
            "paper_id": paper.arxiv_id,
            "paper_type": paper.paper_type,
            "download_status": "success",
            "read_status": "success",
            "download_latency_s": t1 - t0,
            "read_latency_s": t2 - t1,
            "char_count": len(content),
            "char_count_score": char_count_score(len(content), paper.min_char_count),
            "keyword_score": keyword_presence(content, paper.ground_truth_keywords),
            "section_score": section_header_presence(content, paper.expected_sections),
        })

    # --- Aggregates ---
    successful = [r for r in per_paper if r.get("read_status") == "success"]
    aggregate = {
        "success_rate": len(successful) / len(per_paper) if per_paper else 0.0,
        "mean_keyword_score": (
            sum(r["keyword_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_section_score": (
            sum(r["section_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_char_count": (
            sum(r["char_count"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_char_count_score": (
            sum(r["char_count_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_download_latency_s": (
            sum(r["download_latency_s"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "total_papers": len(per_paper),
        "total_errors": len(per_paper) - len(successful),
    }

    return {"per_paper": per_paper, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Background worker eval
# ---------------------------------------------------------------------------

def run_bg_worker_eval(
    interests: list[str],
    rate_limiter: RateLimiter,
    llm_judge: bool = False,
) -> dict[str, Any]:
    """Evaluate the background worker: verify, run_cycle, dedup, relevance."""
    from arxiv_bg_worker import ArxivBackgroundWorker

    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="arxiv_eval_storage_") as storage_dir, \
         tempfile.TemporaryDirectory(prefix="arxiv_eval_state_") as state_dir:

        os.environ["ARXIV_STORAGE_PATH"] = storage_dir
        os.environ["ARXIV_BG_STATE_PATH"] = os.path.join(state_dir, "bg_state.json")

        # --- Test verify() with valid interests ---
        worker = ArxivBackgroundWorker(interests_raw=", ".join(interests))
        ok, msg = worker.verify()
        results["verify_ok"] = ok
        results["verify_message"] = msg

        # --- Test verify() with empty interests ---
        empty_worker = ArxivBackgroundWorker(interests_raw="")
        ok_empty, msg_empty = empty_worker.verify()
        results["empty_verify_fails"] = not ok_empty

        # --- Test run_cycle() with empty interests ---
        r_empty = empty_worker.run_cycle()
        results["empty_returns_no_interests_error"] = r_empty.error == "no_interests"

        # --- Test run_cycle() first run ---
        rate_limiter.wait_sync()
        logger.info("BG worker: running first cycle")
        r1 = worker.run_cycle()
        run1_ids = extract_paper_ids_from_content(r1.content) if r1.content else set()
        results["run1_has_content"] = r1.content is not None and len(r1.content) > 0
        results["run1_count"] = len(run1_ids)
        results["run1_error"] = r1.error

        # --- Test dedup: second run ---
        rate_limiter.wait_sync()
        logger.info("BG worker: running second cycle (dedup test)")
        r2 = worker.run_cycle()
        run2_ids = extract_paper_ids_from_content(r2.content) if r2.content else set()
        results["run2_count"] = len(run2_ids)
        results["dedup_correct"] = dedup_correctness(run1_ids, run2_ids)

        # --- Relevance scoring ---
        if r1.content:
            results["relevance_score"] = recommendation_relevance(
                r1.content, interests
            )
        else:
            results["relevance_score"] = 0.0

        # --- LLM judge (optional) ---
        if llm_judge and r1.content:
            results["llm_judge"] = _run_llm_judge(r1.content, interests)

    return results


# ---------------------------------------------------------------------------
# LLM judge helper
# ---------------------------------------------------------------------------

def _run_llm_judge(content: str, interests: list[str]) -> dict[str, Any]:
    """Score each recommendation with an LLM call (claude-haiku)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM judge")
        return {"skipped": True, "reason": "ANTHROPIC_API_KEY not set"}

    import httpx

    # Parse recommendations from the content
    recommendations = _parse_recommendations(content)
    if not recommendations:
        return {"skipped": True, "reason": "No recommendations to judge"}

    scores: list[dict[str, Any]] = []
    for rec in recommendations:
        prompt = (
            f"Rate the relevance of this paper to the research interest on a scale of 1-5, "
            f"where 1 means completely irrelevant and 5 means highly relevant.\n\n"
            f"Paper: {rec['title']}\n"
            f"Abstract: {rec.get('summary', 'N/A')}\n"
            f"Interest: {rec.get('interest', ', '.join(interests))}\n\n"
            f'Respond with JSON only: {{"score": N, "reason": "..."}}'
        )

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            text = data["content"][0]["text"]
            # Parse JSON from response
            judge_result = json.loads(text)
            scores.append({
                "paper_title": rec["title"],
                "score": judge_result.get("score", 0),
                "reason": judge_result.get("reason", ""),
            })
        except Exception as exc:
            logger.warning("LLM judge call failed for '%s': %s", rec["title"], exc)
            scores.append({
                "paper_title": rec["title"],
                "score": None,
                "error": str(exc),
            })

    valid_scores = [s["score"] for s in scores if isinstance(s.get("score"), (int, float))]
    return {
        "skipped": False,
        "scores": scores,
        "mean_score": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "num_scored": len(valid_scores),
        "num_failed": len(scores) - len(valid_scores),
    }


def _parse_recommendations(content: str) -> list[dict[str, str]]:
    """Extract individual recommendations from bg worker output text."""
    recommendations = []
    lines = content.split("\n")
    current: dict[str, str] = {}

    for line in lines:
        # Match numbered recommendations: "1. Title (arXiv:ID, published DATE)"
        title_match = re.match(r"^\d+\.\s+(.+?)\s+\(arXiv:", line)
        if title_match:
            if current:
                recommendations.append(current)
            current = {"title": title_match.group(1)}
            continue

        if current:
            interest_match = re.match(r"\s+Interest match:\s+(.+)", line)
            if interest_match:
                current["interest"] = interest_match.group(1)
                continue
            summary_match = re.match(r"\s+Abstract snippet:\s+(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1)
                continue

    if current:
        recommendations.append(current)

    return recommendations


# ---------------------------------------------------------------------------
# AlphaXiv runners
# ---------------------------------------------------------------------------


async def _alphaxiv_call(tool_fn, *args, **kwargs) -> dict[str, Any]:
    """Run a single AlphaXiv tool call in its own client session.

    Each call gets a fresh connection so that a 502 on one call does not
    poison the cancel scope for subsequent calls.
    """
    from alphaxiv_client import AlphaXivClient

    try:
        async with AlphaXivClient() as client:
            return await tool_fn(client, *args, **kwargs)
    except BaseException as exc:
        return {"status": "error", "message": f"AlphaXiv session error: {exc}"}


async def run_alphaxiv_search_eval(
    test_papers: list[PaperSpec],
    search_queries: list[SearchQuery],
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Evaluate AlphaXiv search: known-item retrieval + topic queries."""
    from alphaxiv_tools import extract_arxiv_ids, search_papers_alphaxiv

    known_item_results: list[dict[str, Any]] = []
    topic_query_results: list[dict[str, Any]] = []

    # --- Known-item retrieval: semantic search by title ---
    for paper in test_papers:
        await rate_limiter.wait()
        logger.info("AlphaXiv known-item search: %s", paper.arxiv_id)

        t0 = time.monotonic()
        response = await _alphaxiv_call(
            search_papers_alphaxiv, paper.title, mode="semantic"
        )
        t1 = time.monotonic()

        if response.get("status") != "success":
            known_item_results.append({
                "paper_id": paper.arxiv_id,
                "query": paper.title,
                "status": "error",
                "error": response.get("message", "unknown"),
                "latency_s": t1 - t0,
            })
            continue

        content = response.get("content", "")
        retrieved_ids = extract_arxiv_ids(content)
        found = any(
            normalize_arxiv_id(rid) == normalize_arxiv_id(paper.arxiv_id)
            for rid in retrieved_ids
        )
        known_item_results.append({
            "paper_id": paper.arxiv_id,
            "query": paper.title,
            "status": "success",
            "found": found,
            "precision_at_5": precision_at_k(retrieved_ids, {paper.arxiv_id}, k=5),
            "num_results": len(retrieved_ids),
            "latency_s": t1 - t0,
        })

    # --- Topic queries ---
    for sq in search_queries:
        await rate_limiter.wait()
        logger.info("AlphaXiv topic query: %s", sq.description)

        t0 = time.monotonic()
        response = await _alphaxiv_call(
            search_papers_alphaxiv, sq.query, mode="agentic"
        )
        t1 = time.monotonic()

        if response.get("status") != "success":
            topic_query_results.append({
                "query": sq.query,
                "description": sq.description,
                "status": "error",
                "error": response.get("message", "unknown"),
                "latency_s": t1 - t0,
            })
            continue

        content = response.get("content", "")
        retrieved_ids = extract_arxiv_ids(content)
        relevant_set = set(sq.expected_paper_ids)
        topic_query_results.append({
            "query": sq.query,
            "description": sq.description,
            "status": "success",
            "precision_at_5": precision_at_k(retrieved_ids, relevant_set, k=5),
            "recall": recall(retrieved_ids, relevant_set),
            "num_results": len(retrieved_ids),
            "latency_s": t1 - t0,
        })

    # --- Aggregates ---
    successful_known = [r for r in known_item_results if r.get("status") == "success"]
    successful_topic = [r for r in topic_query_results if r.get("status") == "success"]

    aggregate = {
        "known_item_hit_rate": (
            sum(1 for r in successful_known if r.get("found")) / len(successful_known)
            if successful_known else 0.0
        ),
        "known_item_mean_latency_s": (
            sum(r["latency_s"] for r in successful_known) / len(successful_known)
            if successful_known else 0.0
        ),
        "topic_mean_precision_at_5": (
            sum(r["precision_at_5"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "topic_mean_recall": (
            sum(r["recall"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "topic_mean_latency_s": (
            sum(r["latency_s"] for r in successful_topic) / len(successful_topic)
            if successful_topic else 0.0
        ),
        "total_queries": len(known_item_results) + len(topic_query_results),
        "total_errors": (
            len(known_item_results) - len(successful_known)
            + len(topic_query_results) - len(successful_topic)
        ),
    }

    return {
        "known_item": known_item_results,
        "topic_queries": topic_query_results,
        "aggregate": aggregate,
    }


async def run_alphaxiv_extraction_eval(
    test_papers: list[PaperSpec],
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Evaluate AlphaXiv content extraction via get_paper_content."""
    from alphaxiv_tools import read_paper_alphaxiv

    per_paper: list[dict[str, Any]] = []

    for paper in test_papers:
        await rate_limiter.wait()
        logger.info("AlphaXiv extraction: %s", paper.arxiv_id)

        t0 = time.monotonic()
        response = await _alphaxiv_call(read_paper_alphaxiv, paper.arxiv_id)
        t1 = time.monotonic()

        if response.get("status") != "success":
            per_paper.append({
                "paper_id": paper.arxiv_id,
                "paper_type": paper.paper_type,
                "read_status": "error",
                "error": response.get("message", "unknown"),
                "download_latency_s": t1 - t0,
            })
            continue

        content = response.get("content", "")
        per_paper.append({
            "paper_id": paper.arxiv_id,
            "paper_type": paper.paper_type,
            "download_status": "success",
            "read_status": "success",
            "download_latency_s": t1 - t0,
            "char_count": len(content),
            "char_count_score": char_count_score(len(content), paper.min_char_count),
            "keyword_score": keyword_presence(content, paper.ground_truth_keywords),
            "section_score": section_header_presence(content, paper.expected_sections),
        })

    # --- Aggregates ---
    successful = [r for r in per_paper if r.get("read_status") == "success"]
    aggregate = {
        "success_rate": len(successful) / len(per_paper) if per_paper else 0.0,
        "mean_keyword_score": (
            sum(r["keyword_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_section_score": (
            sum(r["section_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_char_count": (
            sum(r["char_count"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_char_count_score": (
            sum(r["char_count_score"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_download_latency_s": (
            sum(r["download_latency_s"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "total_papers": len(per_paper),
        "total_errors": len(per_paper) - len(successful),
    }

    return {"per_paper": per_paper, "aggregate": aggregate}


async def run_alphaxiv_qa_eval(
    test_papers: list[PaperSpec],
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    """Evaluate AlphaXiv Q&A via answer_pdf_queries."""
    from alphaxiv_tools import answer_queries_alphaxiv

    per_question: list[dict[str, Any]] = []

    for paper in test_papers:
        if not paper.qa_questions:
            continue
        for question in paper.qa_questions:
            await rate_limiter.wait()
            logger.info("AlphaXiv QA: %s — %s", paper.arxiv_id, question[:50])

            t0 = time.monotonic()
            response = await _alphaxiv_call(
                answer_queries_alphaxiv, paper.arxiv_id, question
            )
            t1 = time.monotonic()

            if response.get("status") != "success":
                per_question.append({
                    "paper_id": paper.arxiv_id,
                    "question": question,
                    "status": "error",
                    "error": response.get("message", "unknown"),
                    "latency_s": t1 - t0,
                })
                continue

            content = response.get("content", "")
            per_question.append({
                "paper_id": paper.arxiv_id,
                "question": question,
                "status": "success",
                "answer_length": len(content),
                "has_content": len(content) > 20,
                "latency_s": t1 - t0,
            })

    # --- Aggregates ---
    successful = [r for r in per_question if r.get("status") == "success"]
    aggregate = {
        "success_rate": len(successful) / len(per_question) if per_question else 0.0,
        "mean_answer_length": (
            sum(r["answer_length"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "mean_latency_s": (
            sum(r["latency_s"] for r in successful) / len(successful)
            if successful else 0.0
        ),
        "total_questions": len(per_question),
        "total_errors": len(per_question) - len(successful),
    }

    return {"per_question": per_question, "aggregate": aggregate}
