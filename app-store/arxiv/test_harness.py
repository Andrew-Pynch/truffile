"""Comprehensive test harness for the ArXiv Truffle app.

Exercises every code path locally — v1 search, AlphaXiv integration,
background cycle, download pipeline — with clear PASS/FAIL/SKIP output.

Usage:
    cd app-store/arxiv && python test_harness.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Environment overrides (must be set BEFORE importing app modules)
# ---------------------------------------------------------------------------

# Load .env file if present (for AlphaXiv credentials)
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

os.environ.setdefault("ARXIV_RESEARCH_INTERESTS", "multi-agent systems, reinforcement learning, LLM safety")
os.environ.setdefault("ARXIV_BG_STATE_PATH", "/tmp/arxiv_test_bg_state.json")
os.environ.setdefault("ARXIV_STORAGE_PATH", "/tmp/arxiv_test_papers")

# Clean test state
pathlib.Path("/tmp/arxiv_test_bg_state.json").unlink(missing_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Imports (after env setup)
# ---------------------------------------------------------------------------

import arxiv as arxiv_lib
from arxiv_bg_worker import ArxivBackgroundWorker, _run_async, _ALPHAXIV_AVAILABLE
from arxiv_tools import search_papers, download_paper, read_paper
from arxiv_config import is_alphaxiv_configured

if _ALPHAXIV_AVAILABLE:
    from alphaxiv_client import AlphaXivClient
    from alphaxiv_tools import (
        search_papers_alphaxiv,
        read_paper_alphaxiv,
        answer_queries_alphaxiv,
        extract_arxiv_ids,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTEREST = "multi-agent systems"
TEST_PAPER_ID = "2301.01379"  # well-known paper for testing
_pass_count = 0
_fail_count = 0
_skip_count = 0


def header(num: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  TEST {num}: {title}")
    print(f"{'='*60}")


def result(status: str, detail: str = "") -> None:
    global _pass_count, _fail_count, _skip_count
    icon = {"PASS": "+", "FAIL": "X", "SKIP": "-"}[status]
    if status == "PASS":
        _pass_count += 1
    elif status == "FAIL":
        _fail_count += 1
    else:
        _skip_count += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{icon}] {status}{suffix}")


def print_papers(papers: list, max_show: int = 5) -> None:
    for i, p in enumerate(papers[:max_show]):
        if isinstance(p, dict):
            title = p.get("title", "?")[:80]
            pid = p.get("id", "?")
        else:
            title = (p.title or "?").strip()[:80]
            pid = p.get_short_id() if hasattr(p, "get_short_id") else "?"
        print(f"    {i+1}. [{pid}] {title}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_v1_search_relevance_comparison() -> None:
    header(1, "v1 arXiv Search — SubmittedDate vs Relevance")

    client = arxiv_lib.Client()

    # SubmittedDate (what background currently uses)
    search_date = arxiv_lib.Search(
        query=INTEREST, max_results=5, sort_by=arxiv_lib.SortCriterion.SubmittedDate,
    )
    date_results = list(client.results(search_date))

    # Relevance (what foreground uses)
    search_rel = arxiv_lib.Search(
        query=INTEREST, max_results=5, sort_by=arxiv_lib.SortCriterion.Relevance,
    )
    rel_results = list(client.results(search_rel))

    print("\n  sort_by=SubmittedDate (background uses this):")
    print_papers(date_results)

    print("\n  sort_by=Relevance (foreground uses this):")
    print_papers(rel_results)

    # Check if relevance results actually contain the interest term in title
    rel_titles = " ".join((p.title or "").lower() for p in rel_results)
    date_titles = " ".join((p.title or "").lower() for p in date_results)
    rel_hits = sum(1 for w in INTEREST.lower().split() if w in rel_titles)
    date_hits = sum(1 for w in INTEREST.lower().split() if w in date_titles)

    print(f"\n  Title keyword hits — Relevance: {rel_hits}/3, SubmittedDate: {date_hits}/3")
    if rel_hits > date_hits:
        result("PASS", "Relevance sort produces more relevant results (as expected)")
    elif rel_hits == date_hits:
        result("PASS", "Both sorts equally relevant (unusual but ok)")
    else:
        result("FAIL", "SubmittedDate somehow more relevant than Relevance sort")


def test_2_foreground_search_papers() -> None:
    header(2, "Foreground search_papers() — default Relevance sort")

    resp = asyncio.run(search_papers(query=INTEREST, max_results=5))
    status = resp.get("status")
    papers = resp.get("papers", [])

    if status == "success" and papers:
        print(f"\n  Found {resp.get('total_results', 0)} papers:")
        print_papers(papers)
        result("PASS", f"{len(papers)} papers returned")
    elif status == "success":
        result("FAIL", "Search succeeded but returned 0 papers")
    else:
        result("FAIL", resp.get("message", "unknown error"))


def test_3_alphaxiv_auth() -> bool:
    header(3, "AlphaXiv Auth Check")

    if not _ALPHAXIV_AVAILABLE:
        result("SKIP", "AlphaXiv dependencies not installed")
        return False

    if not is_alphaxiv_configured():
        result("SKIP", "No AlphaXiv credentials configured")
        return False

    print("  Credentials found, attempting connection...")
    try:
        async def _check():
            async with AlphaXivClient() as client:
                return client.is_healthy
        healthy = asyncio.run(_check())
        if healthy:
            result("PASS", "AlphaXiv MCP session established")
            return True
        else:
            result("FAIL", "Client connected but reports unhealthy")
            return False
    except Exception as exc:
        result("FAIL", f"Auth/connection failed: {exc}")
        return False


def test_4_alphaxiv_search(auth_ok: bool) -> None:
    header(4, "AlphaXiv Search — 3 modes")

    if not auth_ok:
        result("SKIP", "AlphaXiv auth failed, skipping search tests")
        return

    query = "reinforcement learning"
    for mode in ("agentic", "semantic", "keyword"):
        print(f"\n  Mode: {mode}")
        try:
            async def _search(m=mode):
                async with AlphaXivClient() as client:
                    return await search_papers_alphaxiv(client, query, m)
            resp = asyncio.run(_search())
            status = resp.get("status")
            content = resp.get("content", "")[:300]
            if status == "success":
                ids = extract_arxiv_ids(content) if content else []
                print(f"    Content preview: {content[:150]}...")
                print(f"    Paper IDs found: {ids[:5]}")
                result("PASS", f"{mode}: {len(ids)} paper IDs extracted")
            else:
                result("FAIL", f"{mode}: {resp.get('message', 'unknown')}")
        except Exception as exc:
            result("FAIL", f"{mode}: {exc}")


def test_5_alphaxiv_read_and_answer(auth_ok: bool) -> None:
    header(5, "AlphaXiv Read + Answer")

    if not auth_ok:
        result("SKIP", "AlphaXiv auth failed, skipping")
        return

    # Read paper content
    print(f"\n  Reading paper {TEST_PAPER_ID} via AlphaXiv...")
    try:
        async def _read():
            async with AlphaXivClient() as client:
                return await read_paper_alphaxiv(client, TEST_PAPER_ID)
        resp = asyncio.run(_read())
        if resp.get("status") == "success":
            content = resp.get("content", "")
            print(f"    Content length: {len(content)} chars")
            print(f"    Preview: {content[:150]}...")
            result("PASS", "read_paper_alphaxiv returned content")
        else:
            result("FAIL", f"read: {resp.get('message', 'unknown')}")
    except Exception as exc:
        result("FAIL", f"read: {exc}")

    # Answer a question
    print(f"\n  Asking question about paper {TEST_PAPER_ID}...")
    try:
        async def _answer():
            async with AlphaXivClient() as client:
                return await answer_queries_alphaxiv(
                    client, TEST_PAPER_ID,
                    "What is the main contribution of this paper?"
                )
        resp = asyncio.run(_answer())
        if resp.get("status") == "success":
            content = resp.get("content", "")
            print(f"    Answer length: {len(content)} chars")
            print(f"    Preview: {content[:200]}...")
            result("PASS", "answer_queries_alphaxiv returned answer")
        else:
            result("FAIL", f"answer: {resp.get('message', 'unknown')}")
    except Exception as exc:
        result("FAIL", f"answer: {exc}")


def test_6_full_background_cycle() -> None:
    header(6, "Full Background Cycle (v1 + AlphaXiv)")

    interests = os.environ["ARXIV_RESEARCH_INTERESTS"]
    worker = ArxivBackgroundWorker(interests_raw=interests)

    ok, msg = worker.verify()
    print(f"  Verify: {msg}")
    if not ok:
        result("FAIL", msg)
        return

    t0 = time.time()
    bg_result = worker.run_cycle()
    elapsed = time.time() - t0

    if bg_result.error:
        result("FAIL", f"Cycle error: {bg_result.error}")
        return

    if not bg_result.content:
        result("FAIL", "Cycle returned no content (no papers found)")
        return

    print(f"\n  Cycle completed in {elapsed:.1f}s")
    print(f"  Context length: {len(bg_result.content)} chars")
    print(f"\n  --- submitted context ---")
    print(bg_result.content)
    print(f"  --- end context ---")

    # Count sources
    arxiv_count = bg_result.content.count("[arxiv]")
    alphaxiv_count = bg_result.content.count("[alphaxiv]")
    print(f"\n  Papers: {arxiv_count} from v1, {alphaxiv_count} from AlphaXiv")

    result("PASS", f"{arxiv_count + alphaxiv_count} total papers, {elapsed:.1f}s")


def test_7_download_and_read() -> None:
    header(7, "Download + Read Pipeline")

    print(f"  Downloading paper {TEST_PAPER_ID}...")
    try:
        resp = asyncio.run(download_paper(paper_id=TEST_PAPER_ID))
        status = resp.get("status")
        print(f"  Download response: {resp.get('message', status)}")

        if status in ("success", "converting"):
            # Wait for conversion if needed
            if status == "converting":
                print("  Waiting for PDF conversion...")
                for _ in range(15):
                    time.sleep(1)
                    check = asyncio.run(download_paper(paper_id=TEST_PAPER_ID, check_status=True))
                    if check.get("status") == "success":
                        break

            resp = asyncio.run(read_paper(paper_id=TEST_PAPER_ID))
            if resp.get("status") == "success":
                content = resp.get("content", "")
                print(f"  Paper content: {len(content)} chars")
                print(f"  First 200 chars: {content[:200]}...")
                result("PASS", f"Download + read OK, {len(content)} chars")
            else:
                result("FAIL", f"Read failed: {resp.get('message')}")
        else:
            result("FAIL", f"Download failed: {resp.get('message', status)}")
    except Exception as exc:
        result("FAIL", f"Pipeline error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  ArXiv App — Comprehensive Test Harness")
    print(f"  Interests: {os.environ['ARXIV_RESEARCH_INTERESTS']}")
    print(f"  AlphaXiv available: {_ALPHAXIV_AVAILABLE}")
    print(f"  AlphaXiv configured: {is_alphaxiv_configured()}")
    print("=" * 60)

    t0 = time.time()

    test_1_v1_search_relevance_comparison()
    test_2_foreground_search_papers()
    auth_ok = test_3_alphaxiv_auth()
    test_4_alphaxiv_search(auth_ok)
    test_5_alphaxiv_read_and_answer(auth_ok)
    test_6_full_background_cycle()
    test_7_download_and_read()

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RESULTS: {_pass_count} passed, {_fail_count} failed, {_skip_count} skipped")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
