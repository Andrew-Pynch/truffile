"""Trace the AlphaXiv background search pipeline step by step.

Shows exactly where results get dropped: search, ID extraction,
dedup, metadata lookup, or recommendation building.

Usage:
    cd app-store/arxiv && python test_alphaxiv_bg.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib

# Load .env + override paths
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("ARXIV_RESEARCH_INTERESTS", "multi-agent systems, reinforcement learning, LLM safety")
os.environ.setdefault("ARXIV_BG_STATE_PATH", "/tmp/arxiv_alphaxiv_trace.json")
os.environ.setdefault("ARXIV_STORAGE_PATH", "/tmp/arxiv_test_papers")

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

from alphaxiv_client import AlphaXivClient
from alphaxiv_tools import search_papers_alphaxiv, extract_arxiv_ids
from arxiv_bg_worker import ArxivBackgroundWorker
from arxiv_common import parse_research_interests


async def trace() -> None:
    interests = parse_research_interests(os.environ["ARXIV_RESEARCH_INTERESTS"])
    print(f"Interests: {interests}")
    print()

    # Step 1: Auth
    print("=== Step 1: AlphaXiv Connection ===")
    try:
        async with AlphaXivClient() as client:
            print(f"  Connected: healthy={client.is_healthy}")

            for interest in interests:
                print(f"\n=== Interest: '{interest}' ===")

                # Step 2: Raw search
                print("\n  Step 2: Agentic search...")
                result = await search_papers_alphaxiv(client, interest, mode="agentic")
                status = result.get("status")
                content = result.get("content", "")
                message = result.get("message", "")
                print(f"    Status: {status}")
                if status != "success":
                    print(f"    Error: {message}")
                    continue
                print(f"    Content length: {len(content)} chars")
                print(f"    Content preview: {content[:300]}...")

                # Step 3: Extract IDs
                print("\n  Step 3: Extract arXiv IDs...")
                ids = extract_arxiv_ids(content)
                print(f"    IDs found: {ids}")
                if not ids:
                    print("    STOP: No IDs extracted from content")
                    continue

                # Step 4: Dedup (empty seen_ids for clean test)
                seen_ids: set[str] = set()
                new_ids = [pid for pid in ids if pid not in seen_ids]
                print(f"\n  Step 4: Dedup (seen_ids empty)...")
                print(f"    New IDs: {new_ids}")

                # Step 5: Metadata lookup
                print("\n  Step 5: Fetch arXiv metadata...")
                import arxiv
                arxiv_client = arxiv.Client()
                search = arxiv.Search(id_list=new_ids)
                for paper in arxiv_client.results(search):
                    pid = paper.get_short_id()
                    print(f"    [{pid}] {paper.title.strip()[:80]}")

                print(f"\n  Pipeline complete for '{interest}' — {len(new_ids)} papers would be recommended")

    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()


def main() -> None:
    print("=" * 60)
    print("  AlphaXiv Background Pipeline Trace")
    print("=" * 60)
    asyncio.run(trace())


if __name__ == "__main__":
    main()
