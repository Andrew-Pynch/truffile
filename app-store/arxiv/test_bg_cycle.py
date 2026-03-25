"""Local test harness for the ArXiv background worker.

Run from the arxiv app directory:
    cd app-store/arxiv && python test_bg_cycle.py

Tests what the background worker would find and submit, without needing
the device or app_runtime.
"""

import logging
import os
import sys

# Set up visible logging
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# Use test interests (override env if not set)
if not os.getenv("ARXIV_RESEARCH_INTERESTS"):
    os.environ["ARXIV_RESEARCH_INTERESTS"] = (
        "multi-agent systems, reinforcement learning, LLM safety"
    )

# Override paths that default to /root/ (not writable locally)
if not os.getenv("ARXIV_BG_STATE_PATH"):
    os.environ["ARXIV_BG_STATE_PATH"] = "/tmp/arxiv_bg_state.json"
if not os.getenv("ARXIV_STORAGE_PATH"):
    os.environ["ARXIV_STORAGE_PATH"] = "/tmp/arxiv-papers"

from arxiv_bg_worker import ArxivBackgroundWorker

def main() -> None:
    interests = os.environ["ARXIV_RESEARCH_INTERESTS"]
    print(f"=== ArXiv Background Worker Test Harness ===")
    print(f"Interests: {interests}")
    print()

    worker = ArxivBackgroundWorker(interests_raw=interests)

    # 1. Verify config
    ok, msg = worker.verify()
    print(f"Verify: {'OK' if ok else 'FAIL'} — {msg}")
    if not ok:
        sys.exit(1)

    # 2. Run a cycle
    print()
    print("Running cycle...")
    result = worker.run_cycle()

    if result.error:
        print(f"ERROR: {result.error}")
        sys.exit(1)

    if not result.content:
        print("No new papers found (all seen or no matches)")
        sys.exit(0)

    # 3. Print what would be submitted as context
    print()
    print("=== Context that would be submitted via submit_context() ===")
    print(result.content)
    print()
    print(f"=== Content length: {len(result.content)} chars ===")


if __name__ == "__main__":
    main()
