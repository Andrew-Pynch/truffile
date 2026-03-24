#!/usr/bin/env python3
"""ArXiv truffle app evaluation harness.

Benchmarks search quality, content extraction, and background worker
recommendations against a static test set. Produces JSON reports and
markdown summary tables.

Usage:
    python eval/arxiv_eval.py --tag v1-baseline
    python eval/arxiv_eval.py --tag smoke --skip-download --skip-bg
    python eval/arxiv_eval.py --tag v2-test --llm-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure eval/ and app-store/arxiv/ are importable
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent
_ARXIV_APP_DIR = _REPO_ROOT / "app-store" / "arxiv"

if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
if str(_ARXIV_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_ARXIV_APP_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the arxiv truffle app against a static test set."
    )
    parser.add_argument(
        "--tag", default="v1-baseline",
        help="Version tag for this eval run (default: v1-baseline)",
    )
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip search quality tests",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip content extraction tests (faster iteration)",
    )
    parser.add_argument(
        "--skip-bg", action="store_true",
        help="Skip background worker tests",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable LLM-based scoring for bg worker (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--output-dir", default=str(_EVAL_DIR / "results"),
        help="Directory for JSON reports (default: eval/results)",
    )
    parser.add_argument(
        "--papers", nargs="*",
        help="Specific paper IDs to test (default: all papers in test set)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Markdown report formatting
# ---------------------------------------------------------------------------

def _fmt(val: float | int | bool | None, is_pct: bool = False) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "YES" if val else "NO"
    if isinstance(val, int):
        return str(val)
    if is_pct:
        return f"{val:.1%}"
    return f"{val:.3f}"


def print_markdown_report(results: dict) -> None:
    """Print a human-readable markdown summary to stdout."""
    tag = results.get("tag", "unknown")
    duration = results.get("duration_s", 0)
    print(f"\n{'=' * 60}")
    print(f"  Arxiv Eval Results: {tag}")
    print(f"  Duration: {duration:.1f}s")
    print(f"{'=' * 60}\n")

    # --- Search ---
    if "search" in results:
        agg = results["search"]["aggregate"]
        print("### Search Quality\n")
        print("| Metric                    | Value  |")
        print("|---------------------------|--------|")
        print(f"| Known-Item Hit Rate       | {_fmt(agg.get('known_item_hit_rate'), is_pct=True):>6} |")
        print(f"| Known-Item Mean Latency   | {_fmt(agg.get('known_item_mean_latency_s')):>5}s |")
        print(f"| Topic Mean Precision@5    | {_fmt(agg.get('topic_mean_precision_at_5')):>6} |")
        print(f"| Topic Mean Recall         | {_fmt(agg.get('topic_mean_recall')):>6} |")
        print(f"| Topic Mean Latency        | {_fmt(agg.get('topic_mean_latency_s')):>5}s |")
        print(f"| Total Queries             | {agg.get('total_queries', 0):>6} |")
        print(f"| Total Errors              | {agg.get('total_errors', 0):>6} |")
        print()

    # --- Extraction ---
    if "extraction" in results:
        agg = results["extraction"]["aggregate"]
        print("### Content Extraction\n")
        print("| Metric                    | Value  |")
        print("|---------------------------|--------|")
        print(f"| Success Rate              | {_fmt(agg.get('success_rate'), is_pct=True):>6} |")
        print(f"| Mean Keyword Score        | {_fmt(agg.get('mean_keyword_score')):>6} |")
        print(f"| Mean Section Score        | {_fmt(agg.get('mean_section_score')):>6} |")
        print(f"| Mean Char Count           | {int(agg.get('mean_char_count', 0)):>6} |")
        print(f"| Mean Char Count Score     | {_fmt(agg.get('mean_char_count_score')):>6} |")
        print(f"| Mean Download Latency     | {_fmt(agg.get('mean_download_latency_s')):>5}s |")
        print()

        # Per-paper breakdown
        per_paper = results["extraction"].get("per_paper", [])
        if per_paper:
            print("| Paper ID      | Type       | KW    | Sect  |  Chars | DL(s) |")
            print("|---------------|------------|-------|-------|--------|-------|")
            for p in per_paper:
                pid = p.get("paper_id", "?")[:13]
                ptype = p.get("paper_type", "?")[:10]
                if p.get("read_status") == "success":
                    print(
                        f"| {pid:<13} | {ptype:<10} "
                        f"| {_fmt(p.get('keyword_score')):>5} "
                        f"| {_fmt(p.get('section_score')):>5} "
                        f"| {p.get('char_count', 0):>6} "
                        f"| {_fmt(p.get('download_latency_s')):>5} |"
                    )
                else:
                    err = p.get("error", "unknown")[:30]
                    print(f"| {pid:<13} | {ptype:<10} | ERROR: {err:<28} |")
            print()

    # --- Background Worker ---
    if "bg_worker" in results:
        bg = results["bg_worker"]
        print("### Background Worker\n")
        print("| Check                         | Result |")
        print("|-------------------------------|--------|")
        print(f"| verify() with interests       | {_fmt(bg.get('verify_ok')):>6} |")
        print(f"| verify() without interests    | {_fmt(bg.get('empty_verify_fails')):>6} |")
        print(f"| empty returns no_interests    | {_fmt(bg.get('empty_returns_no_interests_error')):>6} |")
        print(f"| Run 1 has content             | {_fmt(bg.get('run1_has_content')):>6} |")
        print(f"| Run 1 paper count             | {bg.get('run1_count', 0):>6} |")
        print(f"| Run 2 paper count             | {bg.get('run2_count', 0):>6} |")
        print(f"| Dedup correct                 | {_fmt(bg.get('dedup_correct')):>6} |")
        print(f"| Relevance score               | {_fmt(bg.get('relevance_score')):>6} |")
        print()

        if "llm_judge" in bg and not bg["llm_judge"].get("skipped"):
            judge = bg["llm_judge"]
            print("| LLM Judge                     | Result |")
            print("|-------------------------------|--------|")
            print(f"| Mean Score (1-5)              | {_fmt(judge.get('mean_score')):>6} |")
            print(f"| Papers Scored                 | {judge.get('num_scored', 0):>6} |")
            print(f"| Scoring Failures              | {judge.get('num_failed', 0):>6} |")
            print()
            for s in judge.get("scores", []):
                title = s.get("paper_title", "?")[:40]
                score = s.get("score", "ERR")
                reason = s.get("reason", s.get("error", ""))[:50]
                print(f"  {score}/5  {title}  — {reason}")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from arxiv_runners import RateLimiter, run_bg_worker_eval, run_extraction_eval, run_search_eval
    from arxiv_test_set import BG_WORKER_INTERESTS, SEARCH_QUERIES, TEST_PAPERS

    # Filter papers if --papers specified
    papers = TEST_PAPERS
    if args.papers:
        paper_filter = set(args.papers)
        papers = [p for p in TEST_PAPERS if p.arxiv_id in paper_filter]
        if not papers:
            print(f"ERROR: No papers matched filter: {args.papers}", file=sys.stderr)
            sys.exit(1)

    rate_limiter = RateLimiter(min_delay=3.5)

    # Set up isolated storage via temp dirs
    storage_tmp = tempfile.mkdtemp(prefix="arxiv_eval_storage_")
    state_tmp = tempfile.mkdtemp(prefix="arxiv_eval_state_")
    os.environ["ARXIV_STORAGE_PATH"] = storage_tmp
    os.environ["ARXIV_BG_STATE_PATH"] = os.path.join(state_tmp, "bg_state.json")

    results: dict = {
        "tag": args.tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_paper_count": len(papers),
    }

    t_start = time.monotonic()

    logger = logging.getLogger("arxiv_eval")

    try:
        # --- Search eval ---
        if not args.skip_search:
            logger.info("Starting search eval (%d papers, %d queries)...", len(papers), len(SEARCH_QUERIES))
            results["search"] = await run_search_eval(papers, SEARCH_QUERIES, rate_limiter)

        # --- Extraction eval ---
        if not args.skip_download:
            logger.info("Starting extraction eval (%d papers)...", len(papers))
            results["extraction"] = await run_extraction_eval(papers, rate_limiter)

        # --- Background worker eval ---
        if not args.skip_bg:
            logger.info("Starting background worker eval...")
            results["bg_worker"] = run_bg_worker_eval(
                BG_WORKER_INTERESTS, rate_limiter, llm_judge=args.llm_judge
            )

    finally:
        # Clean up temp dirs
        import shutil
        shutil.rmtree(storage_tmp, ignore_errors=True)
        shutil.rmtree(state_tmp, ignore_errors=True)

    results["duration_s"] = round(time.monotonic() - t_start, 2)

    # --- Write JSON report ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = output_dir / f"{args.tag}_{date_str}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logging.getLogger("arxiv_eval").info("Report written to %s", json_path)

    # --- Print markdown summary ---
    print_markdown_report(results)


if __name__ == "__main__":
    asyncio.run(main())
