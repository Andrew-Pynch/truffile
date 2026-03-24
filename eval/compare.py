#!/usr/bin/env python3
"""Compare two arxiv eval result JSON files and output a markdown delta table.

Usage:
    python eval/compare.py eval/results/v1-baseline_2026-03-24.json eval/results/v2-new_2026-03-25.json
    python eval/compare.py --json old.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two arxiv eval result JSON files."
    )
    parser.add_argument("old", help="Path to the baseline (old) result JSON")
    parser.add_argument("new", help="Path to the new result JSON")
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output diff as JSON instead of markdown",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

# Metrics where lower is better (latency, errors).  Delta sign is inverted
# for regression detection on these.
LOWER_IS_BETTER = {
    "known_item_mean_latency_s",
    "topic_mean_latency_s",
    "mean_download_latency_s",
    "total_errors",
    "num_failed",
}


def _collect_metrics(data: dict, prefix: str = "") -> dict[str, float]:
    """Recursively collect all numeric leaf values from a nested dict."""
    out: dict[str, float] = {}
    for key, val in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(_collect_metrics(val, path))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            out[path] = val
    return out


def _section_for_path(path: str) -> str:
    """Map a metric path to a display section name."""
    if path.startswith("search"):
        return "Search Quality"
    if path.startswith("extraction"):
        return "Content Extraction"
    if path.startswith("bg_worker"):
        return "Background Worker"
    return "Other"


def _label_for_path(path: str) -> str:
    """Convert a dotted path to a human-readable label."""
    # Strip the top-level section and "aggregate" prefix
    parts = path.split(".")
    # Remove known structural prefixes
    skip = {"search", "extraction", "bg_worker", "aggregate", "llm_judge"}
    meaningful = [p for p in parts if p not in skip]
    return " ".join(meaningful).replace("_", " ").title()


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def compute_diff(
    old_data: dict, new_data: dict,
) -> tuple[str, str, list[dict[str, Any]], bool]:
    """Compute metric deltas between two result files.

    Returns: (old_tag, new_tag, diff_rows, has_regression)
    """
    old_tag = old_data.get("tag", "old")
    new_tag = new_data.get("tag", "new")

    old_metrics = _collect_metrics(old_data)
    new_metrics = _collect_metrics(new_data)

    # Only compare metrics present in both
    common_keys = sorted(set(old_metrics) & set(new_metrics))

    rows: list[dict[str, Any]] = []
    has_regression = False

    for key in common_keys:
        # Skip metadata fields
        if key in ("duration_s", "test_paper_count"):
            continue

        old_val = old_metrics[key]
        new_val = new_metrics[key]
        delta = new_val - old_val
        delta_pct = (delta / old_val * 100) if old_val != 0 else (100.0 if delta != 0 else 0.0)

        # Determine if this is a regression
        metric_name = key.split(".")[-1]
        if metric_name in LOWER_IS_BETTER:
            is_regression = delta > 0.001  # higher is worse for these
        else:
            is_regression = delta < -0.001  # lower is worse for quality metrics

        if is_regression:
            has_regression = True

        rows.append({
            "path": key,
            "section": _section_for_path(key),
            "label": _label_for_path(key),
            "old": old_val,
            "new": new_val,
            "delta": delta,
            "delta_pct": delta_pct,
            "regression": is_regression,
        })

    return old_tag, new_tag, rows, has_regression


# ---------------------------------------------------------------------------
# Per-paper extraction diff
# ---------------------------------------------------------------------------

def compute_per_paper_diff(old_data: dict, new_data: dict) -> list[dict[str, Any]]:
    """Compare per-paper extraction results if both files have them."""
    old_papers = {
        p["paper_id"]: p
        for p in old_data.get("extraction", {}).get("per_paper", [])
    }
    new_papers = {
        p["paper_id"]: p
        for p in new_data.get("extraction", {}).get("per_paper", [])
    }

    common = sorted(set(old_papers) & set(new_papers))
    if not common:
        return []

    rows = []
    for pid in common:
        old_p = old_papers[pid]
        new_p = new_papers[pid]
        row: dict[str, Any] = {"paper_id": pid}
        for metric in ("keyword_score", "section_score", "char_count", "download_latency_s"):
            old_val = old_p.get(metric)
            new_val = new_p.get(metric)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                row[f"{metric}_old"] = old_val
                row[f"{metric}_new"] = new_val
                row[f"{metric}_delta"] = new_val - old_val
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_val(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.3f}"


def _fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    if abs(d) >= 100:
        return f"{sign}{d:,.0f}"
    return f"{sign}{d:.3f}"


def _fmt_pct(p: float) -> str:
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def print_markdown(
    old_tag: str,
    new_tag: str,
    rows: list[dict],
    per_paper: list[dict],
    has_regression: bool,
) -> None:
    """Print markdown comparison tables to stdout."""
    print(f"\n## {old_tag} -> {new_tag}\n")

    if has_regression:
        print("**WARNING: Regressions detected**\n")

    # Group by section
    sections: dict[str, list[dict]] = {}
    for row in rows:
        sections.setdefault(row["section"], []).append(row)

    for section, section_rows in sections.items():
        print(f"### {section}\n")
        print(f"| Metric | {old_tag} | {new_tag} | Delta | % |")
        print("|--------|--------:|--------:|------:|----:|")
        for r in section_rows:
            marker = " (!)" if r["regression"] else ""
            print(
                f"| {r['label']}{marker} "
                f"| {_fmt_val(r['old'])} "
                f"| {_fmt_val(r['new'])} "
                f"| {_fmt_delta(r['delta'])} "
                f"| {_fmt_pct(r['delta_pct'])} |"
            )
        print()

    # Per-paper extraction diff
    if per_paper:
        print("### Per-Paper Extraction Comparison\n")
        print("| Paper | KW Old | KW New | KW Delta | Sect Old | Sect New | Sect Delta |")
        print("|-------|-------:|-------:|---------:|---------:|---------:|-----------:|")
        for r in per_paper:
            pid = r["paper_id"][:13]
            kw_old = r.get("keyword_score_old", "")
            kw_new = r.get("keyword_score_new", "")
            kw_d = r.get("keyword_score_delta", "")
            s_old = r.get("section_score_old", "")
            s_new = r.get("section_score_new", "")
            s_d = r.get("section_score_delta", "")
            print(
                f"| {pid} "
                f"| {_fmt_val(kw_old) if isinstance(kw_old, (int, float)) else 'N/A'} "
                f"| {_fmt_val(kw_new) if isinstance(kw_new, (int, float)) else 'N/A'} "
                f"| {_fmt_delta(kw_d) if isinstance(kw_d, (int, float)) else 'N/A'} "
                f"| {_fmt_val(s_old) if isinstance(s_old, (int, float)) else 'N/A'} "
                f"| {_fmt_val(s_new) if isinstance(s_new, (int, float)) else 'N/A'} "
                f"| {_fmt_delta(s_d) if isinstance(s_d, (int, float)) else 'N/A'} |"
            )
        print()


def main() -> None:
    args = parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)

    if not old_path.exists():
        print(f"ERROR: File not found: {old_path}", file=sys.stderr)
        sys.exit(2)
    if not new_path.exists():
        print(f"ERROR: File not found: {new_path}", file=sys.stderr)
        sys.exit(2)

    old_data = json.loads(old_path.read_text(encoding="utf-8"))
    new_data = json.loads(new_path.read_text(encoding="utf-8"))

    old_tag, new_tag, rows, has_regression = compute_diff(old_data, new_data)
    per_paper = compute_per_paper_diff(old_data, new_data)

    if args.output_json:
        output = {
            "old_tag": old_tag,
            "new_tag": new_tag,
            "metrics": rows,
            "per_paper": per_paper,
            "has_regression": has_regression,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_markdown(old_tag, new_tag, rows, per_paper, has_regression)

    # Exit code: 1 if any regressions detected
    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
