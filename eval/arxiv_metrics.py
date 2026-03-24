"""Metric computation functions for arxiv app evaluation.

All functions are pure — no side effects, no network calls.
"""

from __future__ import annotations

import re


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip version suffix from an arxiv ID (e.g. '1706.03762v7' -> '1706.03762')."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """Fraction of top-k retrieved results that are in the relevant set."""
    top_k = [normalize_arxiv_id(rid) for rid in retrieved_ids[:k]]
    normed_relevant = {normalize_arxiv_id(r) for r in relevant_ids}
    if not top_k:
        return 0.0
    return len(set(top_k) & normed_relevant) / len(top_k)


def recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of relevant items that appear anywhere in retrieved results."""
    if not relevant_ids:
        return 1.0
    normed_retrieved = {normalize_arxiv_id(rid) for rid in retrieved_ids}
    normed_relevant = {normalize_arxiv_id(r) for r in relevant_ids}
    return len(normed_retrieved & normed_relevant) / len(normed_relevant)


def keyword_presence(content: str, keywords: list[str]) -> float:
    """Fraction of ground truth keywords found in the content (case-insensitive)."""
    if not keywords:
        return 1.0
    content_lower = content.lower()
    found = sum(1 for kw in keywords if kw.lower() in content_lower)
    return found / len(keywords)


def section_header_presence(content: str, expected_sections: list[str]) -> float:
    """Fraction of expected section headers found in the content (case-insensitive substring)."""
    if not expected_sections:
        return 1.0
    content_lower = content.lower()
    found = sum(1 for s in expected_sections if s.lower() in content_lower)
    return found / len(expected_sections)


def char_count_score(actual: int, expected_min: int) -> float:
    """1.0 if actual >= expected_min, else the fraction actual/expected_min."""
    if expected_min <= 0:
        return 1.0
    if actual >= expected_min:
        return 1.0
    return actual / expected_min


def recommendation_relevance(text: str, interest_keywords: list[str]) -> float:
    """Fraction of interest keywords appearing in recommendation text (case-insensitive)."""
    if not interest_keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in interest_keywords if kw.lower() in text_lower)
    return found / len(interest_keywords)


def dedup_correctness(run1_ids: set[str], run2_ids: set[str]) -> bool:
    """True if the second run has no paper ID overlap with the first."""
    normed1 = {normalize_arxiv_id(r) for r in run1_ids}
    normed2 = {normalize_arxiv_id(r) for r in run2_ids}
    return len(normed1 & normed2) == 0


def extract_paper_ids_from_content(content: str) -> set[str]:
    """Extract arxiv paper IDs from recommendation text using regex."""
    # Matches patterns like arXiv:2301.00001 or arxiv.org/abs/2301.00001
    pattern = r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})"
    return {normalize_arxiv_id(m) for m in re.findall(pattern, content, re.IGNORECASE)}
