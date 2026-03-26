"""Centralized configuration for the ArXiv Truffle app.

AlphaXiv auth comes from env vars set by the installer's OAuth step.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("arxiv.config")

# ---------------------------------------------------------------------------
# v1 arXiv (always available, no auth)
# ---------------------------------------------------------------------------

ARXIV_RESEARCH_INTERESTS: str = os.getenv("ARXIV_RESEARCH_INTERESTS", "")

# ---------------------------------------------------------------------------
# AlphaXiv (optional)
# ---------------------------------------------------------------------------

ALPHAXIV_MCP_URL: str = os.getenv(
    "ALPHAXIV_MCP_URL", "https://api.alphaxiv.org/mcp/v1"
).strip()

_SENTINELS = frozenset({"none", "", "null", "undefined", "n/a"})


def is_alphaxiv_configured() -> bool:
    """Return True if alphaxiv credentials are available (refresh token or access token)."""
    for var in ("ALPHAXIV_REFRESH_TOKEN", "ALPHAXIV_ACCESS_TOKEN"):
        val = os.getenv(var, "").strip()
        if val.lower() not in _SENTINELS:
            return True
    return False


def is_alphaxiv_enrich_enabled() -> bool:
    """Enabled by default when AlphaXiv is configured. Set ALPHAXIV_ENRICH_PDF=0 to disable."""
    val = os.getenv("ALPHAXIV_ENRICH_PDF", "").strip().lower()
    if val in ("0", "false", "no"):
        return False
    return is_alphaxiv_configured()
