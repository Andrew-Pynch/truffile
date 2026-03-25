"""AlphaXiv MCP client with Clerk OAuth token management.

Connects to AlphaXiv's MCP server over streamable HTTP transport.

Token lifecycle
---------------
There are two authentication paths:

1. **Development (local machine):**
   ``sync_creds.py`` reads tokens from Claude Code's credential cache
   (``~/.claude/.credentials.json``), writes them into ``truffile.yaml``
   env vars, and optionally into ``.env`` for local test scripts.
   Run ``python sync_creds.py`` before every ``truffile deploy``.

2. **Production (Truffle device):**
   The installer's OAuth step (``truffile.yaml`` ``type: oauth``) handles
   the Clerk sign-in flow and injects ``ALPHAXIV_REFRESH_TOKEN``,
   ``ALPHAXIV_CLIENT_ID``, and ``ALPHAXIV_ACCESS_TOKEN`` into the process
   environment.  ``~/.claude/.credentials.json`` does not exist on device,
   so ``ensure_loaded()`` goes straight to ``_load_env_fallback()``.

In both paths, ``ClaudeCredentialAuth`` refreshes expired access tokens
via Clerk's OIDC token endpoint using the refresh token + client ID.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import contextlib

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from arxiv_config import ALPHAXIV_MCP_URL, _CLAUDE_CREDS_PATH, _SENTINELS

logger = logging.getLogger("arxiv.alphaxiv_client")


# ---------------------------------------------------------------------------
# Auth handler
# ---------------------------------------------------------------------------


class ClaudeCredentialAuth:
    """Token manager with Clerk refresh support.

    Not an httpx.Auth subclass — tokens are passed as plain headers to the
    MCP transport to avoid interfering with SSE response streaming.

    Resolution order:
    1. Claude credential cache  (~/.claude/.credentials.json → mcpOAuth → alphaxiv|*)
    2. ALPHAXIV_ACCESS_TOKEN env var  (no refresh capability)
    """

    def __init__(self, credentials_path: Path | None = None) -> None:
        self._creds_path = credentials_path or _CLAUDE_CREDS_PATH
        self._creds_key: str | None = None  # e.g. "alphaxiv|d940b2c43ce9ee4d"

        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0  # unix seconds
        self._client_id: str | None = None
        self._auth_server_url: str | None = None  # e.g. "https://clerk.alphaxiv.org"
        self._token_endpoint: str | None = None  # discovered from OIDC config

        self._loaded = False
        self.is_healthy = True

    # ---- credential loading ------------------------------------------------

    def _load_cached(self) -> bool:
        """Load tokens from Claude credential cache.  Returns True if found."""
        if not self._creds_path.exists():
            return False
        try:
            data = json.loads(self._creds_path.read_text(encoding="utf-8"))
            for key, entry in data.get("mcpOAuth", {}).items():
                if not key.startswith("alphaxiv|"):
                    continue
                access = entry.get("accessToken", "")
                if not access:
                    continue
                self._creds_key = key
                self._access_token = access
                self._refresh_token = entry.get("refreshToken")
                # expiresAt is milliseconds in the cache
                expires_ms = entry.get("expiresAt", 0)
                self._expires_at = expires_ms / 1000.0 if expires_ms else 0.0
                self._client_id = entry.get("clientId")
                ds = entry.get("discoveryState") or {}
                self._auth_server_url = ds.get("authorizationServerUrl")
                logger.debug(
                    "Loaded AlphaXiv credentials from cache (key=%s, expires=%s)",
                    key,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._expires_at)),
                )
                return True
        except Exception as exc:
            logger.warning("Failed to read Claude credential cache: %s", exc)
        return False

    def _load_env_fallback(self) -> bool:
        """Fall back to env vars.

        Preferred path: ALPHAXIV_REFRESH_TOKEN (+ CLIENT_ID, AUTH_SERVER_URL)
        which enables self-refresh on device.  Legacy path: ALPHAXIV_ACCESS_TOKEN
        (static, no refresh — token dies in ~60s).
        """
        # --- preferred: refresh-token path ---
        refresh = os.getenv("ALPHAXIV_REFRESH_TOKEN", "").strip()
        if refresh.lower() not in _SENTINELS:
            self._refresh_token = refresh
            self._client_id = os.getenv("ALPHAXIV_CLIENT_ID", "").strip() or None
            self._auth_server_url = (
                os.getenv("ALPHAXIV_AUTH_SERVER_URL", "").strip()
                or "https://clerk.alphaxiv.org"
            )
            # Use access token from env if available (e.g. synced by sync_creds.py)
            access = os.getenv("ALPHAXIV_ACCESS_TOKEN", "").strip()
            self._access_token = access if access.lower() not in _SENTINELS else None
            self._expires_at = 0.0
            logger.debug("Using ALPHAXIV_REFRESH_TOKEN env var (self-refresh enabled)")
            return True

        # --- legacy: static access-token path ---
        token = os.getenv("ALPHAXIV_ACCESS_TOKEN", "").strip()
        if token.lower() not in _SENTINELS:
            self._access_token = token
            self._refresh_token = None
            self._expires_at = 0.0  # unknown expiry; try until 401
            logger.debug("Using ALPHAXIV_ACCESS_TOKEN env var (no refresh)")
            return True

        return False

    # ---- OIDC discovery & token refresh ------------------------------------

    async def _discover_token_endpoint(self) -> str | None:
        """Discover Clerk's token endpoint via OIDC configuration."""
        if self._token_endpoint:
            return self._token_endpoint
        if not self._auth_server_url:
            return None
        url = f"{self._auth_server_url.rstrip('/')}/.well-known/openid-configuration"
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                data = resp.json()
                self._token_endpoint = data.get("token_endpoint")
                logger.debug("Discovered Clerk token endpoint: %s", self._token_endpoint)
                return self._token_endpoint
        except Exception as exc:
            logger.warning("OIDC discovery failed for %s: %s", url, exc)
            return None

    async def _do_refresh(self) -> bool:
        """Refresh the access token via Clerk's token endpoint."""
        if not self._refresh_token or not self._client_id:
            logger.debug("Cannot refresh: missing refresh_token or client_id")
            return False

        token_url = await self._discover_token_endpoint()
        if not token_url:
            logger.warning("Cannot refresh: token endpoint not discovered")
            return False

        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                        "client_id": self._client_id,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code != 200:
                    logger.warning("Token refresh failed: %s %s", resp.status_code, resp.text[:200])
                    return False

                token_data = resp.json()
                self._access_token = token_data["access_token"]
                if "refresh_token" in token_data:
                    self._refresh_token = token_data["refresh_token"]
                expires_in = token_data.get("expires_in")
                if expires_in is not None:
                    self._expires_at = time.time() + int(expires_in)
                else:
                    self._expires_at = 0.0

                # Persist refreshed tokens back to Claude credential cache
                self._write_back_to_cache()
                logger.debug("Token refreshed successfully (expires_in=%s)", expires_in)
                return True
        except Exception as exc:
            logger.warning("Token refresh error: %s", exc)
            return False

    def _write_back_to_cache(self) -> None:
        """Update the credential cache with refreshed tokens."""
        if not self._creds_key or not self._creds_path.exists():
            return
        try:
            data = json.loads(self._creds_path.read_text(encoding="utf-8"))
            entry = data.get("mcpOAuth", {}).get(self._creds_key)
            if entry is None:
                return
            entry["accessToken"] = self._access_token
            if self._refresh_token:
                entry["refreshToken"] = self._refresh_token
            entry["expiresAt"] = int(self._expires_at * 1000) if self._expires_at else 0
            self._creds_path.write_text(
                json.dumps(data, separators=(",", ":")),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Failed to write back to credential cache: %s", exc)

    # ---- public interface -----------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load credentials if not already loaded."""
        if not self._loaded:
            # On device, ~/.claude/.credentials.json doesn't exist.
            # Always use env vars (populated by sync_creds.py / YAML environment).
            self._loaded = self._load_env_fallback()
            if not self._loaded:
                self.is_healthy = False

    def get_current_token(self) -> str | None:
        """Return the current access token (loads credentials if needed)."""
        self.ensure_loaded()
        return self._access_token

    async def refresh_if_expired(self) -> bool:
        """Refresh the token if it is known to be expired.  Returns True if token is usable."""
        self.ensure_loaded()
        needs_refresh = (
            self._refresh_token
            and (
                # first boot: have refresh token but no access token yet
                self._access_token is None
                # known expired
                or (self._expires_at and time.time() > self._expires_at)
            )
        )
        if needs_refresh:
            refreshed = await self._do_refresh()
            if not refreshed:
                self.is_healthy = False
                return False
        return self._access_token is not None


# ---------------------------------------------------------------------------
# MCP client wrapper
# ---------------------------------------------------------------------------


class AlphaXivClient:
    """Async context manager wrapping an MCP SSE session to AlphaXiv.

    Usage::

        async with AlphaXivClient() as client:
            result = await client.search_keyword("attention is all you need")
            print(result)
    """

    def __init__(
        self,
        url: str = ALPHAXIV_MCP_URL,
        credentials_path: Path | None = None,
    ) -> None:
        self._url = url
        self._auth = ClaudeCredentialAuth(credentials_path)
        self._session: ClientSession | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    async def __aenter__(self) -> AlphaXivClient:
        self._auth.ensure_loaded()
        await self._auth.refresh_if_expired()
        token = self._auth.get_current_token()
        if not token:
            raise RuntimeError("No AlphaXiv credentials available")

        stack = contextlib.AsyncExitStack()
        await stack.__aenter__()

        try:
            read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                streamablehttp_client(
                    url=self._url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                    sse_read_timeout=300,
                )
            )
            session = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=120),
            )
            await stack.enter_async_context(session)
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise

        self._session = session
        self._exit_stack = stack
        logger.info("AlphaXiv MCP session initialized at %s", self._url)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._session = None
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None

    @property
    def is_healthy(self) -> bool:
        return self._auth.is_healthy and self._session is not None

    # ---- generic tool call -------------------------------------------------

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an AlphaXiv MCP tool.  Returns a status/content dict."""
        if self._session is None:
            return {"status": "error", "message": "AlphaXiv session not initialized"}
        try:
            result = await self._session.call_tool(tool_name, arguments)
            if result.isError:
                text_parts = [
                    getattr(c, "text", str(c)) for c in (result.content or [])
                ]
                return {"status": "error", "message": "\n".join(text_parts)}
            text_parts = [
                getattr(c, "text", str(c)) for c in (result.content or [])
            ]
            return {"status": "success", "content": "\n".join(text_parts)}
        except Exception as exc:
            logger.warning("AlphaXiv tool call '%s' failed: %s", tool_name, exc)
            return {"status": "error", "message": f"AlphaXiv error: {exc}"}

    # ---- convenience methods -----------------------------------------------

    async def search_agentic(self, query: str) -> dict[str, Any]:
        return await self.call("agentic_paper_retrieval", {"query": query})

    async def search_semantic(self, query: str) -> dict[str, Any]:
        return await self.call("embedding_similarity_search", {"query": query})

    async def search_keyword(self, query: str) -> dict[str, Any]:
        return await self.call("full_text_papers_search", {"query": query})

    async def get_paper_content(
        self, url: str, *, full_text: bool = False
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"url": url}
        if full_text:
            args["fullText"] = True
        return await self.call("get_paper_content", args)

    async def answer_pdf_query(self, url: str, query: str) -> dict[str, Any]:
        return await self.call("answer_pdf_queries", {"urls": [url], "queries": [query]})

    async def read_github(
        self, github_url: str, path: str = "/"
    ) -> dict[str, Any]:
        return await self.call(
            "read_files_from_github_repository",
            {"githubUrl": github_url, "path": path},
        )
