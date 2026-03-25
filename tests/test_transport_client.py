import asyncio
import sys
from unittest.mock import AsyncMock, Mock

import grpc
import pytest

import truffile
from truffile.transport.client import (
    GRPC_MAX_MESSAGE_BYTES,
    TruffleClient,
    _is_retryable_finish_error,
)


class _ReadyChannel:
    async def channel_ready(self) -> None:
        return None


def test_connect_sets_grpc_message_size_limits(monkeypatch):
    captured = {}
    fake_channel = _ReadyChannel()

    def fake_insecure_channel(address, options=None):
        captured["address"] = address
        captured["options"] = options
        return fake_channel

    monkeypatch.setattr("truffile.transport.client.aio.insecure_channel", fake_insecure_channel)
    monkeypatch.setattr("truffile.transport.client.TruffleOSStub", Mock(return_value="stub"))

    client = TruffleClient("127.0.0.1:80", token="token")
    asyncio.run(client.connect())

    assert client.channel is fake_channel
    assert client.stub == "stub"
    assert captured["address"] == "127.0.0.1:80"
    assert captured["options"] == [
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
    ]


def test_init_prepends_repo_root_for_bundled_truffle(monkeypatch):
    repo_root = "/Users/truffle/work/truffile"
    monkeypatch.setattr(sys, "path", ["/tmp/external"])

    truffile._ensure_bundled_truffle_on_path()

    assert sys.path[0] == repo_root


# ---------------------------------------------------------------------------
# finish_app retry logic
# ---------------------------------------------------------------------------

_FG_PAYLOAD = {"cmd": "python", "args": ["-m", "app"], "cwd": "/", "env": []}


def _make_client() -> TruffleClient:
    """Create a TruffleClient wired up with a mock stub for testing."""
    client = TruffleClient("127.0.0.1:80", token="tok")
    client.stub = Mock()
    client.app_uuid = "test-uuid"
    client.access_path = "test-path"
    return client


def _make_resp(error: str | None = None, details: str = ""):
    """Build a fake FinishBuildSessionResponse."""
    resp = Mock()
    if error is not None:
        resp.HasField = lambda f: f == "error"
        resp.error.error = error
        resp.error.details = details
    else:
        resp.HasField = lambda _: False
    return resp


def test_finish_app_retries_on_taskgroup_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _make_client()

    fail_resp = _make_resp(
        error="failed to get tools from builder foreground app: unhandled errors in a TaskGroup (1 sub-exception)",
        details="RuntimeError(...)",
    )
    ok_resp = _make_resp()

    client.stub.Builder_FinishBuildSession = AsyncMock(side_effect=[fail_resp, ok_resp])

    result = asyncio.run(
        client.finish_app(
            name="App",
            bundle_id=None,
            description="",
            icon=None,
            foreground=_FG_PAYLOAD,
            background=None,
            default_schedule=None,
        )
    )

    assert result is ok_resp
    assert client.stub.Builder_FinishBuildSession.call_count == 2
    assert client.app_uuid is None  # cleared on success


def test_finish_app_no_retry_on_permanent_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _make_client()

    fail_resp = _make_resp(error="invalid bundle_id", details="not found")
    client.stub.Builder_FinishBuildSession = AsyncMock(return_value=fail_resp)

    with pytest.raises(RuntimeError, match="finish failed: invalid bundle_id"):
        asyncio.run(
            client.finish_app(
                name="App",
                bundle_id=None,
                description="",
                icon=None,
                foreground=_FG_PAYLOAD,
                background=None,
                default_schedule=None,
            )
        )

    assert client.stub.Builder_FinishBuildSession.call_count == 1


def test_finish_app_exhausts_retries(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _make_client()

    fail_resp = _make_resp(
        error="failed to get tools from builder foreground app: TaskGroup",
        details="timeout",
    )
    client.stub.Builder_FinishBuildSession = AsyncMock(return_value=fail_resp)

    with pytest.raises(RuntimeError, match="finish failed:"):
        asyncio.run(
            client.finish_app(
                name="App",
                bundle_id=None,
                description="",
                icon=None,
                foreground=_FG_PAYLOAD,
                background=None,
                default_schedule=None,
            )
        )

    # all 3 attempts were made before giving up
    assert client.stub.Builder_FinishBuildSession.call_count == 3


def test_finish_app_preserves_session_on_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    client = _make_client()

    fail_resp = _make_resp(error="some permanent error", details="")
    client.stub.Builder_FinishBuildSession = AsyncMock(return_value=fail_resp)

    with pytest.raises(RuntimeError):
        asyncio.run(
            client.finish_app(
                name="App",
                bundle_id=None,
                description="",
                icon=None,
                foreground=_FG_PAYLOAD,
                background=None,
                default_schedule=None,
            )
        )

    # app_uuid preserved so cli.py error handler can call discard()
    assert client.app_uuid == "test-uuid"
    assert client.access_path == "test-path"
