"""Blocker #3 — Prompt-cache wire format.

Verifies that the Fidelis scaffold integrates correctly with Anthropic's
prompt-caching API contract using httpx.MockTransport (no real API credits).

What this tests:
  - The scaffold's open marker is at the START of the system text.
  - When the caller wraps system as a list of TextBlockParams with
    cache_control: {"type": "ephemeral"}, the SDK serialises that correctly
    into the HTTP request body.
  - The caller's code can read cache_read_input_tokens / cache_creation_input_tokens
    from the response usage block.

What this does NOT test (MockTransport limitation):
  - That Anthropic's servers actually populate the cache (requires real credits
    + two sequential identical calls with <5 min gap).
  - Actual cache-hit latency reduction.
  - Whether the 5-min TTL vs 1-hr TTL behaves differently at the server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import anthropic
from anthropic.types import TextBlockParam

from fidelis.scaffold import SCAFFOLD_OPEN, wrap_system_prompt

# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------

_CAPTURED_REQUESTS: list[dict[str, Any]] = []


def _make_cache_response(
    cache_read_tokens: int = 1500,
    cache_creation_tokens: int = 0,
) -> httpx.Response:
    """Return a minimal well-formed Anthropic Messages API response."""
    body = {
        "id": "msg_mock_cache_001",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Answer: mocked response."}],
        "model": "claude-3-5-haiku-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 10,
            "cache_creation_input_tokens": cache_creation_tokens,
            "cache_read_input_tokens": cache_read_tokens,
        },
    }
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
    )


def _handler_capturing(request: httpx.Request) -> httpx.Response:
    """Intercept every SDK HTTP call; record the parsed body."""
    body = json.loads(request.content)
    _CAPTURED_REQUESTS.append(body)
    return _make_cache_response(cache_read_tokens=1500, cache_creation_tokens=0)


@pytest.fixture(autouse=True)
def _clear_captured():
    _CAPTURED_REQUESTS.clear()
    yield
    _CAPTURED_REQUESTS.clear()


def _make_client() -> anthropic.Anthropic:
    transport = httpx.MockTransport(_handler_capturing)
    http = httpx.Client(transport=transport)
    return anthropic.Anthropic(api_key="test-key-mock", http_client=http)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScaffoldOpenMarkerPosition:
    """Blocker #3a — scaffold open marker is the FIRST token of system content."""

    def test_wrap_system_prompt_starts_with_open_marker(self):
        """Pure unit test — no API call needed."""
        for qtype in [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "knowledge-update",
            "multi-session",
            "temporal-reasoning",
        ]:
            prompt = wrap_system_prompt(qtype)
            assert prompt.startswith(SCAFFOLD_OPEN), (
                f"qtype={qtype!r}: system prompt does not start with {SCAFFOLD_OPEN!r}. "
                f"Actual start: {prompt[:60]!r}"
            )

    def test_scaffold_open_marker_is_stable_string(self):
        assert SCAFFOLD_OPEN == "[FIDELIS-SCAFFOLD-v0.1.0]"


class TestCacheControlWireFormat:
    """Blocker #3b — SDK serialises cache_control block correctly in HTTP body."""

    def test_system_list_with_cache_control_is_in_request_body(self):
        """
        Call client.messages.create with system as a list of TextBlockParam,
        one of which carries cache_control: {"type": "ephemeral"}.
        Assert the serialised HTTP body contains that structure.
        """
        client = _make_client()
        scaffold_text = wrap_system_prompt("single-session-user")

        system_block: TextBlockParam = {
            "type": "text",
            "text": scaffold_text,
            "cache_control": {"type": "ephemeral"},
        }

        client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=[system_block],  # list form required for cache_control
            messages=[{"role": "user", "content": "What did I say about coffee?"}],
        )

        assert len(_CAPTURED_REQUESTS) == 1
        body = _CAPTURED_REQUESTS[0]

        # SDK must have serialised system as a list of blocks
        assert "system" in body, "Request body missing 'system' field"
        system_field = body["system"]
        assert isinstance(system_field, list), (
            f"Expected system to be a list of blocks, got {type(system_field)}"
        )
        assert len(system_field) == 1
        block = system_field[0]

        assert block.get("type") == "text"
        assert "cache_control" in block, (
            "cache_control was stripped from the serialised request body"
        )
        assert block["cache_control"]["type"] == "ephemeral", (
            f"Expected ephemeral cache_control, got {block['cache_control']!r}"
        )

    def test_scaffold_text_is_intact_in_wire_system_block(self):
        """The scaffold text sent to the API must start with SCAFFOLD_OPEN."""
        client = _make_client()
        scaffold_text = wrap_system_prompt("multi-session")

        system_block: TextBlockParam = {
            "type": "text",
            "text": scaffold_text,
            "cache_control": {"type": "ephemeral"},
        }

        client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=[system_block],
            messages=[{"role": "user", "content": "Combine all sessions."}],
        )

        assert len(_CAPTURED_REQUESTS) == 1
        wire_text = _CAPTURED_REQUESTS[0]["system"][0]["text"]
        assert wire_text.startswith(SCAFFOLD_OPEN), (
            f"Scaffold text in wire body does not start with open marker. "
            f"Got: {wire_text[:80]!r}"
        )

    def test_system_string_form_has_no_cache_control(self):
        """When system is a plain string (no caching), no cache_control key appears."""
        client = _make_client()
        scaffold_text = wrap_system_prompt("knowledge-update")

        client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=scaffold_text,  # plain string — caching NOT requested
            messages=[{"role": "user", "content": "What changed?"}],
        )

        assert len(_CAPTURED_REQUESTS) == 1
        body = _CAPTURED_REQUESTS[0]
        # Plain string system stays as string (no list wrapping, no cache_control)
        assert isinstance(body["system"], str), (
            "Expected plain string system to remain a string in wire body"
        )


class TestCacheResponseFieldsReadable:
    """Blocker #3c — caller code can read cache usage fields from the response."""

    def test_cache_read_input_tokens_accessible(self):
        client = _make_client()
        scaffold_text = wrap_system_prompt("single-session-user")
        system_block: TextBlockParam = {
            "type": "text",
            "text": scaffold_text,
            "cache_control": {"type": "ephemeral"},
        }

        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=[system_block],
            messages=[{"role": "user", "content": "Test."}],
        )

        # The response usage object must expose cache fields
        assert hasattr(resp.usage, "cache_read_input_tokens"), (
            "resp.usage.cache_read_input_tokens not present — "
            "caller code cannot read cache hit count"
        )
        assert resp.usage.cache_read_input_tokens == 1500, (
            f"Expected 1500 cache_read_input_tokens from mock, "
            f"got {resp.usage.cache_read_input_tokens}"
        )

    def test_cache_creation_input_tokens_accessible(self):
        client = _make_client()
        scaffold_text = wrap_system_prompt("temporal-reasoning")
        system_block: TextBlockParam = {
            "type": "text",
            "text": scaffold_text,
            "cache_control": {"type": "ephemeral"},
        }

        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=[system_block],
            messages=[{"role": "user", "content": "Test."}],
        )

        assert hasattr(resp.usage, "cache_creation_input_tokens")
        assert resp.usage.cache_creation_input_tokens == 0

    def test_zero_cache_fields_when_string_system(self):
        """When system is plain string, mock returns same body; fields still readable."""
        client = _make_client()
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=wrap_system_prompt("knowledge-update"),
            messages=[{"role": "user", "content": "Test."}],
        )
        # Mock always returns 1500 cache_read; the point is the field is accessible
        assert resp.usage.cache_read_input_tokens is not None
