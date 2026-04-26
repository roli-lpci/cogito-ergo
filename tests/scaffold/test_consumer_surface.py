"""Tests for v0.1 consumer-surface modules: init_cmd, watch_cmd, mcp_cmd, mcp_server, augment.

These tests do NOT spin up real services; they verify the CLI surface, file-write
behavior, and module importability. Live integration is gated by the manual
gates in PUBLISH-PLAN-20260425.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_init_cmd_imports():
    from fidelis import init_cmd
    assert hasattr(init_cmd, "cmd_init")
    assert init_cmd.SERVICE_LABEL == "ai.hermeslabs.fidelis-server"
    assert init_cmd.PORT == 19420


def test_watch_cmd_imports():
    from fidelis import watch_cmd
    assert hasattr(watch_cmd, "cmd_watch")
    assert hasattr(watch_cmd, "_file_hash")
    assert hasattr(watch_cmd, "_load_ledger")


def test_mcp_cmd_imports():
    from fidelis import mcp_cmd
    assert hasattr(mcp_cmd, "cmd_mcp_install")
    assert hasattr(mcp_cmd, "cmd_mcp_uninstall")
    assert mcp_cmd.MCP_SERVER_NAME == "fidelis"


def test_mcp_server_module_imports():
    from fidelis import mcp_server
    assert hasattr(mcp_server, "TOOLS")
    assert hasattr(mcp_server, "TOOL_HANDLERS")
    assert hasattr(mcp_server, "main")
    tool_names = {t["name"] for t in mcp_server.TOOLS}
    assert "fidelis_recall" in tool_names
    assert "fidelis_query" in tool_names
    assert "fidelis_health" in tool_names


def test_mcp_server_handles_tools_list():
    from fidelis import mcp_server
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    resp = mcp_server._handle(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "tools" in resp["result"]
    assert len(resp["result"]["tools"]) >= 3


def test_mcp_server_handles_initialize():
    from fidelis import mcp_server
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    resp = mcp_server._handle(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["serverInfo"]["name"] == "fidelis"


def test_mcp_server_unknown_method_returns_error():
    from fidelis import mcp_server
    req = {"jsonrpc": "2.0", "id": 1, "method": "no/such/method"}
    resp = mcp_server._handle(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_mcp_server_unknown_tool_returns_error():
    from fidelis import mcp_server
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}}
    resp = mcp_server._handle(req)
    assert "error" in resp
    assert "unknown tool" in resp["error"]["message"]


def test_mcp_install_writes_correct_entry(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_install
    settings_path = tmp_path / "settings.local.json"
    settings_path.write_text(json.dumps({"mcpServers": {}}))

    args = MagicMock()
    args.settings = str(settings_path)
    args.force = False
    rc = cmd_mcp_install(args)
    assert rc == 0

    settings = json.loads(settings_path.read_text())
    assert "mcpServers" in settings
    assert "fidelis" in settings["mcpServers"]
    entry = settings["mcpServers"]["fidelis"]
    assert "command" in entry
    assert "args" in entry
    assert "mcp_server.py" in entry["args"][0]


def test_mcp_install_idempotent_recognizes_self(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_install
    settings_path = tmp_path / "settings.local.json"
    settings_path.write_text(json.dumps({"mcpServers": {}}))

    args = MagicMock()
    args.settings = str(settings_path)
    args.force = False

    # Install twice
    rc1 = cmd_mcp_install(args)
    rc2 = cmd_mcp_install(args)
    assert rc1 == 0
    assert rc2 == 0  # second install should recognize self + succeed


def test_mcp_install_refuses_non_fidelis_entry(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_install
    settings_path = tmp_path / "settings.local.json"
    settings_path.write_text(json.dumps({
        "mcpServers": {"fidelis": {"command": "/usr/bin/some-other-tool", "args": ["--arg"]}}
    }))

    args = MagicMock()
    args.settings = str(settings_path)
    args.force = False
    rc = cmd_mcp_install(args)
    assert rc == 1  # refuses to overwrite


def test_mcp_install_force_overwrites(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_install
    settings_path = tmp_path / "settings.local.json"
    settings_path.write_text(json.dumps({
        "mcpServers": {"fidelis": {"command": "/usr/bin/some-other-tool", "args": ["--arg"]}}
    }))

    args = MagicMock()
    args.settings = str(settings_path)
    args.force = True
    rc = cmd_mcp_install(args)
    assert rc == 0


def test_mcp_uninstall_removes_entry(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_install, cmd_mcp_uninstall
    settings_path = tmp_path / "settings.local.json"
    settings_path.write_text(json.dumps({"mcpServers": {}}))

    args_install = MagicMock(); args_install.settings = str(settings_path); args_install.force = False
    cmd_mcp_install(args_install)

    args_uninstall = MagicMock(); args_uninstall.settings = str(settings_path)
    rc = cmd_mcp_uninstall(args_uninstall)
    assert rc == 0

    settings = json.loads(settings_path.read_text())
    assert "fidelis" not in settings.get("mcpServers", {})


def test_mcp_uninstall_handles_missing_settings(tmp_path):
    from fidelis.mcp_cmd import cmd_mcp_uninstall
    settings_path = tmp_path / "no-such-file.json"
    args = MagicMock(); args.settings = str(settings_path)
    rc = cmd_mcp_uninstall(args)
    assert rc == 0  # graceful no-op


def test_augment_imports():
    from fidelis import augment as aug
    assert hasattr(aug, "augment")
    assert callable(aug.augment)


def test_augment_with_mocked_recall_and_llm(tmp_path):
    """End-to-end: augment should call recall, wrap with scaffold, invoke LLM, return text."""
    from fidelis import augment as aug

    captured = {}

    def fake_llm(system, user_msg):
        captured["system"] = system
        captured["user_msg"] = user_msg
        return "FAKE_LLM_RESPONSE"

    with patch.object(aug, "_recall", return_value=("retrieved memory chunk", 0.85)):
        out = aug.augment(
            question="What did the user say?",
            qtype="single-session-user",
            llm_call=fake_llm,
        )
    assert out == "FAKE_LLM_RESPONSE"
    # system prompt must contain the scaffold markers
    assert "[FIDELIS-SCAFFOLD-" in captured["system"]
    # user msg must contain retrieved context + question
    assert "retrieved memory chunk" in captured["user_msg"]
    assert "What did the user say?" in captured["user_msg"]


def test_augment_propagates_recall_failure():
    from fidelis import augment as aug

    def fake_llm(system, user_msg):
        return "should not be called"

    with patch.object(aug, "_recall", side_effect=RuntimeError("server down")):
        with pytest.raises(RuntimeError, match="server down"):
            aug.augment(question="q", qtype="single-session-user", llm_call=fake_llm)


def test_watch_file_hash_stable(tmp_path):
    from fidelis.watch_cmd import _file_hash
    f = tmp_path / "x.md"
    f.write_text("hello world")
    h1 = _file_hash(f)
    h2 = _file_hash(f)
    assert h1 == h2
    f.write_text("hello world!")
    h3 = _file_hash(f)
    assert h1 != h3


def test_watch_ledger_roundtrip(tmp_path, monkeypatch):
    from fidelis import watch_cmd
    ledger_path = tmp_path / "watched.json"
    monkeypatch.setattr(watch_cmd, "LEDGER_PATH", ledger_path)
    assert watch_cmd._load_ledger() == {}
    watch_cmd._save_ledger({"a": "h1", "b": "h2"})
    assert watch_cmd._load_ledger() == {"a": "h1", "b": "h2"}


def test_watch_scan_files_respects_max(tmp_path):
    from fidelis.watch_cmd import _scan_files
    for i in range(20):
        (tmp_path / f"f{i}.md").write_text("x")
    found = _scan_files(tmp_path, ("*.md",), max_files=10)
    assert len(found) == 10


def test_init_cmd_locates_server_bin():
    """Server binary should be on PATH after pip install -e ."""
    from fidelis.init_cmd import _server_bin
    bin_path = _server_bin()
    assert "fidelis-server" in bin_path
