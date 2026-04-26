"""fidelis watch — auto-ingest new markdown/text files from a directory.

Behavior:
- Initial scan: ingests all matching files (capped by --max-files)
- Continuous: polls for new/modified files every --interval seconds
- Idempotent: tracks ingested file hashes in ~/.fidelis/watched.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_GLOB_PATTERNS = ("*.md", "*.txt")
DEFAULT_MAX_FILES = 500
DEFAULT_INTERVAL_S = 5.0
LEDGER_PATH = Path.home() / ".fidelis" / "watched.json"


def _server_url() -> str:
    port = os.environ.get("FIDELIS_PORT", os.environ.get("COGITO_PORT", "19420"))
    return f"http://127.0.0.1:{port}"


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict | None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_server_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"  [warn] {path} request failed: {e}", file=sys.stderr)
        return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2))


def _scan_files(root: Path, patterns: tuple, max_files: int) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        for f in root.rglob(pat):
            if f.is_file():
                found.append(f)
                if len(found) >= max_files:
                    return found
    return found


def _ingest_file(path: Path, verbose: bool) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  [skip] {path}: {e}", file=sys.stderr)
        return False
    if not text.strip():
        return False
    res = _post("/store", {"text": text, "metadata": {"source_path": str(path)}})
    if res and verbose:
        print(f"  [ingest] {path}")
    return res is not None


def cmd_watch(args) -> int:
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    patterns = tuple(args.glob) if args.glob else DEFAULT_GLOB_PATTERNS
    max_files = args.max_files
    interval = args.interval
    verbose = args.verbose

    ledger = _load_ledger()
    print(f"watching {root} (patterns: {','.join(patterns)}, max-files: {max_files})")

    # Initial scan
    initial = _scan_files(root, patterns, max_files)
    print(f"initial scan: {len(initial)} files")
    for f in initial:
        h = _file_hash(f)
        if ledger.get(str(f)) == h:
            continue
        if _ingest_file(f, verbose):
            ledger[str(f)] = h
    _save_ledger(ledger)

    if args.once:
        print(f"--once flag set, exiting after initial scan ({len(initial)} files processed)")
        return 0

    # Continuous polling
    print(f"polling every {interval}s (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(interval)
            current = _scan_files(root, patterns, max_files)
            new_or_changed = 0
            for f in current:
                h = _file_hash(f)
                if ledger.get(str(f)) == h:
                    continue
                if _ingest_file(f, verbose):
                    ledger[str(f)] = h
                    new_or_changed += 1
            if new_or_changed:
                print(f"[+{new_or_changed}] ingested new/changed files")
                _save_ledger(ledger)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0
