"""fidelis init — install + start the fidelis service so memory is "on" automatically.

Cross-platform:
- macOS: launchd plist at ~/Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist
- Linux: systemd user unit at ~/.config/systemd/user/fidelis-server.service
- Other: fallback to nohup (best-effort, no auto-start on reboot)

Idempotent: re-running install upgrades the unit in place. Uninstall removes
the unit cleanly + stops the service.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 19420
SERVICE_LABEL = "ai.hermeslabs.fidelis-server"

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{server_bin}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"""

SYSTEMD_TEMPLATE = """[Unit]
Description=Fidelis agent memory server
After=network.target

[Service]
ExecStart={server_bin}
Restart=on-failure
RestartSec=3
StandardOutput=append:{log_path}
StandardError=append:{log_path}
WorkingDirectory={working_dir}

[Install]
WantedBy=default.target
"""


def _server_bin() -> str:
    """Locate the fidelis-server entry point installed by pip."""
    bin_path = shutil.which("fidelis-server")
    if bin_path:
        return bin_path
    # Fallback: look in the same dir as the active python
    candidate = Path(sys.executable).parent / "fidelis-server"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(
        "fidelis-server entry point not found on PATH. "
        "Did you `pip install fidelis`? If running from source, "
        "`pip install -e .` to register the console script."
    )


def _health_check(timeout_s: float = 10.0) -> bool:
    """Wait up to timeout_s for /health to return ok."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def _install_macos(uninstall: bool = False) -> int:
    plist_path = Path.home() / "Library/LaunchAgents" / f"{SERVICE_LABEL}.plist"

    if uninstall:
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            plist_path.unlink()
            print(f"removed {plist_path}")
        else:
            print(f"no service installed at {plist_path}")
        return 0

    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    plist = PLIST_TEMPLATE.format(
        label=SERVICE_LABEL,
        server_bin=server_bin,
        working_dir=str(Path.home()),
        log_path=str(log_path),
    )
    # Backup existing plist before overwrite
    if plist_path.exists():
        backup = plist_path.with_suffix(f".plist.bak.{int(time.time())}")
        shutil.copy(plist_path, backup)
        print(f"backed up existing plist to {backup}")
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)

    plist_path.write_text(plist)
    print(f"wrote {plist_path}")
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"launchctl load failed: {result.stderr}", file=sys.stderr)
        return 1
    print(f"loaded service {SERVICE_LABEL}")
    return 0


def _install_linux(uninstall: bool = False) -> int:
    unit_path = Path.home() / ".config/systemd/user" / "fidelis-server.service"

    if uninstall:
        if unit_path.exists():
            subprocess.run(["systemctl", "--user", "stop", "fidelis-server.service"], check=False)
            subprocess.run(["systemctl", "--user", "disable", "fidelis-server.service"], check=False)
            unit_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            print(f"removed {unit_path}")
        else:
            print(f"no service installed at {unit_path}")
        return 0

    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    unit = SYSTEMD_TEMPLATE.format(
        server_bin=server_bin,
        working_dir=str(Path.home()),
        log_path=str(log_path),
    )
    if unit_path.exists():
        backup = unit_path.with_suffix(f".service.bak.{int(time.time())}")
        shutil.copy(unit_path, backup)
        print(f"backed up existing unit to {backup}")

    unit_path.write_text(unit)
    print(f"wrote {unit_path}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "fidelis-server.service"], check=True)
    subprocess.run(["systemctl", "--user", "start", "fidelis-server.service"], check=True)
    print("started fidelis-server.service")
    return 0


def _install_fallback(uninstall: bool = False) -> int:
    """nohup-based fallback for unsupported platforms. No auto-start on reboot."""
    if uninstall:
        # Best-effort: kill any running fidelis-server
        subprocess.run(["pkill", "-f", "fidelis-server"], check=False)
        print("attempted to stop any running fidelis-server (no auto-start was configured)")
        return 0

    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"WARNING: platform '{platform.system()}' has no auto-start support; "
          "starting under nohup. Will NOT survive reboot.")
    subprocess.Popen(
        [server_bin],
        stdout=open(log_path, "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return 0


def cmd_init(args) -> int:
    """Install + start fidelis-server as a system service.

    --uninstall: stop service + remove the unit/plist.
    """
    system = platform.system()

    if args.uninstall:
        if system == "Darwin":
            return _install_macos(uninstall=True)
        elif system == "Linux":
            return _install_linux(uninstall=True)
        else:
            return _install_fallback(uninstall=True)

    print(f"installing fidelis-server as a {system} service...")
    if system == "Darwin":
        rc = _install_macos()
    elif system == "Linux":
        rc = _install_linux()
    else:
        rc = _install_fallback()

    if rc != 0:
        return rc

    print("waiting for service to come up...")
    if _health_check(timeout_s=10.0):
        print(f"✓ fidelis-server is up at http://127.0.0.1:{PORT}")
        print(f"  log: {Path.home() / '.fidelis' / 'server.log'}")
        print()
        print("next steps:")
        print(f"  fidelis health                  # confirm")
        print(f"  fidelis watch ~/notes           # auto-ingest a directory")
        print(f"  fidelis mcp install             # wire up Claude Code")
        return 0
    else:
        print(f"✗ service installed but /health did not respond within 10s")
        print(f"  check log: {Path.home() / '.fidelis' / 'server.log'}")
        return 2
