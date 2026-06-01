"""
CLI-layer utilities — port selection and subprocess / Pixi helpers.

These functions are inherently CLI-adjacent (they deal with OS processes,
sockets, and shell invocation) and are not part of the library API. They
are shared between cli.launch and cli.run_resilient.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────
# Port selection
# ─────────────────────────────────────────

def is_port_free(port: int) -> bool:
    """Return True if the given localhost port is not currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def pick_port(preferred: int, max_port: int) -> int:
    """
    Return the first free port in [preferred, max_port].

    Raises RuntimeError if no port is available in the range.
    """
    for port in range(preferred, max_port + 1):
        if is_port_free(port):
            return port
    raise RuntimeError(
        f"No free port found in range {preferred}–{max_port}. "
        "Stop an existing Streamlit process or expand the range."
    )


# ─────────────────────────────────────────
# Pixi / subprocess helpers
# ─────────────────────────────────────────

# Substrings in Pixi output that indicate a certificate / solver failure worth
# retrying via the direct Python fallback path.
_PIXI_FALLBACK_HINTS = ("unknownissuer", "certificate", "ssl", "solver")


def pixi_task_for_app(app: str) -> str:
    """Map 'fast' | 'full' to the corresponding Pixi task name."""
    return "run-dashboard-fast" if app == "fast" else "run-dashboard"


def should_fallback_from_output(output: str) -> bool:
    """Return True if Pixi output contains a known cert / solver failure hint."""
    text = output.lower()
    return any(token in text for token in _PIXI_FALLBACK_HINTS)


def run_pixi_task(task: str) -> subprocess.CompletedProcess[str]:
    """Run a named Pixi task, capturing stdout/stderr."""
    return subprocess.run(
        ["pixi", "run", task],
        text=True,
        capture_output=True,
        check=False,
    )


def fallback_python() -> str:
    """
    Return the Python executable to use when Pixi is unavailable.

    Prefers the project .venv on Windows; falls back to the current interpreter.
    """
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    return str(venv_python) if venv_python.exists() else sys.executable


def run_streamlit_fallback(
    app_path: str,
    *,
    preferred_port: int,
    max_port: int,
    headless: str,
) -> int:
    """Launch Streamlit directly via cli.launch, bypassing Pixi."""
    cmd = [
        fallback_python(),
        "-m", "epiforcasts_nhs.cli.launch",
        "--app", app_path,
        "--preferred-port", str(preferred_port),
        "--max-port", str(max_port),
        "--headless", headless,
    ]
    print(f"[FALLBACK] {' '.join(cmd)}")
    return subprocess.call(cmd)
