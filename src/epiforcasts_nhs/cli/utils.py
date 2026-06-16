"""
CLI-layer utilities — port selection and subprocess helpers.

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
# Subprocess helpers
# ─────────────────────────────────────────

def venv_python() -> str:
    """
    Return the Python executable to use for subprocess launches.

    Prefers the project .venv on Windows; falls back to the current interpreter.
    """
    venv = Path(".venv") / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def run_streamlit(
    app_path: str,
    *,
    preferred_port: int,
    max_port: int,
    headless: str,
) -> int:
    """Launch Streamlit directly via cli.launch in the project interpreter."""
    cmd = [
        venv_python(),
        "-m", "epiforcasts_nhs.cli.launch",
        "--app", app_path,
        "--preferred-port", str(preferred_port),
        "--max-port", str(max_port),
        "--headless", headless,
    ]
    print(f"[LAUNCH] {' '.join(cmd)}")
    return subprocess.call(cmd)
