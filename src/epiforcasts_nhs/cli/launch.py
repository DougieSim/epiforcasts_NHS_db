"""
Launch Streamlit with automatic port fallback.

Usage:
    epiforcasts-launch --app src/epiforcasts_nhs/dashboard/app.py --preferred-port 8501
    epiforcasts-launch --app src/epiforcasts_nhs/dashboard/app_fast.py --preferred-port 8501
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).parent.parent  # src/epiforcasts_nhs/

APP_FULL = str(_PACKAGE_ROOT / "dashboard" / "app.py")
APP_FAST = str(_PACKAGE_ROOT / "dashboard" / "app_fast.py")


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(preferred: int, max_port: int) -> int:
    for port in range(preferred, max_port + 1):
        if _is_port_free(port):
            return port
    raise RuntimeError(
        f"No free port found in range {preferred}-{max_port}. "
        "Stop an existing Streamlit process or expand the range."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Streamlit with safe port fallback.")
    parser.add_argument(
        "--app",
        default=APP_FULL,
        help="Streamlit app file to run (default: full dashboard).",
    )
    parser.add_argument("--preferred-port", type=int, default=8501, help="Preferred starting port.")
    parser.add_argument("--max-port", type=int, default=8510, help="Maximum port to try.")
    parser.add_argument(
        "--headless",
        choices=["true", "false"],
        default="true",
        help="Set Streamlit headless mode.",
    )
    args = parser.parse_args()

    try:
        port = _pick_port(args.preferred_port, args.max_port)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if port != args.preferred_port:
        print(
            f"[WARN] Preferred port {args.preferred_port} is busy. "
            f"Falling back to {port}."
        )
    else:
        print(f"[OK] Using preferred port {port}.")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        args.app,
        "--server.port",
        str(port),
        "--server.headless",
        args.headless,
    ]
    print(f"[OK] Starting: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
