"""
CLI entry point — Streamlit launcher with automatic port fallback.

Parses arguments and delegates to cli.utils for port selection.

Usage:
    epiforcasts-launch
    epiforcasts-launch --app src/epiforcasts_nhs/dashboard/app_fast.py
    epiforcasts-launch --preferred-port 8502 --max-port 8520
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from epiforcasts_nhs.cli.utils import pick_port

_PACKAGE_ROOT = Path(__file__).parent.parent  # src/epiforcasts_nhs/
APP_FULL = str(_PACKAGE_ROOT / "dashboard" / "app.py")
APP_FAST = str(_PACKAGE_ROOT / "dashboard" / "app_fast.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Streamlit with safe port fallback.")
    parser.add_argument("--app",            default=APP_FULL,
                        help="Streamlit app file (default: full dashboard).")
    parser.add_argument("--preferred-port", type=int, default=8501)
    parser.add_argument("--max-port",       type=int, default=8510)
    parser.add_argument("--headless",       choices=["true", "false"], default="true")
    args = parser.parse_args()

    try:
        port = pick_port(args.preferred_port, args.max_port)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if port != args.preferred_port:
        print(f"[WARN] Port {args.preferred_port} busy — using {port}.")
    else:
        print(f"[OK] Using port {port}.")

    cmd = [sys.executable, "-m", "streamlit", "run", args.app,
           "--server.port", str(port), "--server.headless", args.headless]
    print(f"[OK] Starting: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
