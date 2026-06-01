"""
CLI entry point — resilient dashboard launcher.

Tries Pixi first; falls back to direct Python invocation if Pixi fails.

Usage:
    epiforcasts-resilient --app full
    epiforcasts-resilient --app fast --preferred-port 8502
    epiforcasts-resilient --app full --fallback-on-any-failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epiforcasts_nhs.config import STREAMLIT_MAX_PORT, STREAMLIT_PREFERRED_PORT
from epiforcasts_nhs.cli.utils import (
    pixi_task_for_app,
    run_pixi_task,
    run_streamlit_fallback,
    should_fallback_from_output,
)

_PACKAGE_ROOT = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dashboard with Pixi-first fallback.")
    parser.add_argument("--app",      choices=["full", "fast"], default="full")
    parser.add_argument("--preferred-port", type=int, default=STREAMLIT_PREFERRED_PORT)
    parser.add_argument("--max-port",       type=int, default=STREAMLIT_MAX_PORT)
    parser.add_argument("--headless",       choices=["true", "false"], default="true")
    parser.add_argument("--fallback-on-any-failure", action="store_true")
    args = parser.parse_args()

    task = pixi_task_for_app(args.app)
    print(f"[INFO] Trying Pixi: pixi run {task}")
    result = run_pixi_task(task)

    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        return 0

    combined = f"{result.stdout}\n{result.stderr}".strip()
    print("[WARN] Pixi launch failed.")
    if combined:
        print(combined)

    if args.fallback_on_any_failure or should_fallback_from_output(combined):
        print("[INFO] Falling back to direct launcher.")
        app_path = str(_PACKAGE_ROOT / "dashboard" / ("app_fast.py" if args.app == "fast" else "app.py"))
        return run_streamlit_fallback(
            app_path,
            preferred_port=args.preferred_port,
            max_port=args.max_port,
            headless=args.headless,
        )

    print("[FAIL] Pixi failed and fallback criteria not met.", file=sys.stderr)
    print("[HINT] Re-run with --fallback-on-any-failure to force fallback.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
