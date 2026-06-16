"""
CLI entry point — Streamlit launcher with automatic port fallback.

Usage:
    epiforcasts-launch
    epiforcasts-launch --app src/epiforcasts_nhs/dashboard/app_fast.py
    epiforcasts-launch --preferred-port 8502 --max-port 8520
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from epiforcasts_nhs.config import STREAMLIT_MAX_PORT, STREAMLIT_PREFERRED_PORT
from epiforcasts_nhs.cli.utils import pick_port

_PACKAGE_ROOT = Path(__file__).parent.parent
APP_FULL = str(_PACKAGE_ROOT / "dashboard" / "app.py")
APP_FAST = str(_PACKAGE_ROOT / "dashboard" / "app_fast.py")


@click.command(name="launch")
@click.option("--app", default=APP_FULL, show_default=False,
              help="Streamlit app path (defaults to the full dashboard).")
@click.option("--preferred-port", type=int, default=STREAMLIT_PREFERRED_PORT, show_default=True)
@click.option("--max-port",       type=int, default=STREAMLIT_MAX_PORT, show_default=True)
@click.option("--headless", type=click.Choice(["true", "false"]), default="true", show_default=True)
def main(app: str, preferred_port: int, max_port: int, headless: str) -> None:
    """Launch Streamlit with automatic port fallback."""
    try:
        port = pick_port(preferred_port, max_port)
    except RuntimeError as exc:
        click.echo(f"[FAIL] {exc}", err=True)
        raise SystemExit(1)

    if port != preferred_port:
        click.echo(f"[WARN] Port {preferred_port} busy — using {port}.")
    else:
        click.echo(f"[OK] Using port {port}.")

    cmd = [sys.executable, "-m", "streamlit", "run", app,
           "--server.port", str(port), "--server.headless", headless]
    click.echo(f"[OK] Starting: {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
