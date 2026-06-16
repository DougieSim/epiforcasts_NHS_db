from __future__ import annotations

from pathlib import Path

import click

from epiforcasts_nhs.config import STREAMLIT_MAX_PORT, STREAMLIT_PREFERRED_PORT
from epiforcasts_nhs.cli.utils import run_streamlit

_PACKAGE_ROOT = Path(__file__).parent.parent


@click.command(name="dashboard")
@click.option("--fast", is_flag=True, help="Launch the lightweight fast dashboard (app_fast.py).")
@click.option("--preferred-port", type=int, default=STREAMLIT_PREFERRED_PORT, show_default=True)
@click.option("--max-port",       type=int, default=STREAMLIT_MAX_PORT, show_default=True)
@click.option("--headless", type=click.Choice(["true", "false"]), default="true", show_default=True)
def main(fast: bool, preferred_port: int, max_port: int, headless: str) -> None:
    """Run the Streamlit dashboard with automatic port fallback."""
    app_path = str(_PACKAGE_ROOT / "dashboard" / ("app_fast.py" if fast else "app.py"))
    raise SystemExit(
        run_streamlit(
            app_path,
            preferred_port=preferred_port,
            max_port=max_port,
            headless=headless,
        )
    )


if __name__ == "__main__":
    main()
