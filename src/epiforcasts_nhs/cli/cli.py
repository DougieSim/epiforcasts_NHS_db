from __future__ import annotations

import click

from epiforcasts_nhs.cli.daemon import main as daemon_cmd
from epiforcasts_nhs.cli.inference import main as inference_cmd
from epiforcasts_nhs.cli.launch import main as launch_cmd
from epiforcasts_nhs.cli.run_resilient import main as dashboard_cmd
from epiforcasts_nhs.core.cache import main as cache_cmd
from epiforcasts_nhs.data.generate import main as generate_cmd
from epiforcasts_nhs.data.generator import main as seed_cmd
from epiforcasts_nhs.ops.acceptance import main as acceptance_cmd
from epiforcasts_nhs.ops.covariate import main as covariate_cmd
from epiforcasts_nhs.ops.evidence import main as evidence_cmd
from epiforcasts_nhs.ops.feedback import main as feedback_cmd
from epiforcasts_nhs.ops.health import main as health_cmd


@click.group()
@click.version_option(package_name="epiforcasts-nhs", message="%(version)s")
def cli() -> None:
    """Probabilistic NHS Winter / System Pressure Early Warning toolkit."""


# ── Individual commands (mirror the epiforcasts-* console scripts) ────────────
cli.add_command(seed_cmd)        # epiforcasts seed
cli.add_command(generate_cmd)    # epiforcasts generate
cli.add_command(inference_cmd)   # epiforcasts inference
cli.add_command(daemon_cmd)      # epiforcasts daemon
cli.add_command(cache_cmd)       # epiforcasts cache
cli.add_command(health_cmd)      # epiforcasts health
cli.add_command(acceptance_cmd)  # epiforcasts acceptance
cli.add_command(evidence_cmd)    # epiforcasts evidence
cli.add_command(feedback_cmd)    # epiforcasts feedback
cli.add_command(covariate_cmd)   # epiforcasts covariate
cli.add_command(launch_cmd)      # epiforcasts launch
cli.add_command(dashboard_cmd)   # epiforcasts dashboard


# ── Composite workflows (formerly Pixi `run-full-pipeline` / `dev`) ───────────

def _invoke(ctx: click.Context, command: click.Command, **params) -> None:
    """
    Invoke a sibling command in-process and abort the composite if it fails.

    Individual commands signal failure by raising SystemExit with a non-zero
    code (and success with SystemExit(0) or by returning). We translate a
    non-zero exit into click.Abort so the composite stops cleanly.
    """
    try:
        ctx.invoke(command, **params)
    except SystemExit as exc:
        code = exc.code or 0
        if code != 0:
            click.echo(f"[ABORT] '{command.name}' exited with code {code}.", err=True)
            raise click.exceptions.Exit(code)


@cli.command(name="full-pipeline")
@click.pass_context
def full_pipeline(ctx: click.Context) -> None:
    """Seed the initial dataset, then run the daemon continuously (fast cycles)."""
    _invoke(ctx, seed_cmd)
    _invoke(ctx, daemon_cmd, interval_hours=0.01)


@cli.command(name="dev")
@click.pass_context
def dev(ctx: click.Context) -> None:
    """Dev loop: append a week, run one inference cycle, then launch the fast UI."""
    _invoke(ctx, generate_cmd)
    _invoke(ctx, daemon_cmd, once=True)
    _invoke(ctx, dashboard_cmd, fast=True)


if __name__ == "__main__":
    cli()
