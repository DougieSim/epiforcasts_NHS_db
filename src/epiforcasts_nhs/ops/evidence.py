"""
Run a standardized evidence cycle and append results to the evidence log.

Usage:
    epiforcasts-evidence --run-inference-fast
    epiforcasts-evidence
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import click

from epiforcasts_nhs.config import (
    CI_ADVI_STEPS,
    CI_METADATA_PATH,
    CI_POSTERIORS_PATH,
    DEFAULT_RANDOM_SEED,
)

LOG_PATH = Path("docs/90-changelog/logs/EVIDENCE_RUN_LOG.md")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _git_short_hash() -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"])
    return result.stdout.strip() or "unknown" if result.returncode == 0 else "unknown"


def _append_entry(entry: str) -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing evidence log: {LOG_PATH}")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(entry.rstrip())
        f.write("\n")


@click.command(name="evidence")
@click.option("--run-inference-fast", is_flag=True, help="Run a fast inference pass before the checks.")
@click.option("--draws",  type=int, default=400,  show_default=True)
@click.option("--n-icbs", type=int, default=7,    show_default=True)
@click.option("--n-obs",  type=int, default=1092, show_default=True)
def main(run_inference_fast: bool, draws: int, n_icbs: int, n_obs: int) -> None:
    """Execute evidence checks and append a run-log entry."""
    start = dt.datetime.now(dt.timezone.utc)
    inference_cmd = "not run"

    if run_inference_fast:
        inference = [
            sys.executable, "-m", "epiforcasts_nhs.cli.inference",
            "--fast",
            "--advi-steps", str(CI_ADVI_STEPS),
            "--seed",        str(DEFAULT_RANDOM_SEED),
            "--output-path", str(CI_POSTERIORS_PATH),
        ]
        inference_cmd = " ".join(inference)
        print(f"[RUN] {inference_cmd}")
        result = _run(inference)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            print("[FAIL] Inference step failed; evidence log not updated.", file=sys.stderr)
            raise SystemExit(result.returncode)

    health = [sys.executable, "-m", "epiforcasts_nhs.ops.health"]
    print(f"[RUN] {' '.join(health)}")
    health_result = _run(health)
    if health_result.stdout:
        print(health_result.stdout, end="")
    if health_result.returncode != 0:
        if health_result.stderr:
            print(health_result.stderr, file=sys.stderr, end="")
        print("[FAIL] Health check failed; evidence log not updated.", file=sys.stderr)
        raise SystemExit(health_result.returncode)

    acceptance = [
        sys.executable, "-m", "epiforcasts_nhs.ops.acceptance",
        "--posterior", str(CI_POSTERIORS_PATH),
        "--metadata",  str(CI_METADATA_PATH),
    ]
    print(f"[RUN] {' '.join(acceptance)}")
    acceptance_result = _run(acceptance)
    if acceptance_result.stdout:
        print(acceptance_result.stdout, end="")
    if acceptance_result.returncode != 0:
        if acceptance_result.stderr:
            print(acceptance_result.stderr, file=sys.stderr, end="")
        print("[FAIL] Acceptance check failed; evidence log not updated.", file=sys.stderr)
        raise SystemExit(acceptance_result.returncode)

    end = dt.datetime.now(dt.timezone.utc)
    commit_hash = _git_short_hash()

    entry = (
        f"## {start.date()} (Automated independent cycle)\n\n"
        f"- Date/time (UTC): {start.isoformat().replace('+00:00', 'Z')} to {end.isoformat().replace('+00:00', 'Z')}\n"
        f"- Operator: automated local run\n"
        f"- Commit hash: {commit_hash}\n"
        f"- Environment: local Python execution\n"
        f"- Inference command: `{inference_cmd}`\n"
        f"- Health command: `{' '.join(health)}`\n"
        f"- Acceptance command: `{' '.join(acceptance)}`\n"
        f"- Result: PASS\n"
        f"- Key metrics:\n"
        f"  - draws: {draws}\n"
        f"  - n_icbs: {n_icbs}\n"
        f"  - n_obs: {n_obs}\n"
        f"- Warnings observed: see terminal output\n"
        f"- Failures observed: none\n"
        f"- Follow-up actions: continue daily evidence accumulation\n"
    )

    _append_entry(entry)
    print(f"[OK] Evidence entry appended to {LOG_PATH}")


if __name__ == "__main__":
    main()
