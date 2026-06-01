"""
Run a standardized evidence cycle and append results to the evidence log.

Usage:
    epiforcasts-evidence --run-inference-fast
    epiforcasts-evidence
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


LOG_PATH = Path("docs/90-changelog/logs/EVIDENCE_RUN_LOG.md")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _git_short_hash() -> str:
    result = _run(["git", "rev-parse", "--short", "HEAD"])
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _append_entry(entry: str) -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing evidence log: {LOG_PATH}")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(entry.rstrip())
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute evidence checks and append run log entry.")
    parser.add_argument(
        "--run-inference-fast",
        action="store_true",
        help="Run fast inference before checks (recommended for independent evidence cycle).",
    )
    parser.add_argument("--draws", type=int, default=400, help="Draw count to log when fast inference runs.")
    parser.add_argument("--n-icbs", type=int, default=7, help="ICB count to log for synthetic setup.")
    parser.add_argument("--n-obs", type=int, default=1092, help="Observation count to log for synthetic setup.")
    args = parser.parse_args()

    start = dt.datetime.now(dt.timezone.utc)
    inference_cmd = "not run"

    if args.run_inference_fast:
        inference = [
            sys.executable, "-m", "epiforcasts_nhs.cli.inference",
            "--fast",
            "--advi-steps", "500",
            "--seed", "42",
            "--output-path", "ci_posteriors.nc",
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
            return result.returncode

    health = [sys.executable, "-m", "epiforcasts_nhs.ops.health"]
    print(f"[RUN] {' '.join(health)}")
    health_result = _run(health)
    if health_result.stdout:
        print(health_result.stdout, end="")
    if health_result.returncode != 0:
        if health_result.stderr:
            print(health_result.stderr, file=sys.stderr, end="")
        print("[FAIL] Health check failed; evidence log not updated.", file=sys.stderr)
        return health_result.returncode

    acceptance = [
        sys.executable, "-m", "epiforcasts_nhs.ops.acceptance",
        "--posterior", "ci_posteriors.nc",
        "--metadata", "ci_posteriors_metadata.nc",
    ]
    print(f"[RUN] {' '.join(acceptance)}")
    acceptance_result = _run(acceptance)
    if acceptance_result.stdout:
        print(acceptance_result.stdout, end="")
    if acceptance_result.returncode != 0:
        if acceptance_result.stderr:
            print(acceptance_result.stderr, file=sys.stderr, end="")
        print("[FAIL] Acceptance check failed; evidence log not updated.", file=sys.stderr)
        return acceptance_result.returncode

    end = dt.datetime.now(dt.timezone.utc)
    commit_hash = _git_short_hash()

    entry = f"""## {start.date()} (Automated independent cycle)\n\n- Date/time (UTC): {start.isoformat().replace('+00:00', 'Z')} to {end.isoformat().replace('+00:00', 'Z')}\n- Operator: automated local run\n- Commit hash: {commit_hash}\n- Environment: local Python execution\n- Inference command: `{inference_cmd}`\n- Health command: `{' '.join(health)}`\n- Acceptance command: `{' '.join(acceptance)}`\n- Result: PASS\n- Key metrics:\n  - draws: {args.draws}\n  - n_icbs: {args.n_icbs}\n  - n_obs: {args.n_obs}\n- Warnings observed: see terminal output\n- Failures observed: none\n- Follow-up actions: continue daily evidence accumulation and link utility feedback when available\n"""

    _append_entry(entry)
    print(f"[OK] Evidence entry appended to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
