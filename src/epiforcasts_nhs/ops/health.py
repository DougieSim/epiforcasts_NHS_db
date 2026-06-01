"""
One-command local health checks for the NHS pressure demo.

Usage:
    epiforcasts-health
    python -m epiforcasts_nhs.ops.health
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.ops.utils import fail, ok, warn


def main() -> int:
    failures = 0

    ok(f"Python executable: {sys.executable}")
    if shutil.which("g++") or shutil.which("clang++") or shutil.which("cl"):
        ok("C/C++ compiler detected for PyTensor acceleration")
    else:
        warn("No C/C++ compiler detected; inference may be significantly slower")

    data_path = Path("synthetic_nhs_pressure.csv")
    if data_path.exists():
        ok(f"Data file present: {data_path}")
    else:
        fail(f"Missing data file: {data_path}")
        failures += 1

    posterior_path = Path("posteriors.nc")
    if posterior_path.exists():
        ok(f"Posterior artifact present: {posterior_path}")
    else:
        warn("Posterior artifact missing: run `epiforcasts-inference --fast`")

    cache = CacheManager(posteriors_path=posterior_path)
    if cache.is_valid():
        ok("Cache is valid and ready for dashboards")

        health = cache.get_health_check()
        if health.get("status") != "not_warmed":
            ok(f"Cache health metadata loaded (n_icbs={health.get('n_icbs')}, n_draws={health.get('n_draws')})")
        else:
            warn("Cache health metadata not found")

        try:
            with open(cache.summary_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            ok(f"Summary stats readable: {cache.summary_stats_path} ({len(stats)} areas)")
        except Exception as exc:
            fail(f"Could not read summary stats: {exc}")
            failures += 1
    else:
        warn("Cache is not valid yet. Run inference then warm cache.")

    if failures:
        print(f"\nHealth check failed with {failures} blocking issue(s).")
        return 1

    print("\nHealth check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
