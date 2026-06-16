"""
Minimal acceptance checks for robustness evidence.

Usage:
    epiforcasts-acceptance
    epiforcasts-acceptance --posterior ci_posteriors.nc --metadata ci_posteriors_metadata.nc
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import click

from epiforcasts_nhs.config import (
    CI_METADATA_PATH,
    CI_MIN_DRAWS,
    CI_MIN_ICBS,
    CI_MIN_OBS,
    CI_POSTERIORS_PATH,
    GITIGNORE_ARTIFACT_ENTRIES,
    VALID_INFERENCE_METHODS,
)
from epiforcasts_nhs.ops.utils import check, fail

_LIFECYCLE_DOC  = Path("docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md")
_ROBUSTNESS_DOC = Path("docs/50-governance/ROBUSTNESS_TARGET_95.md")
_GITIGNORE      = Path(".gitignore")


@click.command(name="acceptance")
@click.option("--posterior", type=click.Path(path_type=Path), default=CI_POSTERIORS_PATH, show_default=True)
@click.option("--metadata",  type=click.Path(path_type=Path), default=CI_METADATA_PATH, show_default=True)
def main(posterior: Path, metadata: Path) -> None:
    """Run minimal robustness acceptance checks."""
    failures = 0

    failures += check(_LIFECYCLE_DOC.exists(),  "Lifecycle changelog document exists",     "Missing lifecycle changelog document")
    failures += check(_ROBUSTNESS_DOC.exists(), "Robustness target plan document exists",  "Missing robustness target plan document")

    if _LIFECYCLE_DOC.exists():
        failures += check(
            "Complete Commit Ledger" in _LIFECYCLE_DOC.read_text(encoding="utf-8"),
            "Lifecycle document contains commit ledger section",
            "Lifecycle document missing commit ledger section",
        )

    if _ROBUSTNESS_DOC.exists():
        failures += check(
            "Acceptance Checks by Dimension" in _ROBUSTNESS_DOC.read_text(encoding="utf-8"),
            "Robustness plan contains measurable acceptance section",
            "Robustness plan missing acceptance section",
        )

    failures += check(posterior.exists(), f"Posterior artifact exists: {posterior}", f"Posterior artifact missing: {posterior}")
    failures += check(metadata.exists(),  f"Metadata artifact exists: {metadata}",   f"Metadata artifact missing: {metadata}")

    if metadata.exists():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        failures += check(int(meta.get("n_icbs", 0)) >= CI_MIN_ICBS, "Metadata has at least one ICB",              "Metadata n_icbs is invalid")
        failures += check(int(meta.get("n_obs",  0)) >= CI_MIN_OBS,  "Metadata has expected observation count floor", "Metadata n_obs below expected floor")

    if posterior.exists():
        idata = az.from_netcdf(str(posterior))
        attrs = dict(idata.attrs or {})
        draws = int(idata.posterior.sizes.get("draw", 0))

        failures += check(draws >= CI_MIN_DRAWS, f"Posterior has sufficient draws ({draws})", f"Posterior draw count too low ({draws})")
        failures += check(
            str(attrs.get("inference_method", "")) in VALID_INFERENCE_METHODS,
            "Inference method metadata is valid",
            "Inference method metadata is missing/invalid",
        )
        failures += check("inference_seed" in attrs, "Inference seed metadata is present", "Inference seed metadata is missing")

    if _GITIGNORE.exists():
        ig_text = _GITIGNORE.read_text(encoding="utf-8")
        failures += check(
            all(entry in ig_text for entry in GITIGNORE_ARTIFACT_ENTRIES),
            "Artifact ignore policy entries are present",
            "Artifact ignore policy entries missing in .gitignore",
        )

    if failures:
        print(f"\nAcceptance checks failed: {failures}")
        raise SystemExit(1)
    print("\nAcceptance checks passed.")


if __name__ == "__main__":
    main()
