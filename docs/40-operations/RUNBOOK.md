# Operations Runbook

## Audience

Developers, operators, and technical reviewers.

## Purpose

Provide one canonical operations guide for local development, offline inference, and dashboard serving.

## Scope

In scope:

1. local and development execution;
2. cache lifecycle and health checks;
3. offline inference workflows;
4. troubleshooting basics.

Out of scope:

1. regulated production deployment approval;
2. live NHS data integration.

## Core commands

The project uses [uv](https://docs.astral.sh/uv/). Every task is a subcommand of
the unified `epiforcasts` CLI (`uv run epiforcasts --help`).

```bash
uv sync
uv run epiforcasts dev
uv run epiforcasts daemon --once
uv run epiforcasts daemon
uv run epiforcasts dashboard
uv run epiforcasts dashboard --fast
uv run epiforcasts evidence --run-inference-fast
uv run epiforcasts cache status
uv run epiforcasts cache check
uv run epiforcasts full-pipeline
```

Dashboard startup protection:

1. `uv run epiforcasts dashboard` (and `--fast`) launch with automatic port fallback.
2. If `8501` is occupied, startup falls through `8502` to `8510` and continues.
3. The selected port is printed in terminal output.

Evidence confidence loop:

1. Run `uv run epiforcasts evidence --run-inference-fast` daily or per release candidate.
2. Capture at least one Trust/ICB utility session using `uv run epiforcasts feedback ...`.
3. Link resulting change actions in changelog and decision logs before claiming strict 9.5 confidence.

Need a minimal first-run path?

1. [FIRST_RUN_DUMMIES.md](FIRST_RUN_DUMMIES.md)

## Recommended daily workflow

1. Generate or refresh synthetic data.
2. Run offline inference once.
3. Warm cache and verify status.
4. Start full or fast dashboard.

## Fast path versus full path

| Path | Purpose | Typical cost | Typical users |
| --- | --- | --- | --- |
| app_fast.py + cache | Read-only communication and stakeholder walkthroughs. | Low runtime cost | Wider audiences |
| app.py + richer controls | Exploratory analysis and model communication detail. | Moderate runtime cost | Analysts and engineers |

## Cache lifecycle

1. Posteriors are produced offline.
2. CacheManager warms summary statistics.
3. UI reads pre-computed artefacts.
4. If cache invalid, UI should stop with clear remediation.

## Troubleshooting

1. Cache not ready:
Run `uv run epiforcasts daemon --once` then `uv run epiforcasts cache status`.
2. Slow inference:
Check compiler setup and use fast mode during iteration.
3. Confusing output language:
Verify terms against glossary and update explanatory copy.
4. Streamlit port already in use:
Use `uv run epiforcasts dashboard` (safe launcher). It auto-selects the next free port and prints it.
5. Inference save fails with file lock on `posteriors.nc`:
The script now retries and then saves to a timestamped `posteriors_YYYYMMDD_HHMMSS.nc` fallback instead of exiting.

## Related links

1. docs home: ../README.md
2. technical summary: ../30-model/TECHNICAL_SUMMARY_ADVANCED.md
3. governance overview: ../50-governance/GOVERNANCE_OVERVIEW.md
4. assumptions register: ../70-reference/assumptions-register.md
5. first-run quick path: FIRST_RUN_DUMMIES.md
