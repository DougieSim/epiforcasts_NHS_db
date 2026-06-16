# Robustness Target Plan (9.3 to 9.5)

## Purpose

Define a minimal, measurable acceptance baseline to support a defensible 9.3 to 9.5 robustness posture.

## Target posture

1. Current practical posture: 8.8 to 9.1 depending on run history.
2. Target posture: 9.3 minimum with path to 9.5 after sustained repeated evidence.

## Acceptance Checks by Dimension

| Dimension | Minimal measurable check | Evidence source |
| --- | --- | --- |
| Reliability | Fast inference artifact produced and parseable with valid metadata and draw floor. | `python acceptance_check.py --posterior ci_posteriors.nc --metadata ci_posteriors_metadata.nc` |
| Reproducibility | Inference metadata includes deterministic seed and inference method. | `ci_posteriors.nc` attrs validated by `acceptance_check.py` |
| Operability | One-command health checks pass in local and CI context. | `python health_check.py`, CI smoke workflow |
| Observability | Health and inference status fields are emitted and validated in checks. | `health_check.py` output, `acceptance_check.py` validations |
| Maintainability | Core scripts compile and acceptance script compiles/runs in CI. | CI `py_compile` + acceptance step |
| Governance/Traceability | Lifecycle git ledger and robustness plan documents exist and contain required sections. | `docs/90-changelog/logs/LIFECYCLE_GIT_CHANGELOG.md`, this file |
| Developer Experience | Single commands exist for health and acceptance checks. | `uv run epiforcasts health`, `uv run epiforcasts acceptance` |
| CI/Release Readiness | CI runs syntax compile, health check, inference smoke, and acceptance gate. | `.github/workflows/smoke-check.yml` |
| Security Posture (scope) | Runtime artifacts are excluded by default in ignore policy. | `.gitignore` checks in `acceptance_check.py` |

## Gate decision rule

1. All acceptance checks must pass in CI.
2. No blocking failures in local `health_check.py`.
3. Two consecutive green CI runs are required to claim 9.3+ defensibly.
4. 9.5 claim requires sustained trend evidence beyond initial implementation window.
5. Minimum sustained evidence for a 9.5 claim: five consecutive green runs across at least three calendar days.
6. Time-series evidence entries should be logged in `docs/90-changelog/logs/EVIDENCE_RUN_LOG.md` using `docs/90-changelog/logs/EVIDENCE_LOG_TEMPLATE.md`.
7. At least one Trust/ICB utility feedback session must be logged in `docs/90-changelog/logs/TRUST_ICB_UTILITY_FEEDBACK_LOG.md` using `docs/90-changelog/logs/TRUST_ICB_UTILITY_FEEDBACK_TEMPLATE.md`.

## Commands

1. `uv run epiforcasts health`
2. `uv run epiforcasts acceptance`

## Last updated

2026-05-26
