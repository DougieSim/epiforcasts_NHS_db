# `epiforcasts acceptance`

**Standalone script:** `epiforcasts-acceptance`

Run the **minimal robustness acceptance checks** — the governance/CI gate that
validates required documents, artifact presence, posterior draw counts,
inference-method metadata, and the artifact ignore policy.

## Usage

```bash
uv run epiforcasts acceptance
uv run epiforcasts acceptance --posterior ci_posteriors.nc --metadata ci_posteriors_metadata.nc
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--posterior` | path | `ci_posteriors.nc` | Posterior artifact to validate. |
| `--metadata` | path | `ci_posteriors_metadata.nc` | Metadata artifact to validate. |
| `--help` | flag | — | Show help and exit. |

## What it checks

- Lifecycle changelog and robustness-target documents exist with required sections.
- Posterior and metadata artifacts exist.
- Metadata floors: minimum ICB count and observation count.
- Posterior has at least the minimum draws; inference method + seed metadata present.
- `.gitignore` contains the required artifact-ignore entries.

## Exit code

- `0` — all checks passed.
- `1` — one or more checks failed (count printed).

## Related

- [`evidence`](evidence.md) — runs inference, health, and this acceptance gate together.
- [`health`](health.md) — lighter local check.
