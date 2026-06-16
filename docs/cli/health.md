# `epiforcasts health`

**Standalone script:** `epiforcasts-health`

One-command **local health check**. Verifies the environment and artifacts are
in a runnable state before you demo or run inference — a quick "is everything
wired up?" gate.

## Usage

```bash
uv run epiforcasts health
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--help` | flag | — | Show help and exit. |

Takes no configuration; it inspects the standard project paths.

## What it checks

- Python interpreter in use.
- Presence of a C/C++ compiler (PyTensor acceleration — a warning, not a failure).
- Data file present (`synthetic_nhs_pressure.csv`).
- Posterior artifact present (`posteriors.nc`).
- Cache validity and readable summary stats.

## Exit code

- `0` — all blocking checks passed (warnings allowed).
- `1` — one or more blocking issues (e.g. missing data file or unreadable cache).

## Related

- [`acceptance`](acceptance.md) — stricter, artifact-level checks for CI.
