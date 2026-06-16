# CLI reference

All functionality is exposed through the unified **`epiforcasts`** click CLI.
Run any command with `uv run epiforcasts <command> [OPTIONS]`, or use the
equivalent standalone `epiforcasts-*` console script.

```bash
uv run epiforcasts --help            # list all commands
uv run epiforcasts <command> --help  # options for one command
```

Each command also has its own page below documenting its purpose, options,
defaults, and examples.

## Commands

| Command | Standalone script | Purpose |
| --- | --- | --- |
| [`seed`](seed.md) | `epiforcasts-create-initial-data` | Build the initial synthetic dataset (run once). |
| [`generate`](generate.md) | `epiforcasts-generate` | Append new synthetic week(s) to the rolling panel. |
| [`inference`](inference.md) | `epiforcasts-inference` | Run offline Bayesian inference and save posteriors. |
| [`daemon`](daemon.md) | `epiforcasts-daemon` | Continuous / single-cycle generate → infer → warm-cache loop. |
| [`cache`](cache.md) | `epiforcasts-cache` | Warm, check, inspect, or clear the dashboard cache. |
| [`health`](health.md) | `epiforcasts-health` | One-command local health check. |
| [`acceptance`](acceptance.md) | `epiforcasts-acceptance` | Minimal robustness acceptance gate (CI). |
| [`evidence`](evidence.md) | `epiforcasts-evidence` | Run an evidence cycle and append a run-log entry. |
| [`feedback`](feedback.md) | `epiforcasts-feedback` | Record one Trust/ICB utility feedback session. |
| [`covariate`](covariate.md) | `epiforcasts-covariate` | Covariate alignment correlation checks. |
| [`launch`](launch.md) | `epiforcasts-launch` | Low-level Streamlit launcher (explicit app path). |
| [`dashboard`](dashboard.md) | `epiforcasts-dashboard` | Launch the full or fast dashboard with port fallback. |
| [`full-pipeline`](full-pipeline.md) | — (group only) | Seed, then run the daemon continuously. |
| [`dev`](dev.md) | — (group only) | Generate → one inference cycle → fast UI. |

## Environment variables

These influence inference behaviour regardless of which command triggers it:

| Variable | Default | Effect |
| --- | --- | --- |
| `PRESSURE_MODEL_FAST` | `1` | When truthy, defaults inference to the ADVI fast path. |
| `PRESSURE_MODEL_ADVI_STEPS` | `20000` | Default ADVI iteration count (overridden by `inference --advi-steps`). |
| `PRESSURE_MODEL_PYTENSOR_MODE` | _(unset)_ | Override the PyTensor backend, e.g. `NUMBA`, `JAX`, `FAST_RUN`. Left unset, PyTensor's own default is used. See [PYTENSOR_COMPILER.md](../40-operations/PYTENSOR_COMPILER.md). |

## Conventions

- **Exit codes:** commands exit non-zero on failure (missing data, failed
  checks, locked artifacts). Composite commands (`full-pipeline`, `dev`) abort
  if any step fails.
- **Paths:** all path defaults are relative to the current working directory.
