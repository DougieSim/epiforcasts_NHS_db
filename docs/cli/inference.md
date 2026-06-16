# `epiforcasts inference`

**Standalone script:** `epiforcasts-inference`

Run **offline Bayesian inference** on the AR(1) pressure model and save posterior
summaries. This is the core modelling step; the dashboard only ever reads the
cached results it produces.

## Usage

```bash
uv run epiforcasts inference --fast                       # ADVI fast path (default)
uv run epiforcasts inference --full                       # NUTS with retries
uv run epiforcasts inference --fast --advi-steps 5000     # fewer ADVI steps
uv run epiforcasts-inference --output-path posteriors.nc  # standalone
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--data-path` | path | `synthetic_nhs_pressure.csv` | Input weekly panel CSV. |
| `--output-path` | path | `posteriors.nc` | Destination for posterior summaries. |
| `--fast / --full` | flag | `--fast` | `--fast`: ADVI fast path. `--full`: NUTS with retries / ADVI fallback. |
| `--seed` | int | `42` | Random seed for reproducibility. |
| `--advi-steps` | int | `$PRESSURE_MODEL_ADVI_STEPS` or `20000` | ADVI optimisation iterations (fast mode and NUTS fallback). |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- Exits non-zero if the input data file is missing.
- `--full` tries NUTS at two `target_accept` settings, then falls back to ADVI.
- Saving uses a Windows-safe atomic swap; if `posteriors.nc` is locked it stages
  the result for the next cycle rather than crashing.

## Notes

- The PyTensor backend can be switched via `PRESSURE_MODEL_PYTENSOR_MODE` (e.g.
  `NUMBA`) — see [PYTENSOR_COMPILER.md](../40-operations/PYTENSOR_COMPILER.md).
  For this model size the default backend is already fast.

## Related

- [`daemon`](daemon.md) — `--infer-only` wraps this step and warms the cache.
- [`cache`](cache.md) — warm the dashboard cache from a saved posterior.
