# `epiforcasts cache`

**Standalone script:** `epiforcasts-cache`

Manage the **pre-computed cache** the dashboard reads from. All MCMC sampling
happens offline; the UI only ever reads cached summary statistics, so the cache
must be warmed from a saved posterior before launching a dashboard.

Takes a single positional **action**: `warm | status | clear | check`.

## Usage

```bash
uv run epiforcasts cache warm     # build cache from posteriors.nc
uv run epiforcasts cache status   # human-readable status report
uv run epiforcasts cache check    # exit 0 if valid, non-zero if not
uv run epiforcasts cache clear    # delete cached artifacts
```

## Arguments & options

| Argument / option | Type | Default | Description |
| --- | --- | --- | --- |
| `COMMAND` | choice | _(required)_ | One of `warm`, `status`, `clear`, `check`. |
| `--posteriors` | path | `posteriors.nc` | Posterior file to warm/validate against. |
| `--cache-dir` | path | `.cache` | Cache directory to operate on. |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- `warm` — extracts summary stats and per-ICB samples; exits non-zero on failure.
- `check` — exits non-zero if the cache is invalid or incomplete (CI-friendly).
- `status` — prints a report (paths, sample count, last-warm time, staleness).
- `clear` — removes cached files.

## Related

- [`inference`](inference.md) — produces the posterior the cache is warmed from.
- [`daemon`](daemon.md) — warms the cache automatically after inference.
