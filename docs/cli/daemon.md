# `epiforcasts daemon`

**Standalone script:** `epiforcasts-daemon`

The inference **daemon** — orchestrates the full cycle of generate new data →
run inference → warm cache. With no flags it runs **continuously** at a fixed
interval; flags let you run a single cycle or an individual step.

## Usage

```bash
uv run epiforcasts daemon                       # run forever, every hour
uv run epiforcasts daemon --once                # single cycle then exit
uv run epiforcasts daemon --once --fast         # single fast cycle
uv run epiforcasts daemon --generate-only       # only append a week
uv run epiforcasts daemon --infer-only          # only infer + warm cache
uv run epiforcasts daemon --interval-hours 0.5  # continuous, every 30 min
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--once` | flag | off | Run a single cycle then exit. |
| `--generate-only` | flag | off | Only append a new data week, then exit. |
| `--infer-only` | flag | off | Only run inference + warm cache, then exit. |
| `--fast` | flag | off | Use the ADVI fast inference path. |
| `--interval-hours` | float | `1.0` | Continuous-mode interval between cycles. |
| `--data` | path | `synthetic_nhs_pressure.csv` | Panel CSV to read/extend. |
| `--posterior-path` | path | `posteriors.nc` | Where posteriors are written. |
| `--cache-dir` | path | `.cache` | Cache directory to warm. |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- Single-shot modes (`--once`, `--infer-only`) exit non-zero if the cycle fails.
- Continuous mode logs the next-cycle time and retries on the next interval if a
  cycle fails. Stop with `Ctrl+C`.
- Mode precedence: `--generate-only` → `--infer-only` → `--once` → continuous.

## Related

- [`generate`](generate.md), [`inference`](inference.md), [`cache`](cache.md) — the individual steps.
- [`full-pipeline`](full-pipeline.md) — seed then run the daemon continuously.
