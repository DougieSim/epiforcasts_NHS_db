# `epiforcasts evidence`

**Standalone script:** `epiforcasts-evidence`

Run a standardized **evidence cycle** and append a timestamped entry to the
evidence run log. Optionally runs a fast inference pass first, then runs the
health and acceptance gates; the log entry is only written if all steps pass.

## Usage

```bash
uv run epiforcasts evidence                       # checks only
uv run epiforcasts evidence --run-inference-fast  # infer first, then check
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-inference-fast` | flag | off | Run a fast inference pass before the checks. |
| `--draws` | int | `400` | Draws recorded in the log entry's metrics. |
| `--n-icbs` | int | `7` | ICB count recorded in the log entry's metrics. |
| `--n-obs` | int | `1092` | Observation count recorded in the log entry's metrics. |
| `--help` | flag | — | Show help and exit. |

> The `--draws` / `--n-icbs` / `--n-obs` values are recorded in the log entry;
> they document the run, they do not change inference.

## Behaviour

- Runs (optionally) inference → health → acceptance, in order.
- Aborts and exits non-zero (log **not** updated) if any step fails.
- On success, appends an entry to `docs/90-changelog/logs/EVIDENCE_RUN_LOG.md`.

## Related

- [`acceptance`](acceptance.md), [`health`](health.md) — the underlying gates.
