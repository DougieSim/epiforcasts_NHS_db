# `epiforcasts full-pipeline`

**Group-only command** (no standalone `epiforcasts-*` script).

Composite workflow: **seed the initial dataset, then run the daemon
continuously** with fast cycles. Equivalent to the old Pixi `run-full-pipeline`
task. Use it for a from-scratch, self-updating demo environment.

## Usage

```bash
uv run epiforcasts full-pipeline
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--help` | flag | — | Show help and exit. |

## Steps (run in-process, in order)

1. [`seed`](seed.md) — build the initial dataset.
2. [`daemon --interval-hours 0.01`](daemon.md) — run continuous fast cycles
   (generate → infer → warm cache). This runs until interrupted (`Ctrl+C`).

If the seed step fails, the composite aborts before starting the daemon.

## Related

- [`dev`](dev.md) — shorter loop that ends by launching the fast UI.
