# `epiforcasts dev`

**Group-only command** (no standalone `epiforcasts-*` script).

Composite **development loop**: append a week of data, run one inference cycle,
then launch the fast UI. Equivalent to the old Pixi `dev` task. The quickest way
to see an end-to-end change working.

## Usage

```bash
uv run epiforcasts dev
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--help` | flag | — | Show help and exit. |

## Steps (run in-process, in order)

1. [`generate`](generate.md) — append one new week.
2. [`daemon --once`](daemon.md) — run a single generate → infer → warm-cache cycle.
3. [`dashboard --fast`](dashboard.md) — launch the lightweight dashboard.

If any step fails, the composite aborts before the next one. Assumes the initial
dataset already exists — run [`seed`](seed.md) first on a fresh checkout (or use
[`full-pipeline`](full-pipeline.md)).

## Related

- [`full-pipeline`](full-pipeline.md) — from-scratch seed + continuous daemon.
