# `epiforcasts seed`

**Standalone script:** `epiforcasts-create-initial-data`

Build the **initial** synthetic dataset from scratch — the full historical
weekly panel (ICB × weeks) and the patient-episode table that seed the model.
Run this **once** before anything else; subsequent weeks are added with
[`generate`](generate.md).

All values are fabricated — not real individuals or operational returns.

## Usage

```bash
uv run epiforcasts seed
# or
uv run epiforcasts-create-initial-data
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--help` | flag | — | Show help and exit. |

This command takes no configuration options; output paths are fixed by
`config.py` (`synthetic_nhs_pressure.csv`, `synthetic_patient_episodes.csv`).

## Output

- `synthetic_nhs_pressure.csv` — weekly panel (one row per ICB + England aggregate).
- `synthetic_patient_episodes.csv` — patient-episode table.

## Related

- [`generate`](generate.md) — extend the panel one week at a time.
- [`daemon`](daemon.md) — automate generate → infer → cache.
