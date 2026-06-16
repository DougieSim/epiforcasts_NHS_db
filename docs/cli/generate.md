# `epiforcasts generate`

**Standalone script:** `epiforcasts-generate`

Append one or more **new synthetic weeks** to the rolling panel CSV. Each new
week extends the latent pressure path one random-walk step and regenerates the
full set of correlated indicators, preserving the dataset's distributional
structure.

The rolling panel keeps a **fixed length**: each run appends the newest week and
drops the earliest, so the model's `n_weeks` stays constant. Every week ever
generated is also appended to a never-trimmed archive
(`synthetic_nhs_pressure_all.csv`).

## Usage

```bash
uv run epiforcasts generate                 # add one week
uv run epiforcasts generate --weeks 4       # add four weeks
uv run epiforcasts-generate --data my.csv   # standalone, custom file
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--data` | path | `synthetic_nhs_pressure.csv` | Rolling-window panel CSV to extend. |
| `--weeks` | int | `1` | Number of new weeks to generate. |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- Requires the data file to already exist (run [`seed`](seed.md) first); errors
  out and exits non-zero if missing.
- Writes the extended rolling panel back to `--data` and appends to the archive.

## Related

- [`seed`](seed.md) — create the initial dataset.
- [`daemon`](daemon.md) — `--generate-only` wraps this step.
