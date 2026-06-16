# `epiforcasts covariate`

**Standalone script:** `epiforcasts-covariate`

**Covariate alignment checks** — prints within-ICB correlations between candidate
operational covariates and bed occupancy, covariate inter-correlations, and
lagged correlations (covariate at week *t* predicting occupancy at *t+lag*). A
diagnostic for sanity-checking the synthetic data's correlation structure.

## Usage

```bash
uv run epiforcasts covariate
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--help` | flag | — | Show help and exit. |

Takes no configuration. The holdout window and lag set are fixed in `config.py`
(`COVARIATE_CHECK_HOLDOUT_WEEKS`, `COVARIATE_CHECK_LAGS`).

## Behaviour

- Reads `synthetic_nhs_pressure.csv`; exits non-zero if it is missing.
- Computes correlations on the training portion (excluding the holdout weeks),
  de-meaned within each ICB.

## Output

- Within-ICB correlations with `bed_occupancy`, ranked.
- Covariate inter-correlation matrix.
- Lagged correlations for each configured lag.

## Related

- [`seed`](seed.md) / [`generate`](generate.md) — produce the data being checked.
