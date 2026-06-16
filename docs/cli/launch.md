# `epiforcasts launch`

**Standalone script:** `epiforcasts-launch`

Low-level **Streamlit launcher** with automatic port fallback. Takes an explicit
app path, so it can run any Streamlit app in the project. For day-to-day use
prefer [`dashboard`](dashboard.md), which selects the full/fast app for you.

## Usage

```bash
uv run epiforcasts launch
uv run epiforcasts launch --app src/epiforcasts_nhs/dashboard/app_fast.py
uv run epiforcasts launch --preferred-port 8502 --max-port 8520
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--app` | text | full dashboard (`dashboard/app.py`) | Streamlit app path to run. |
| `--preferred-port` | int | `8501` | First port to try. |
| `--max-port` | int | `8510` | Highest port to try before giving up. |
| `--headless` | choice | `true` | `true` \| `false` — Streamlit headless mode. |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- Picks the first free port in `[preferred-port, max-port]`; exits non-zero if
  none are free.
- Prints the chosen port, then runs Streamlit and returns its exit code.

## Related

- [`dashboard`](dashboard.md) — higher-level launcher (`--fast` selects the app).
