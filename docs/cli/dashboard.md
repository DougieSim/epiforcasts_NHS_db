# `epiforcasts dashboard`

**Standalone script:** `epiforcasts-dashboard`

Launch the Streamlit **dashboard** with automatic port fallback. Selects the
full dashboard by default, or the lightweight fast dashboard with `--fast`. This
is the recommended way to start the UI.

## Usage

```bash
uv run epiforcasts dashboard          # full dashboard
uv run epiforcasts dashboard --fast   # lightweight fast dashboard
uv run epiforcasts dashboard --fast --preferred-port 8502
```

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--fast` | flag | off | Launch the lightweight fast dashboard (`app_fast.py`) instead of the full one. |
| `--preferred-port` | int | `8501` | First port to try. |
| `--max-port` | int | `8510` | Highest port to try before giving up. |
| `--headless` | choice | `true` | `true` \| `false` — Streamlit headless mode. |
| `--help` | flag | — | Show help and exit. |

## Behaviour

- Resolves the app path (full vs fast) and launches via the direct Streamlit
  launcher, which auto-falls back through the port range and prints the chosen
  port.
- Requires a warmed cache — run [`daemon --once`](daemon.md) or
  [`cache warm`](cache.md) first if the dashboard reports an invalid cache.

> Historically this command tried Pixi first and fell back to direct Python;
> under uv there is a single launch path.

## Related

- [`launch`](launch.md) — lower-level launcher with an explicit `--app` path.
- [`cache`](cache.md) — warm the cache the dashboard reads.
