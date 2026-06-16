# Legacy Redirect: Offline Inference

This page is retained as a redirect stub.

Canonical locations:

1. [../../40-operations/RUNBOOK.md](../../40-operations/RUNBOOK.md)
2. [../../30-model/TECHNICAL_SUMMARY_ADVANCED.md](../../30-model/TECHNICAL_SUMMARY_ADVANCED.md)

Related canonical pages:

1. [../../40-operations/FIRST_RUN_DUMMIES.md](../../40-operations/FIRST_RUN_DUMMIES.md)
2. [../../../README.md](../../../README.md)

Simple JSON with ICB names and counts (for quick reference by the UI).

## Code Changes

### New: `run_inference.py`
- Loads synthetic data
- Fits PyMC model
- Converts trace to ArviZ InferenceData
- Saves to NetCDF

**Key function**: `fit_pressure_model(df, fast=True)` returns `(model, idata)`

### Updated: `app.py`

**Removed**:
- `from bayesian_pressure_model import fit_pressure_model`
- `run_model()` cache function
- Inline inference code

**Added**:
- `import arviz as az`
- `load_posteriors()` function — loads pre-computed posterior from NetCDF
- Updated `_pressure_index_samples()` — works with ArviZ InferenceData instead of PyMC trace

**Key change**: 
```python
# Before: run inference on demand
_, trace = run_model(subset)
samples = _pressure_index_samples(trace, icb_codes)

# After: load pre-computed posterior
idata = load_posteriors(POSTERIORS_NC, mtime)
samples = _pressure_index_samples(idata, icb_idx)
```

### Updated: `pyproject.toml`

Added commands (subcommands of the unified `epiforcasts` click CLI):
- `epiforcasts inference` — runs offline Bayesian inference
- `epiforcasts full-pipeline` — generate → infer → dashboard in sequence

### New: `inference_daemon.py`

Continuously monitors data and automatically reruns inference when needed.

**Key features:**
- Configurable check interval (default: 1 hour)
- Only reruns if data is newer than posterior
- Logs all activity to `inference_daemon.log`
- Supports both continuous (daemon mode) and one-shot (CI/cron) execution
- Graceful error handling and shutdown

**Usage:**
```bash
# Continuous daemon mode (default)
python inference_daemon.py

# Custom interval (30 minutes)
python inference_daemon.py --interval 1800

# Full sampling
python inference_daemon.py --full

# One-shot execution (for cron/CI)
python inference_daemon.py --once
```

### New: `app_fast.py`

Ultra-lightweight Streamlit UI for read-only posterior display.

**Optimizations vs `app.py`:**
- No data file loading (uses cached posteriors only)
- Fixed display parameters (no interactive tuning)
- Minimal UI components (reduces rendering overhead)
- Fast startup (<0.5s)
- Perfect for:
  - Embedded dashboards
  - High-traffic scenarios
  - Read-only access
  - Mobile/low-bandwidth clients

**Usage:**
```bash
uv run epiforcasts dashboard --fast

# Or on custom port
streamlit run app_fast.py --server.port 8502
```

| Aspect | Before | After |
|--------|--------|-------|
| **Dashboard startup** | Slow (waits for sampling) | Instant (loads precomputed) |
| **Inference latency** | Embedded in UI (blocks all users) | Decoupled (run offline/scheduled) |
| **Scalability** | Hard to share results or cache | Easy (posteriors are files) |
| **Resource usage** | UI process needs compiler + PyMC | UI only needs pandas + arviz |
| **Updates** | Manual re-run per user | Scheduled batch inference |
| **DevOps** | Can't containerize easily | Separate inference container/Lambda |

## Deployment Scenarios

### Development
```bash
uv run epiforcasts generate
uv run epiforcasts inference --fast
uv run epiforcasts dashboard
```

### Scheduled (Cloud/CI)
```bash
# Nightly cron job
python run_inference.py --data-path s3://bucket/latest-data.csv --output-path posteriors.nc
```

### Production: Daemon + Fast Serving (Recommended)
```bash
# Terminal 1: Continuous background inference (checks hourly for data updates)
uv run epiforcasts daemon

# Terminal 2: Full interactive dashboard (port 8501)
uv run epiforcasts dashboard

# Terminal 3: Fast read-only view (port 8502)
streamlit run app_fast.py --server.port 8502
```

Benefits:
- **Daemon** handles all inference autonomously
- **Full dashboard** (`app.py`) for exploratory analysis and tuning
- **Fast dashboard** (`app_fast.py`) for read-only access, embedded views, or high-traffic scenarios
- Zero blocking on UI startup

### Multi-Container (Cloud)
```dockerfile
# Dockerfile.inference
FROM python:3.11
RUN pip install uv && uv sync --project /app
ENTRYPOINT ["uv", "run", "--project", "/app", "epiforcasts", "daemon", "--once"]

# Dockerfile.dashboard-full
FROM python:3.11
COPY posteriors.nc /app/posteriors.nc
RUN pip install uv && uv sync --project /app
ENTRYPOINT ["uv", "run", "--project", "/app", "epiforcasts", "dashboard"]

# Dockerfile.dashboard-fast
FROM python:3.11
COPY posteriors.nc /app/posteriors.nc
RUN pip install uv && uv sync --project /app
ENTRYPOINT ["uv", "run", "--project", "/app", "epiforcasts", "dashboard", "--fast"]
```

Orchestration example (Kubernetes):
```yaml
---
kind: CronJob
metadata:
  name: inference-job
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: inference
            image: inference:latest
            volumeMounts:
            - name: posteriors
              mountPath: /shared
          volumes:
          - name: posteriors
            persistentVolumeClaim:
              claimName: posteriors-pvc
          restartPolicy: OnFailure
---
kind: Deployment
metadata:
  name: dashboard-fast
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dashboard-fast
  template:
    metadata:
      labels:
        app: dashboard-fast
    spec:
      containers:
      - name: app
        image: dashboard-fast:latest
        volumeMounts:
        - name: posteriors
          mountPath: /shared
      volumes:
      - name: posteriors
        persistentVolumeClaim:
          claimName: posteriors-pvc
```

## Fallback Handling

If `posteriors.nc` is missing, the dashboard shows an error with helpful instructions:
```
Posterior file not found: posteriors.nc
Please run `python run_inference.py` first to generate posteriors.
```

## Next Steps

1. **Validation**: Verify posterior quality by checking ArviZ diagnostics (R̂, ESS) in `run_inference.py`:
   ```python
   print(az.summary(idata))
   ```

2. **Scheduling**: Set up a cron or Lambda to run inference on a schedule:
   ```bash
   0 2 * * * cd /path/to/repo && uv run epiforcasts generate && uv run epiforcasts inference --fast
   ```

3. **Monitoring**: Log posterior fits to a monitoring system (e.g., CloudWatch) to track drift or failures.

4. **Versioning**: Add a version stamp to `posteriors_metadata.json` for audit trails.
