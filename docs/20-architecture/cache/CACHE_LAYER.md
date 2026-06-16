# Cache Layer: Zero MCMC UI Guarantee

## Overview

The cache layer ensures that **MCMC never runs during user interaction**, providing instant dashboard startup and guaranteed read-only access to pre-computed results.

## How It Works

### Three-Layer Architecture

```
LAYER 1: Offline Inference
├── run_inference.py (manual)
└── inference_daemon.py (continuous)
    └─→ Fits PyMC model (takes 2-5 minutes)
    └─→ Saves posteriors.nc (NetCDF)
    └─→ Calls cache_manager.warm_cache()

LAYER 2: Cache Warming
├── cache_manager.warm_cache()
    ├─→ Loads posteriors.nc (binary read)
    ├─→ Pre-computes all statistics
    │   ├─ summary_stats.json (means, quantiles, probabilities)
    │   ├─ icb_samples/*.npy (individual ICB samples as arrays)
    │   └─ health_check.json (validation metadata)
    └─→ Returns: ✓ Cache is ready

LAYER 3: UI Loading
├── app.py / app_fast.py
    ├─→ Initialize CacheManager()
    ├─→ Call cache.is_valid()  ← Returns instantly (no I/O)
    │   └─ Checks: posteriors.nc exists?
    │   └─ Checks: .cache/summary_stats.json exists?
    ├─→ If valid:
    │   ├─ Load from cache (instant JSON + .npy read)
    │   └─ Display (zero computation)
    └─→ If invalid:
        └─ Show error + instructions
        └─ Stop (refuse to run MCMC)
```

## Cache Files

### Summary Statistics: `.cache/summary_stats.json`

Pre-computed for all ICBs during warm. Instant access (JSON is fast).

```json
{
  "NHS Region A": {
    "icb_name": "NHS Region A",
    "icb_idx": 0,
    "n_samples": 400,
    "mean": 0.42,
    "median": 0.38,
    "std": 0.85,
    "min": -2.1,
    "max": 3.5,
    "quantiles": {
      "q05": -1.2,
      "q25": -0.1,
      "q50": 0.38,
      "q75": 0.9,
      "q95": 2.1
    },
    "probabilities": {
      "p_above_baseline": 0.55,
      "p_above_concern": 0.12,
      "p_above_elevated": 0.03
    }
  },
  ...
}
```

### Sample Arrays: `.cache/icb_samples/*.npy`

Binary NumPy arrays (one per ICB). Fast for UI rendering.

```
.cache/icb_samples/
├── 00_NHS_Region_A.npy      # 400 posterior samples
├── 01_NHS_Region_B.npy
└── ...
```

### Health Check: `.cache/health_check.json`

Validation metadata to detect stale or mismatched cache.

```json
{
  "timestamp": "2026-04-23T14:30:00.123456",
  "posteriors_mtime": 1713900600.0,
  "n_icbs": 6,
  "n_draws": 400,
  "cache_version": 1
}
```

## Usage in UI

### app.py (Full Interactive)

```python
from cache_manager import CacheManager

# Initialize
cache = CacheManager(posteriors_path="posteriors.nc")

# Validate (instant, no I/O to posteriors)
if not cache.is_valid():
    st.error("Cache not ready")
    st.stop()

# Load (from .cache/, zero MCMC)
idata = cache.load_posteriors()           # Fast binary read from netcdf
stats = cache.load_summary_stats()        # Fast JSON read
```

### app_fast.py (Lightweight Read-Only)

```python
from cache_manager import CacheManager

cache = CacheManager(posteriors_path="posteriors.nc")

# Same validation
if not cache.is_valid():
    st.error("Cache not ready")
    st.stop()

# Load (same layer)
idata = cache.load_posteriors()
stats = cache.load_summary_stats()

# Extract ICB samples from pre-cached .npy
samples = cache.load_samples(icb_idx=0, icb_name="NHS Region A")
```

## Automatic Warming

### After `run_inference.py`

```python
# run_inference.py: save_posterior_summaries()
idata.to_netcdf("posteriors.nc")

# Auto-warm
from cache_manager import CacheManager
cache = CacheManager(posteriors_path="posteriors.nc")
cache.warm_cache()  # ← Populates .cache/ instantly
```

Output:
```
✓ Cache warmed: 6 ICBs, 6 sample files
```

### After `inference_daemon.py` (Continuous)

```python
# inference_daemon.py: run_inference_safe()
idata.to_netcdf("posteriors.nc")

# Auto-warm
cache = CacheManager()
cache.warm_cache()
```

This means:
- Daemon runs inference in background
- Cache is automatically refreshed
- UI always sees fresh data + cache
- No manual cache management needed

## Manual Cache Management

### Check Status

```bash
uv run epiforcasts cache status
# Output:
# ✓ Posteriors: posteriors.nc (2.4 MB)
# Updated: 2026-04-23 14:30:00
# Stale: False
# ✓ Summary stats: .cache/summary_stats.json
# ✓ Samples: 6 files
# ✓ Last warm: 2026-04-23 14:30:15
```

### Validate Before UI

```bash
uv run epiforcasts cache check
# Output: ✓ Cache is valid and ready
# Exit code: 0
```

### Force Warm

```bash
uv run epiforcasts cache warm
# Useful if posteriors.nc exists but cache is missing
```

### Clear Cache (Keep Posteriors)

```bash
uv run epiforcasts cache clear
# Clears .cache/ only
# Posteriors.nc remains
# UI will show error until cache is re-warmed
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| `cache.is_valid()` | <1ms | Stat check only, no I/O |
| `cache.load_posteriors()` | 100-500ms | NetCDF read (depends on size) |
| `cache.load_summary_stats()` | <10ms | JSON read (very fast) |
| `cache.load_samples(icb_idx)` | <50ms | .npy read (binary) |
| `cache.warm_cache()` | 5-15 seconds | First time only, then incremental |

## Guarantees

✅ **MCMC never runs during UI interaction**
- Validation fails fast if cache missing
- UI exits gracefully with instructions

✅ **No computation on UI load**
- All stats pre-computed
- Only reading from disk

✅ **Instant startup**
- `.cache/summary_stats.json` is ~50KB (instant load)
- No model fitting

✅ **Automatic refresh**
- `inference_daemon.py` warms cache after each fit
- UI always sees fresh data

✅ **Graceful degradation**
- Missing cache → clear error message
- Missing posterior → clear error message
- UI refuses to run inference as fallback

## Troubleshooting

**Q: Cache is missing**
```bash
uv run epiforcasts daemon --once  # Regenerates cache
# or
uv run epiforcasts cache warm
```

**Q: Cache is stale**
```bash
# Check age
uv run epiforcasts cache status

# Refresh
uv run epiforcasts daemon --once
```

**Q: Dashboard still slow**
```bash
# Verify cache is loaded
uv run epiforcasts cache check

# Check disk I/O
ls -lh .cache/

# Monitor load time
time python -c "from cache_manager import CacheManager; CacheManager().load_summary_stats()"
```

**Q: Cache and posteriors out of sync**
```bash
# Clear and regenerate
uv run epiforcasts cache clear
uv run epiforcasts daemon --once
```
