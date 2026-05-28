"""
Cache management for pre-computed posteriors.

Ensures all MCMC sampling happens offline; the UI only reads from cache.
Provides cache validation, warming, and pre-computed statistics.

Usage:
    from cache_manager import CacheManager
    
    cache = CacheManager()
    if cache.is_valid():
        stats = cache.load_summary_stats()
        idata = cache.load_posteriors()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import arviz as az

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cached posteriors and summary statistics.
    
    Ensures:
    - No MCMC runs during UI interaction
    - Pre-computed summaries for instant access
    - Cache validation and health checks
    - Automatic cache warming
    """
    
    def __init__(
        self,
        posteriors_path: Path | str = "posteriors.nc",
        cache_dir: Path | str = ".cache",
        ttl_hours: int = 24,
    ):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        posteriors_path : Path | str
            Path to NetCDF posterior file.
        cache_dir : Path | str
            Directory for pre-computed statistics cache.
        ttl_hours : int
            Cache validity period in hours (for warning if stale).
        """
        self.posteriors_path = Path(posteriors_path)
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        
        # Derived paths
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.summary_stats_path = self.cache_dir / "summary_stats.json"
        self.icb_samples_path = self.cache_dir / "icb_samples"
        self.health_check_path = self.cache_dir / "health_check.json"
        
        self.icb_samples_path.mkdir(parents=True, exist_ok=True)
    
    def is_posterior_available(self) -> bool:
        """Check if posterior file exists."""
        return self.posteriors_path.exists()
    
    def is_cache_warm(self) -> bool:
        """Check if summary statistics cache is populated."""
        return self.summary_stats_path.exists()
    
    def is_valid(self) -> bool:
        """Check if cache is valid and complete."""
        if not self.is_posterior_available():
            logger.warning(f"Posterior file missing: {self.posteriors_path}")
            return False
        
        if not self.is_cache_warm():
            logger.warning("Summary statistics cache not warmed; run warm_cache()")
            return False
        
        return True
    
    def is_stale(self) -> bool:
        """Check if posterior is older than TTL."""
        if not self.posteriors_path.exists():
            return True
        
        mtime = self.posteriors_path.stat().st_mtime
        age_seconds = (datetime.now().timestamp() - mtime)
        age_hours = age_seconds / 3600
        
        is_stale = age_hours > self.ttl_hours
        if is_stale:
            logger.warning(
                f"Posterior is stale ({age_hours:.1f}h > {self.ttl_hours}h); "
                "consider running inference_daemon or run_inference"
            )
        return is_stale
    
    def warm_cache(self) -> bool:
        """
        Pre-compute and cache summary statistics.
        
        This should be run after inference completes, so the UI has instant access.
        
        Returns True if successful, False otherwise.
        """
        if not self.is_posterior_available():
            logger.error(f"Cannot warm cache; posterior missing: {self.posteriors_path}")
            return False
        
        try:
            logger.info(f"Warming cache from {self.posteriors_path}...")
            
            # Load posteriors
            idata = az.from_netcdf(str(self.posteriors_path), engine="h5netcdf")
            
            # Extract ICB names and basic info
            icbs = idata.attrs.get("icbs", [])
            n_draws = idata.posterior.dims.get("draw", 0)
            
            if not icbs:
                logger.error("No ICB metadata in posterior; skipping cache warm")
                return False
            
            # Pre-compute summary statistics for each ICB
            summary_stats = {}
            for icb_idx, icb_name in enumerate(icbs):
                logger.debug(f"Computing statistics for {icb_name}…")
                
                samples = self._extract_samples(idata, icb_idx)
                
                # Compute statistics
                stats = {
                    "icb_name": icb_name,
                    "icb_idx": icb_idx,
                    "n_samples": len(samples),
                    "mean": float(np.mean(samples)),
                    "median": float(np.median(samples)),
                    "std": float(np.std(samples)),
                    "min": float(np.min(samples)),
                    "max": float(np.max(samples)),
                    "quantiles": {
                        "q05": float(np.percentile(samples, 5)),
                        "q25": float(np.percentile(samples, 25)),
                        "q50": float(np.percentile(samples, 50)),
                        "q75": float(np.percentile(samples, 75)),
                        "q95": float(np.percentile(samples, 95)),
                    },
                    "probabilities": {
                        "p_above_baseline": float(np.mean(samples > 0.0)),
                        "p_above_concern": float(np.mean(samples > 0.5)),
                        "p_above_elevated": float(np.mean(samples > 1.1)),
                    },
                }
                
                summary_stats[icb_name] = stats
                
                # Save samples for this ICB (for UI resampling if needed)
                icb_cache_file = self.icb_samples_path / f"{icb_idx:02d}_{icb_name.replace(' ', '_')}.npy"
                np.save(icb_cache_file, samples)
            
            # Save summary stats
            with open(self.summary_stats_path, "w") as f:
                json.dump(summary_stats, f, indent=2)
            
            # Write health check
            health = {
                "timestamp": datetime.now().isoformat(),
                "posteriors_mtime": self.posteriors_path.stat().st_mtime,
                "n_icbs": len(icbs),
                "n_draws": int(n_draws),
                "cache_version": 1,
            }
            with open(self.health_check_path, "w") as f:
                json.dump(health, f, indent=2)
            
            logger.info(
                f"[OK] Cache warmed: {len(summary_stats)} ICBs, "
                f"{len(self._list_cached_samples())} sample files"
            )
            return True
        
        except Exception as e:
            logger.error(f"Cache warming failed: {e}", exc_info=True)
            return False
    
    def load_posteriors(self) -> az.InferenceData:
        """Load posteriors from cache (no MCMC)."""
        if not self.is_posterior_available():
            raise FileNotFoundError(
                f"Posterior cache missing: {self.posteriors_path}\n"
                "Run: python inference_daemon.py --once"
            )
        
        logger.debug(f"Loading cached posteriors from {self.posteriors_path}")
        return az.from_netcdf(str(self.posteriors_path), engine="h5netcdf")
    
    def load_summary_stats(self) -> dict:
        """Load pre-computed summary statistics."""
        if not self.is_cache_warm():
            raise FileNotFoundError(
                f"Summary stats cache missing: {self.summary_stats_path}\n"
                "Run: python -c 'from cache_manager import CacheManager; "
                "CacheManager().warm_cache()'"
            )
        
        logger.debug(f"Loading summary stats from {self.summary_stats_path}")
        with open(self.summary_stats_path, "r") as f:
            return json.load(f)
    
    def load_samples(self, icb_idx: int, icb_name: str | None = None) -> np.ndarray:
        """
        Load pre-cached samples for an ICB.
        
        Uses cached .npy files (instant, no loading from NetCDF).
        """
        if icb_name is None:
            icb_name = f"icb_{icb_idx}"
        
        cache_file = self.icb_samples_path / f"{icb_idx:02d}_{icb_name.replace(' ', '_')}.npy"
        
        if not cache_file.exists():
            logger.debug(f"Sample cache missing for ICB {icb_idx}; loading from posterior")
            idata = self.load_posteriors()
            return self._extract_samples(idata, icb_idx)
        
        logger.debug(f"Loading cached samples from {cache_file}")
        return np.load(cache_file)
    
    def clear_cache(self) -> None:
        """Clear all cached statistics (but not posteriors)."""
        logger.info("Clearing cache...")
        import shutil
        if self.summary_stats_path.exists():
            self.summary_stats_path.unlink()
        if self.health_check_path.exists():
            self.health_check_path.unlink()
        if self.icb_samples_path.exists():
            shutil.rmtree(self.icb_samples_path)
        logger.info("[OK] Cache cleared")
    
    def get_health_check(self) -> dict:
        """Get cache health status."""
        if not self.health_check_path.exists():
            return {"status": "not_warmed"}
        
        with open(self.health_check_path, "r") as f:
            return json.load(f)
    
    def get_status_report(self) -> str:
        """Get human-readable cache status."""
        lines = [
            "Cache Status Report",
            "=" * 50,
        ]
        
        # Posterior file
        if self.posteriors_path.exists():
            size_mb = self.posteriors_path.stat().st_size / (1024**2)
            mtime = datetime.fromtimestamp(self.posteriors_path.stat().st_mtime)
            lines.append(f"[OK] Posteriors: {self.posteriors_path} ({size_mb:.1f} MB)")
            lines.append(f"  Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Stale: {self.is_stale()}")
        else:
            lines.append(f"[FAIL] Posteriors: MISSING")
        
        # Cache warmth
        if self.is_cache_warm():
            lines.append(f"[OK] Summary stats: {self.summary_stats_path}")
            lines.append(f"[OK] Samples: {len(self._list_cached_samples())} files")
        else:
            lines.append(f"[FAIL] Summary stats: NOT WARMED")
        
        # Health check
        health = self.get_health_check()
        if health.get("status") != "not_warmed":
            lines.append(f"[OK] Last warm: {health.get('timestamp')}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    @staticmethod
    def _extract_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
        level = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
        return level[:, :, -1, icb_idx].astype(float).ravel()
    
    def _list_cached_samples(self) -> list[Path]:
        """List all cached sample files."""
        return sorted(self.icb_samples_path.glob("*.npy"))


def main():
    """CLI for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument(
        "command",
        choices=["warm", "status", "clear", "check"],
        help="Command to run",
    )
    parser.add_argument(
        "--posteriors",
        type=Path,
        default="posteriors.nc",
        help="Path to posteriors.nc",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=".cache",
        help="Cache directory",
    )
    
    args = parser.parse_args()
    
    cache = CacheManager(posteriors_path=args.posteriors, cache_dir=args.cache_dir)
    
    if args.command == "warm":
        success = cache.warm_cache()
        print(cache.get_status_report())
        return 0 if success else 1
    
    elif args.command == "status":
        print(cache.get_status_report())
        return 0
    
    elif args.command == "check":
        if cache.is_valid():
            print("✓ Cache is valid and ready")
            return 0
        else:
            print("✗ Cache invalid or incomplete")
            return 1
    
    elif args.command == "clear":
        cache.clear_cache()
        print("✓ Cache cleared")
        return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(main() or 0)
