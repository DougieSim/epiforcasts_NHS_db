"""
Cache management for pre-computed posteriors.

Ensures all MCMC sampling happens offline; the UI only reads from cache.
Provides cache validation, warming, and pre-computed statistics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

import click
import numpy as np
import arviz as az

from epiforcasts_nhs.config import CACHE_DIR as _DEFAULT_CACHE_DIR
from epiforcasts_nhs.config import POSTERIORS_PATH as _DEFAULT_POSTERIORS_PATH
import epiforcasts_nhs.core.constants as constants
from epiforcasts_nhs.core.utils import current_pressure_samples as _extract_level_samples

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
        posteriors_path: Path | str = _DEFAULT_POSTERIORS_PATH,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
        ttl_hours: int = constants.CACHE.ttl_hours_default,
    ):
        self.posteriors_path = Path(posteriors_path)
        self.cache_dir       = Path(cache_dir)
        self.ttl_hours       = ttl_hours

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.summary_stats_path = self.cache_dir / "summary_stats.json"
        self.icb_samples_path   = self.cache_dir / "icb_samples"
        self.health_check_path  = self.cache_dir / "health_check.json"

        self.icb_samples_path.mkdir(parents=True, exist_ok=True)

    def is_posterior_available(self) -> bool:
        return self.posteriors_path.exists()

    def is_cache_warm(self) -> bool:
        return self.summary_stats_path.exists()

    def is_valid(self) -> bool:
        if not self.is_posterior_available():
            logger.warning(f"Posterior file missing: {self.posteriors_path}")
            return False
        if not self.is_cache_warm():
            logger.warning("Summary statistics cache not warmed; run warm_cache()")
            return False
        return True

    def is_stale(self) -> bool:
        if not self.posteriors_path.exists():
            return True
        age_hours = (datetime.now().timestamp() - self.posteriors_path.stat().st_mtime) / 3600
        is_stale  = age_hours > self.ttl_hours
        if is_stale:
            logger.warning(
                f"Posterior is stale ({age_hours:.1f}h > {self.ttl_hours}h); "
                "consider running inference_daemon or run_inference"
            )
        return is_stale

    def warm_cache(self) -> bool:
        if not self.is_posterior_available():
            logger.error(f"Cannot warm cache; posterior missing: {self.posteriors_path}")
            return False

        try:
            logger.info(f"Warming cache from {self.posteriors_path}...")
            idata   = az.from_netcdf(str(self.posteriors_path), engine="h5netcdf")
            icbs    = idata.attrs.get("icbs", [])
            n_draws = idata.posterior.dims.get("draw", 0)

            if not icbs:
                logger.error("No ICB metadata in posterior; skipping cache warm")
                return False

            summary_stats: dict = {}
            for icb_idx, icb_name in enumerate(icbs):
                samples = self._extract_samples(idata, icb_idx)
                summary_stats[icb_name] = {
                    "icb_name": icb_name,
                    "icb_idx":  icb_idx,
                    "n_samples": len(samples),
                    "mean":   float(np.mean(samples)),
                    "median": float(np.median(samples)),
                    "std":    float(np.std(samples)),
                    "min":    float(np.min(samples)),
                    "max":    float(np.max(samples)),
                    "quantiles": {
                        key: float(np.percentile(samples, pct))
                        for key, pct in zip(constants.CACHE.summary_quantile_keys, constants.CACHE.summary_quantile_pcts)
                    },
                    "probabilities": {
                        "p_above_baseline": float(np.mean(samples > constants.PRESSURE_THRESHOLDS.baseline)),
                        "p_above_concern":  float(np.mean(samples > constants.PRESSURE_THRESHOLDS.concern)),
                        "p_above_elevated": float(np.mean(samples > constants.PRESSURE_THRESHOLDS.elevated)),
                    },
                }
                icb_cache_file = self.icb_samples_path / f"{icb_idx:02d}_{icb_name.replace(' ', '_')}.npy"
                np.save(icb_cache_file, samples)

            with open(self.summary_stats_path, "w") as f:
                json.dump(summary_stats, f, indent=2)

            health = {
                "timestamp":        datetime.now().isoformat(),
                "posteriors_mtime": self.posteriors_path.stat().st_mtime,
                "n_icbs":           len(icbs),
                "n_draws":          int(n_draws),
                "cache_version":    constants.CACHE.version,
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
        if not self.is_posterior_available():
            raise FileNotFoundError(
                f"Posterior cache missing: {self.posteriors_path}\n"
                "Run: epiforcasts-daemon --once"
            )
        return az.from_netcdf(str(self.posteriors_path), engine="h5netcdf")

    def load_summary_stats(self) -> dict:
        if not self.is_cache_warm():
            raise FileNotFoundError(
                f"Summary stats cache missing: {self.summary_stats_path}\n"
                "Run: epiforcasts-cache warm"
            )
        with open(self.summary_stats_path, "r") as f:
            return json.load(f)

    def load_samples(self, icb_idx: int, icb_name: str | None = None) -> np.ndarray:
        if icb_name is None:
            icb_name = f"icb_{icb_idx}"
        cache_file = self.icb_samples_path / f"{icb_idx:02d}_{icb_name.replace(' ', '_')}.npy"
        if not cache_file.exists():
            return self._extract_samples(self.load_posteriors(), icb_idx)
        return np.load(cache_file)

    def clear_cache(self) -> None:
        import shutil
        logger.info("Clearing cache...")
        if self.summary_stats_path.exists():
            self.summary_stats_path.unlink()
        if self.health_check_path.exists():
            self.health_check_path.unlink()
        if self.icb_samples_path.exists():
            shutil.rmtree(self.icb_samples_path)
        logger.info("[OK] Cache cleared")

    def get_health_check(self) -> dict:
        if not self.health_check_path.exists():
            return {"status": "not_warmed"}
        with open(self.health_check_path, "r") as f:
            return json.load(f)

    def get_status_report(self) -> str:
        lines = ["Cache Status Report", "=" * 50]
        if self.posteriors_path.exists():
            size_mb = self.posteriors_path.stat().st_size / (1024 ** 2)
            mtime   = datetime.fromtimestamp(self.posteriors_path.stat().st_mtime)
            lines.append(f"[OK] Posteriors: {self.posteriors_path} ({size_mb:.1f} MB)")
            lines.append(f"  Updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  Stale: {self.is_stale()}")
        else:
            lines.append("[FAIL] Posteriors: MISSING")
        if self.is_cache_warm():
            lines.append(f"[OK] Summary stats: {self.summary_stats_path}")
            lines.append(f"[OK] Samples: {len(self._list_cached_samples())} files")
        else:
            lines.append("[FAIL] Summary stats: NOT WARMED")
        health = self.get_health_check()
        if health.get("status") != "not_warmed":
            lines.append(f"[OK] Last warm: {health.get('timestamp')}")
        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def _extract_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
        """
        Extract current-pressure samples for one ICB.

        Expects posteriors from the AR(1) model, which stores the latent level
        as ``idata.posterior["level"]`` with shape (chains, draws, n_weeks, n_icb).
        Returns the final week's level for the given ICB.
        """
        if "level" not in idata.posterior:
            raise KeyError(
                "Posterior does not contain 'level'. "
                "Ensure posteriors were generated by the AR(1) model "
                "(epiforcasts-daemon or epiforcasts-inference)."
            )
        return _extract_level_samples(idata, icb_idx)

    def _list_cached_samples(self) -> list[Path]:
        return sorted(self.icb_samples_path.glob("*.npy"))


@click.command(name="cache")
@click.argument("command", type=click.Choice(["warm", "status", "clear", "check"]))
@click.option("--posteriors", type=click.Path(path_type=Path), default=_DEFAULT_POSTERIORS_PATH, show_default=True)
@click.option("--cache-dir",  type=click.Path(path_type=Path), default=_DEFAULT_CACHE_DIR, show_default=True)
def main(command: str, posteriors: Path, cache_dir: Path) -> None:
    """Cache management utility: warm | status | clear | check."""
    logging.basicConfig(level=logging.INFO)

    cache = CacheManager(posteriors_path=posteriors, cache_dir=cache_dir)

    if command == "warm":
        success = cache.warm_cache()
        print(cache.get_status_report())
        raise SystemExit(0 if success else 1)
    elif command == "status":
        print(cache.get_status_report())
    elif command == "check":
        if cache.is_valid():
            print("Cache is valid and ready")
        else:
            print("Cache invalid or incomplete")
            raise SystemExit(1)
    elif command == "clear":
        cache.clear_cache()
        print("Cache cleared")


if __name__ == "__main__":
    main()
