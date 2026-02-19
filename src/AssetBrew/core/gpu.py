"""GPU VRAM monitoring and adaptive tile sizing."""

import logging
from typing import Dict

from ..config import PipelineConfig

logger = logging.getLogger("asset_pipeline")


def _parse_device_id(device_str: str) -> int:
    """Extract CUDA device index from a device string (e.g. 'cuda:1' -> 1)."""
    if ":" in device_str:
        try:
            return int(device_str.split(":")[1])
        except (ValueError, IndexError):
            logger.warning("Could not parse CUDA device index from %s; using 0.", device_str)
            pass
    return 0


class GPUMonitor:
    """Monitor GPU VRAM usage and adaptively adjust tile sizes."""

    def __init__(self, config: PipelineConfig):
        """Initialize monitor state and detect CUDA availability.

        Importing torch is intentionally deferred unless the resolved runtime
        device is CUDA. This avoids touching torch on CPU-only runs where
        probing should never trigger native DLL loading.
        """
        self.cfg = config.gpu
        self._has_cuda = False
        self._max_vram_bytes = 0
        self._torch = None
        resolved_device = str(config.resolve_device()).strip().lower()
        self._device_id = _parse_device_id(resolved_device)

        if not resolved_device.startswith("cuda"):
            logger.debug(
                "GPU monitor disabled for non-CUDA device '%s'.",
                resolved_device,
            )
            return

        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                requested_id = self._device_id
                try:
                    device_count = int(torch.cuda.device_count())
                except Exception:
                    device_count = 0

                if device_count > 0 and requested_id >= device_count:
                    logger.warning(
                        f"Requested CUDA device {requested_id} is out of range "
                        f"(available: 0..{device_count - 1}). Falling back to cuda:0"
                    )
                    self._device_id = 0

                self._has_cuda = True
                if self.cfg.max_vram_mb > 0:
                    self._max_vram_bytes = self.cfg.max_vram_mb * 1024 * 1024
                else:
                    total = torch.cuda.get_device_properties(
                        self._device_id
                    ).total_memory
                    self._max_vram_bytes = int(total * 0.80)
        except ImportError:
            logger.debug("torch not available; GPU monitor disabled.")
            pass
        except Exception as e:
            logger.warning(
                f"GPU monitor disabled due to CUDA query error: {e}"
            )
            self._has_cuda = False
            self._max_vram_bytes = 0

    @property
    def available(self) -> bool:
        """Return whether CUDA monitoring is available and enabled."""
        return self._has_cuda and self.cfg.enabled

    def get_vram_usage(self) -> Dict[str, float]:
        """Return VRAM usage in MB."""
        if not self._has_cuda:
            logger.debug("VRAM usage requested but CUDA monitor is not available.")
            return {"allocated_mb": 0, "reserved_mb": 0, "max_mb": 0, "free_mb": 0}
        try:
            torch = self._torch
            if torch is None:
                return {"allocated_mb": 0, "reserved_mb": 0, "max_mb": 0, "free_mb": 0}
            did = self._device_id
            alloc = torch.cuda.memory_allocated(did)
            reserved = torch.cuda.memory_reserved(did)
            total = torch.cuda.get_device_properties(did).total_memory
            return {
                "allocated_mb": round(alloc / 1024 / 1024, 1),
                "reserved_mb": round(reserved / 1024 / 1024, 1),
                "max_mb": round(total / 1024 / 1024, 1),
                "free_mb": round((total - reserved) / 1024 / 1024, 1),
            }
        except Exception as e:
            logger.debug(f"VRAM query failed: {e}")
            return {"allocated_mb": 0, "reserved_mb": 0, "max_mb": 0, "free_mb": 0}

    def log_usage(self, label: str = ""):
        """Log current VRAM usage."""
        if not self.available or not self.cfg.log_vram:
            logger.debug(
                "Skipping VRAM log: available=%s, log_vram=%s",
                self.available, self.cfg.log_vram
            )
            return
        usage = self.get_vram_usage()
        prefix = f"[{label}] " if label else ""
        logger.debug(
            f"{prefix}VRAM: {usage['allocated_mb']}MB allocated, "
            f"{usage['free_mb']}MB free / {usage['max_mb']}MB total"
        )

    def check_oom_risk(self, needed_mb: float = 0) -> bool:
        """Check if we're at risk of OOM."""
        if not self.available:
            logger.debug("OOM risk requested but GPU monitor unavailable.")
            return False
        try:
            torch = self._torch
            if torch is None:
                return False
            reserved = torch.cuda.memory_reserved(self._device_id)
            needed_bytes = needed_mb * 1024 * 1024
            return (reserved + needed_bytes) > self._max_vram_bytes
        except Exception as e:
            logger.debug(f"OOM risk query failed: {e}")
            return False

    def suggest_tile_size(self, current_tile: int, image_h: int, image_w: int) -> int:
        """Suggest a tile size that fits in VRAM, reducing if necessary."""
        if not self.available:
            logger.debug("Tile size suggestion skipped because GPU monitor unavailable.")
            return current_tile

        min_tile = min(self.cfg.min_tile_size, current_tile)

        # Estimate VRAM needed: Real-ESRGAN uses ~300x raw tile size due to
        # 23 RRDB blocks of intermediate activations
        estimated_mb = (current_tile * current_tile * 3 * 4 * 300) / (1024 * 1024)

        tile = current_tile
        while self.check_oom_risk(estimated_mb) and tile > min_tile:
            tile = max(tile // 2, min_tile)
            estimated_mb = (tile * tile * 3 * 4 * 300) / (1024 * 1024)
            logger.warning(f"Reducing tile size to {tile} to avoid OOM")

        return max(tile, min_tile)

    def clear_cache(self):
        """Clear GPU cache."""
        if not self._has_cuda:
            logger.debug("GPU cache clear skipped; CUDA monitor is not initialized.")
            return
        try:
            torch = self._torch
            if torch is None:
                return
            torch.cuda.empty_cache()
            logger.debug(
                "Cleared CUDA cache on device %s (monitor enabled=%s)",
                self._device_id,
                self._has_cuda,
            )
        except Exception as e:
            logger.debug("Failed to clear CUDA cache: %s", e)
