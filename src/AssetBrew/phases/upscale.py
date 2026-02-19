"""Upscale texture assets with Real-ESRGAN and robust fallbacks.

This phase handles model resolution, adaptive tiling, OOM recovery, and
texture-type-specific post-processing for game assets.
"""

import math
import os
import logging
import hashlib
import sys
import threading
from typing import Tuple

import numpy as np
import cv2

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord, load_image, save_image, ensure_rgb, extract_alpha,
    merge_alpha, pad_for_tiling, crop_from_padded, get_intermediate_path,
    GPUMonitor, srgb_to_linear, linear_to_srgb
)
from ..core.compat import ensure_torchvision_functional_tensor_alias

logger = logging.getLogger("asset_pipeline.upscaler")

# Known model URLs for auto-download
MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}


def _snap_to_pot(n: int) -> int:
    """Snap a positive integer to the nearest power of two."""
    if n <= 1:
        return 1
    log2 = math.log2(n)
    lower = 1 << int(log2)
    upper = lower << 1
    # Pick whichever PoT is closer; prefer upper on tie
    return lower if (n - lower) < (upper - n) else upper


def _compute_target_dims(
    h: int, w: int, target_res: int, enforce_power_of_two: bool = False
) -> Tuple[int, int]:
    """Compute target dimensions from source size and target resolution.

    Default behavior preserves aspect ratio without forcing power-of-two output.
    Set ``enforce_power_of_two=True`` to snap both dimensions to PoT values.
    """
    max_dim = max(h, w)
    if max_dim <= 0:
        return target_res, target_res

    if not enforce_power_of_two:
        scale = max(float(target_res) / float(max_dim), 1e-8)
        target_h = max(int(round(h * scale)), 1)
        target_w = max(int(round(w * scale)), 1)
        return target_h, target_w

    # Snap the target resolution itself to PoT
    pot_target = _snap_to_pot(target_res)
    original_ratio = max(h, w) / max(min(h, w), 1)

    if h >= w:
        # Height is the larger dimension
        target_h = pot_target
        ideal_w = max(int(round(w * pot_target / h)), 1)
        target_w = _pick_best_pot(ideal_w, target_h, original_ratio, is_height_major=True)
    else:
        # Width is the larger dimension
        target_w = pot_target
        ideal_h = max(int(round(h * pot_target / w)), 1)
        target_h = _pick_best_pot(ideal_h, target_w, original_ratio, is_height_major=False)

    # Warn if aspect ratio distortion is significant
    new_ratio = max(target_h, target_w) / max(min(target_h, target_w), 1)
    if original_ratio > 0:
        distortion = abs(new_ratio - original_ratio) / original_ratio
        if distortion > 0.05:
            logger.warning(
                f"PoT snapping distorts aspect ratio by {distortion:.0%}: "
                f"{w}x{h} (ratio {original_ratio:.2f}) -> "
                f"{target_w}x{target_h} (ratio {new_ratio:.2f})"
            )

    return target_h, target_w


def _pick_best_pot(ideal: int, major_dim: int, original_ratio: float,
                   is_height_major: bool) -> int:
    """Pick the PoT neighbor for the minor dimension that minimizes aspect distortion."""
    if ideal <= 1:
        return 1
    log2 = math.log2(ideal)
    lower = 1 << int(log2)
    upper = lower << 1

    def _ratio(minor):
        return major_dim / max(minor, 1)

    dist_lower = abs(_ratio(lower) - original_ratio)
    dist_upper = abs(_ratio(upper) - original_ratio)
    return lower if dist_lower <= dist_upper else upper


class TextureUpscaler:
    """AI-powered texture upscaler with game-asset-specific handling."""

    def __init__(self, config: PipelineConfig):
        """Initialize upscaler state, device selection, and model cache."""
        self.config = config
        self.cfg = config.upscale
        self.device = config.resolve_device()
        self.gpu_monitor = GPUMonitor(config)
        self._model = None
        self._model_name = None
        self._model_netscale = 4  # native model scale (2 or 4)
        self._current_tile_size = self.cfg.tile_size
        self._state_lock = threading.RLock()

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        """Return True only for CUDA/HIP out-of-memory style failures."""
        msg = str(exc).lower()
        markers = (
            "out of memory",
            "cuda error: out of memory",
            "cudaerrormemoryallocation",
            "cudnn_status_alloc_failed",
            "cublas_status_alloc_failed",
            "hip error: out of memory",
        )
        return any(marker in msg for marker in markers)

    @staticmethod
    def _is_binary_alpha(alpha: np.ndarray, tolerance: float = 0.02) -> bool:
        """Return True when alpha behaves like a binary mask."""
        if alpha.size == 0:
            return False
        a = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        return bool(np.all((a <= tolerance) | (a >= 1.0 - tolerance)))

    @staticmethod
    def _sha256_file(path: str) -> str:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_model_checksum(
        self, model_path: str, model_name: str, required: bool = False
    ):
        """Verify model checksum against configured SHA-256 values."""
        expected = (self.cfg.model_sha256.get(model_name) or "").strip().lower()
        if not expected:
            if required:
                raise RuntimeError(
                    f"Refusing to use unverified model download for '{model_name}'. "
                    "Set upscale.model_sha256.<model_name> to the expected SHA-256, "
                    "or explicitly set upscale.allow_unverified_model_download=true."
                )
            return
        actual = self._sha256_file(model_path).lower()
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for model '{model_name}': "
                f"expected {expected}, got {actual}. "
                "Delete the model file and retry download."
            )

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model .pth path: check local dir, download if needed."""
        model_dir = self.cfg.model_dir
        os.makedirs(model_dir, exist_ok=True)

        local_path = os.path.join(model_dir, f"{model_name}.pth")
        if os.path.exists(local_path):
            self._verify_model_checksum(local_path, model_name)
            logger.info(f"Using local model: {local_path}")
            return local_path

        # Check if basicsr/realesrgan cached it
        cache_dirs = [
            os.path.expanduser("~/.cache/realesrgan"),
            os.path.join(os.path.dirname(__file__), "weights"),
        ]
        for d in cache_dirs:
            cached = os.path.join(d, f"{model_name}.pth")
            if os.path.exists(cached):
                self._verify_model_checksum(cached, model_name)
                logger.info(f"Using cached model: {cached}")
                return cached

        # Download
        url = MODEL_URLS.get(model_name)
        if not url:
            raise FileNotFoundError(
                f"Model '{model_name}' not found locally and no download URL known. "
                f"Place the .pth file in {model_dir}/ or use one of: "
                f"{list(MODEL_URLS.keys())}"
            )

        if not self.cfg.allow_unverified_model_download:
            expected = (self.cfg.model_sha256.get(model_name) or "").strip().lower()
            if not expected:
                raise RuntimeError(
                    f"Refusing to auto-download '{model_name}' without SHA-256. "
                    "Set upscale.model_sha256.<model_name> to the expected checksum, "
                    "or set upscale.allow_unverified_model_download=true."
                )

        logger.info(f"Downloading model {model_name} from {url}...")
        try:
            from basicsr.utils.download_util import load_file_from_url
            downloaded = load_file_from_url(
                url, model_dir=model_dir, progress=True, file_name=None
            )
            self._verify_model_checksum(
                downloaded,
                model_name,
                required=not self.cfg.allow_unverified_model_download,
            )
            if self.cfg.allow_unverified_model_download and not self.cfg.model_sha256.get(
                model_name
            ):
                logger.warning(
                    f"Unverified model download allowed for '{model_name}'. "
                    "This is not recommended for production."
                )
            logger.info(f"Model downloaded to: {downloaded}")
            return downloaded
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model {model_name}: {e}\n"
                f"Manually download from {url} and place in {model_dir}/"
            ) from e

    def _load_model(self, model_name: str = None):
        """Load Real-ESRGAN model with explicit path resolution."""
        model_name = model_name or self.cfg.model_name
        with self._state_lock:
            if self._model is not None and self._model_name == model_name:
                logger.debug(
                    "Reusing cached upscaler model '%s' for faster startup.",
                    model_name
                )
                return self._model

            ensure_torchvision_functional_tensor_alias()

            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError as exc:
                if sys.version_info >= (3, 13):
                    raise ImportError(
                        "Real-ESRGAN/BasicSR is not currently installable on Python 3.13+.\n"
                        "Use Python 3.10-3.12 (recommended: 3.11), then install:\n"
                        "  pip install realesrgan basicsr"
                    ) from exc
                raise ImportError(
                    "Real-ESRGAN not installed. Run:\n"
                    "  pip install realesrgan basicsr"
                ) from exc
            except Exception as exc:
                raise ImportError(
                    "Real-ESRGAN/BasicSR failed to import. "
                    "This usually means the PyTorch install is broken in this environment.\n"
                    "Reinstall torch/torchvision for this Python environment and retry.\n"
                    f"Original error: {exc.__class__.__name__}: {exc}"
                ) from exc

            # Optional: SRVGGNetCompact for realesr-general-x4v3
            _SRVGGNetCompact = None
            try:
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact as _SRVGGNetCompact
            except Exception:
                pass

            model_path = self._resolve_model_path(model_name)

            # Model architecture selection
            if "x4plus_anime_6B" in model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4
                )
                netscale = 4
            elif "x4v3" in model_name:
                # realesr-general-x4v3 uses SRVGGNetCompact, not RRDBNet
                if _SRVGGNetCompact is None:
                    raise ImportError(
                        "SRVGGNetCompact not available. Update realesrgan:\n"
                        "  pip install --upgrade realesrgan"
                    )
                model = _SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_conv=32, upscale=4, act_type='prelu'
                )
                netscale = 4
            elif "x4" in model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4
            elif "x2" in model_name:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2
                )
                netscale = 2
            else:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                netscale = 4

            # Adaptive tile size based on GPU memory
            tile_size = self._current_tile_size
            if self.gpu_monitor.available:
                tile_size = self.gpu_monitor.suggest_tile_size(
                    tile_size, 512, 512  # Conservative estimate
                )
                self._current_tile_size = tile_size

            use_half = self.cfg.half_precision and self.device == "cuda"

            self._model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                model=model,
                tile=tile_size,
                tile_pad=self.cfg.tile_pad,
                pre_pad=0,
                half=use_half,
                device=self.device,
            )
            self._model_name = model_name
            self._model_netscale = netscale
            logger.info(
                f"Loaded model: {model_name} on {self.device} "
                f"(tile={tile_size}, half={use_half})"
            )
            return self._model

    def process(self, record: AssetRecord) -> dict:
        """Upscale a single asset with OOM recovery."""
        tex_type = TextureType(record.texture_type)
        input_path = os.path.join(self.config.input_dir, record.filepath)
        result = {"upscaled": None, "skipped": False, "error": None}

        if tex_type.value in self.cfg.skip_types:
            logger.debug(f"Skipping (type={tex_type.value}): {record.filename}")
            result["skipped"] = True
            return result

        try:
            img = load_image(input_path, max_pixels=self.config.max_image_pixels)
            h, w = img.shape[:2]
            min_dim = max(int(self.config.min_texture_dim), 1)
            if h < min_dim or w < min_dim:
                raise ValueError(
                    f"Input texture is too small for stable processing: {w}x{h}. "
                    f"Minimum supported size is {min_dim}x{min_dim}."
                )
            target_res = self.cfg.hero_resolution if record.is_hero else self.cfg.target_resolution
            # Respect the user-configurable srgb_texture_types list so that
            # e.g. SPECULAR can be treated as linear in metal/rough workflows.
            srgb_types = set(self.config.compression.srgb_texture_types)
            gamma_correct_model_input = tex_type.value in srgb_types

            with self._state_lock:
                if tex_type.value in self.cfg.nearest_neighbor_types:
                    upscaled = self._upscale_nearest(img, h, w, target_res)
                elif tex_type == TextureType.NORMAL:
                    upscaled = self._upscale_normal(img, h, w, target_res)
                else:
                    upscaled = self._upscale_ai_with_recovery(
                        img, h, w, target_res,
                        preserve_tiling=(self.cfg.preserve_tiling and record.is_tileable),
                        gamma_correct_model_input=gamma_correct_model_input,
                    )

            # Invert glossiness -> roughness after upscaling
            if record.is_gloss and tex_type == TextureType.ROUGHNESS:
                upscaled = 1.0 - upscaled
                logger.info(f"Inverted glossiness -> roughness for {record.filename}")

            out_path = get_intermediate_path(
                record.filepath, "01_upscaled",
                self.config.intermediate_dir, ext=".png"
            )
            save_image(upscaled, out_path)
            result["upscaled"] = out_path
            result["was_gloss"] = record.is_gloss
            logger.info(
                f"Upscaled {record.filename}: {w}x{h} -> "
                f"{upscaled.shape[1]}x{upscaled.shape[0]}"
            )

        except Exception as e:
            logger.error(f"Upscale failed for {record.filename}: {e}", exc_info=True)
            result["error"] = str(e)

        if self.gpu_monitor.available and str(self.device).startswith("cuda"):
            self.gpu_monitor.clear_cache()
        self.gpu_monitor.log_usage(f"after:{record.filename}")
        return result

    def _upscale_ai_with_recovery(self, img: np.ndarray, orig_h: int, orig_w: int,
                                    target_res: int,
                                    preserve_tiling: bool = False,
                                    gamma_correct_model_input: bool = False) -> np.ndarray:
        """AI upscale with OOM recovery: reduce tile size or fall back to CPU."""
        with self._state_lock:
            last_error = None
            tile_size = self._current_tile_size

            while tile_size >= self.config.gpu.min_tile_size:
                try:
                    self._current_tile_size = tile_size
                    # Update tile size on existing model if possible.
                    if self._model is not None:
                        self._model.tile = tile_size
                    return self._upscale_ai(
                        img,
                        orig_h,
                        orig_w,
                        target_res,
                        preserve_tiling,
                        gamma_correct_model_input,
                    )

                except RuntimeError as e:
                    if self._is_cuda_oom_error(e):
                        last_error = e
                        logger.warning(
                            f"OOM at tile_size={tile_size}, reducing to {tile_size // 2}"
                        )
                        # Free model weights from VRAM before clearing cache.
                        self._model = None
                        self._model_name = None
                        self.gpu_monitor.clear_cache()
                        tile_size = tile_size // 2
                    else:
                        raise

            # Final fallback: CPU
            if self.config.gpu.fallback_to_cpu and self.device != "cpu":
                logger.warning("All tile sizes exhausted, falling back to CPU")
                old_device = self.device
                self.device = "cpu"
                self._model = None
                self._model_name = None
                self._current_tile_size = self.cfg.tile_size
                try:
                    result = self._upscale_ai(
                        img,
                        orig_h,
                        orig_w,
                        target_res,
                        preserve_tiling,
                        gamma_correct_model_input,
                    )
                    return result
                finally:
                    self.device = old_device
                    self._model = None
                    self._model_name = None

            raise RuntimeError(
                f"Upscale failed after all recovery attempts. Last error: {last_error}"
            )

    def _upscale_ai(self, img: np.ndarray, orig_h: int, orig_w: int,
                    target_res: int,
                    preserve_tiling: bool = False,
                    gamma_correct_model_input: bool = False) -> np.ndarray:
        """Core AI upscale with aspect ratio preservation and tiling support."""
        alpha = extract_alpha(img)
        rgb = ensure_rgb(img)
        # NOTE: Real-ESRGAN was trained on sRGB uint8 data, so we must NOT
        # linearize before inference. The gamma_correct_model_input flag is
        # kept for the resize step below (sRGB-correct Lanczos downscale).

        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        del rgb
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        del rgb_uint8

        if preserve_tiling:
            bgr_float = bgr.astype(np.float32) / 255.0
            del bgr
            padded, pad_info = pad_for_tiling(bgr_float, pad_fraction=0.25)
            del bgr_float
            padded_uint8 = (np.clip(padded, 0, 1) * 255).astype(np.uint8)
            del padded

            model = self._load_model()
            netscale = self._model_netscale
            # Use native model scale to avoid an internal resize inside
            # RealESRGANer.  We do a single resize to target dimensions
            # afterwards to prevent compounding interpolation loss from two
            # sequential resizes.
            output_bgr, _ = model.enhance(padded_uint8, outscale=netscale)
            del padded_uint8

            # Crop using exact pad_info
            output_float = output_bgr.astype(np.float32) / 255.0
            del output_bgr
            output_float = crop_from_padded(
                output_float, pad_info, netscale
            )
            output_bgr = (np.clip(output_float, 0, 1) * 255).astype(np.uint8)
            del output_float
        else:
            model = self._load_model()
            netscale = self._model_netscale
            # Use native model scale to avoid an internal resize; we resize
            # once to target dimensions below.
            output_bgr, _ = model.enhance(bgr, outscale=netscale)
            del bgr

        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        del output_bgr

        # Resize to target while preserving aspect ratio
        target_h, target_w = _compute_target_dims(
            orig_h,
            orig_w,
            target_res,
            enforce_power_of_two=bool(self.cfg.enforce_power_of_two),
        )
        curr_h, curr_w = output_rgb.shape[:2]
        if curr_h != target_h or curr_w != target_w:
            if gamma_correct_model_input:
                # sRGB-correct resize: linearize -> Lanczos -> re-encode
                output_rgb = srgb_to_linear(output_rgb)
                output_rgb = cv2.resize(
                    output_rgb, (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
                output_rgb = linear_to_srgb(np.clip(output_rgb, 0, 1))
            else:
                output_rgb = cv2.resize(
                    output_rgb, (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
            output_rgb = np.clip(output_rgb, 0, 1).astype(np.float32)

        # Handle alpha
        if alpha is not None:
            if self._is_binary_alpha(alpha):
                alpha_interp = cv2.INTER_NEAREST
            elif target_h < alpha.shape[0] or target_w < alpha.shape[1]:
                # Downscaling: use INTER_AREA (box filter) to avoid
                # Lanczos ringing/halos at cutout edges.
                alpha_interp = cv2.INTER_AREA
            else:
                # Upscaling soft alpha: cubic avoids Lanczos overshoot
                alpha_interp = cv2.INTER_CUBIC
            alpha_resized = cv2.resize(
                alpha, (target_w, target_h),
                interpolation=alpha_interp
            )
            alpha_resized = np.clip(alpha_resized, 0, 1).astype(np.float32)
            output_rgb = merge_alpha(output_rgb, alpha_resized)

        return output_rgb

    def _upscale_nearest(self, img: np.ndarray, orig_h: int, orig_w: int,
                         target_res: int) -> np.ndarray:
        """Upscale masks/opacity with nearest neighbor (preserve hard edges)."""
        target_h, target_w = _compute_target_dims(
            orig_h,
            orig_w,
            target_res,
            enforce_power_of_two=bool(self.cfg.enforce_power_of_two),
        )
        # Stay in float32 to preserve gradient precision in soft masks
        resized = cv2.resize(
            np.clip(img, 0, 1).astype(np.float32), (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        return resized

    def _upscale_normal(self, img: np.ndarray, orig_h: int, orig_w: int,
                        target_res: int) -> np.ndarray:
        """Upscale normal maps with safe interpolation + renormalization."""
        target_h, target_w = _compute_target_dims(
            orig_h,
            orig_w,
            target_res,
            enforce_power_of_two=bool(self.cfg.enforce_power_of_two),
        )
        alpha = extract_alpha(img)
        if alpha is not None:
            logger.debug("Normal map has alpha channel; preserving through upscale.")
        rgb = ensure_rgb(img).astype(np.float32, copy=False)

        # Use INTER_AREA for downscale, INTER_LINEAR_EXACT for upscale
        # to avoid Lanczos ringing on encoded [0,1] normals.
        if target_h < rgb.shape[0] or target_w < rgb.shape[1]:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR_EXACT
        resized = cv2.resize(
            np.clip(rgb, 0, 1), (target_w, target_h),
            interpolation=interp,
        ).astype(np.float32)

        # Renormalize
        normals = resized * 2.0 - 1.0
        length = np.sqrt(np.sum(normals ** 2, axis=-1, keepdims=True))
        length = np.maximum(length, 1e-8)
        normals = normals / length
        result = (normals * 0.5 + 0.5).astype(np.float32)

        if alpha is not None:
            alpha_resized = cv2.resize(
                alpha, (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            result = merge_alpha(result, alpha_resized)
        return result

    def cleanup(self):
        """Free GPU memory."""
        with self._state_lock:
            self._model = None
            self._model_name = None
        self.gpu_monitor.clear_cache()

    def __enter__(self):
        """Context manager entry for deterministic cleanup."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit cleanup hook."""
        self.cleanup()
        return False

    def __del__(self):
        """Best-effort cleanup when object is collected."""
        try:
            self.cleanup()
        except Exception:
            pass
