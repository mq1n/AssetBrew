"""Image I/O utilities -- load/save numpy arrays with explicit bit-depth handling."""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Disable Pillow's global decompression bomb check; we validate per-call
# in load_image() and scan_assets() instead.  This avoids the need for a
# global lock that would serialize all parallel image loading.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("asset_pipeline")


@contextmanager
def temporary_max_image_pixels(max_pixels: int):
    """Context manager kept for backward compatibility.

    Actual pixel-count validation happens in load_image() and scan_assets()
    after reading the image header, so this is now a no-op pass-through.
    """
    if max_pixels < 0:
        raise ValueError("max_pixels must be >= 0 (0 = unlimited)")
    yield


# Backward-compatible alias for internal callers/tests.
_temporary_max_image_pixels = temporary_max_image_pixels


def _infer_integer_mode_bit_depth(img: Image.Image, ext: str) -> int:
    """Infer bit depth for Pillow mode ``I`` images.

    Prefers explicit metadata over pixel-statistics heuristics to avoid
    mis-normalizing dark 16-bit images.
    """
    bits_info = img.info.get("bits")
    if isinstance(bits_info, int) and bits_info > 0:
        logger.debug("Using img.info bits metadata for mode I: %s", bits_info)
        return bits_info

    # TIFF BitsPerSample tag
    tag_v2 = getattr(img, "tag_v2", None)
    if tag_v2 is not None:
        bits_tag = tag_v2.get(258)
        if isinstance(bits_tag, tuple) and bits_tag:
            bits_tag = bits_tag[0]
        if isinstance(bits_tag, int) and bits_tag > 0:
            logger.debug("Using TIFF BitsPerSample tag for mode I: %s", bits_tag)
            return bits_tag

    # Conservative fallback: TIFF "I" is frequently 16-bit data promoted to "I".
    if ext in (".tif", ".tiff"):
        logger.debug("Falling back to 16-bit for TIFF integer mode I for extension %s", ext)
        return 16
    logger.debug("Falling back to 32-bit for mode I with extension %s", ext)
    return 32


def load_image(path: str, max_pixels: int = 0) -> np.ndarray:
    """Load image as float32 numpy array.

    Integer formats are normalized to [0, 1]. Floating-point (mode ``F``)
    preserves absolute values except for common 8-bit/16-bit encoded float data.
    """
    ext = Path(path).suffix.lower()

    # DDS: ensure Pillow's DDS plugin is loaded (it auto-registers, but be explicit)
    if ext == ".dds":
        try:
            from PIL import DdsImagePlugin  # noqa: F401
        except ImportError:
            logger.error(
                "DDS support requires Pillow >= 9.0; ensure DDS plugin is installed."
            )
            raise ImportError(
                "DDS support requires Pillow >= 9.0. "
                "Upgrade with: pip install --upgrade Pillow"
            )

    try:
        with temporary_max_image_pixels(max_pixels), Image.open(path) as img:
            # Memory guard
            if max_pixels > 0 and img.width * img.height > max_pixels:
                logger.warning(
                    "Image %s exceeds max_pixels: %d > %d",
                    path,
                    img.width * img.height,
                    max_pixels,
                )
                raise ValueError(
                    f"Image too large: {img.width}x{img.height} = {img.width * img.height:,} "
                    f"pixels (max {max_pixels:,}). Resize input or increase max_image_pixels."
                )

            # Log DDS-specific info
            if ext == ".dds":
                pixel_format = getattr(img, "pixel_format", "unknown")
                logger.debug(
                    f"DDS loaded: {path} mode={img.mode} "
                    f"pixel_format={pixel_format} size={img.size}"
                )

            # 16-bit integer modes
            if img.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
                logger.debug("Loading %s as 16-bit integer mode %s", path, img.mode)
                arr = np.asarray(img, dtype=np.float32) / 65535.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                logger.debug("Loaded %s in 16-bit mode path; returning float array", path)
                return arr

            # 32-bit integer mode (may carry 16-bit or 32-bit source depth)
            if img.mode == "I":
                logger.debug("Loading %s as integer mode I", path)
                arr = np.asarray(img, dtype=np.float32)
                bit_depth = _infer_integer_mode_bit_depth(img, ext)
                max_value = float((1 << min(bit_depth, 32)) - 1)
                if max_value > 0:
                    arr = np.clip(arr / max_value, 0.0, 1.0)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                logger.debug(
                    "Loaded %s in mode I with inferred bit depth %d and clip max %.0f",
                    path,
                    bit_depth,
                    max_value,
                )
                return arr

            # 32-bit float mode -- preserve absolute values by default. Only
            # normalize when explicit metadata says values are integer-encoded.
            if img.mode == "F":
                arr = np.asarray(img, dtype=np.float32)
                amin, amax = float(arr.min()), float(arr.max())
                bits_info = img.info.get("bits")
                if not isinstance(bits_info, int):
                    tag_v2 = getattr(img, "tag_v2", None)
                    if tag_v2 is not None:
                        bits_tag = tag_v2.get(258)
                        if isinstance(bits_tag, tuple) and bits_tag:
                            bits_tag = bits_tag[0]
                        if isinstance(bits_tag, int):
                            bits_info = bits_tag

                if (
                    isinstance(bits_info, int)
                    and bits_info in (8, 16)
                    and amin >= 0.0
                    and amax > 1.0
                ):
                    arr = arr / float((1 << bits_info) - 1)
                    logger.debug(
                        "Float image '%s' normalized using explicit %d-bit metadata.",
                        path,
                        bits_info,
                    )
                elif amin < 0.0 or amax > 1.0:
                    logger.warning(
                        "Float image '%s' has range [%.6f, %.6f]; preserving absolute values "
                        "to avoid silent dynamic-range destruction.",
                        path, amin, amax,
                    )
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                logger.debug("Loaded %s in float mode", path)
                return arr.astype(np.float32, copy=False)

            # Standard 8-bit modes (also covers DDS decompressed output)
            if img.mode == "P":
                logger.debug("Converting palette image '%s' from P->RGBA", path)
                with img.convert("RGBA") as converted:
                    arr = np.asarray(converted, dtype=np.float32) / 255.0
            elif img.mode == "L":
                logger.debug("Converting grayscale image '%s' from L->RGB", path)
                with img.convert("RGB") as converted:
                    arr = np.asarray(converted, dtype=np.float32) / 255.0
            elif img.mode == "LA":
                logger.debug("Converting LA image '%s' from LA->RGBA", path)
                with img.convert("RGBA") as converted:
                    arr = np.asarray(converted, dtype=np.float32) / 255.0
            elif img.mode == "CMYK":
                logger.debug("Converting CMYK image '%s' to RGB", path)
                with img.convert("RGB") as converted:
                    arr = np.asarray(converted, dtype=np.float32) / 255.0
            else:
                logger.debug(
                    "Loading image '%s' using default ndarray path for mode %s",
                    path,
                    img.mode,
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0

            logger.debug("Finalizing image load for %s from mode %s", path, img.mode)

            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
                logger.debug("Expanded 2D array to 3 channels for %s", path)
            return arr.astype(np.float32, copy=False)
    except (ValueError, ImportError):
        raise
    except Exception as e:
        logger.error("Failed to open image '%s' (ext=%s): %s", path, ext, e)
        raise IOError(
            f"Failed to open image: {path}\n"
            f"  Format: {ext}, Error: {e}\n"
            f"  If this is a DDS file, ensure Pillow >= 9.0 is installed."
        ) from e


def save_image(arr: np.ndarray, path: str, quality: int = 95,
               bits: int = 8):
    """Save float32 [0,1] numpy array as image.

    Handles RGB, RGBA, and grayscale (2D).  Uses atomic write (temp file +
    ``os.replace``) to prevent truncated output on crash.

    Args:
        arr: float32 array in [0, 1] range.
        path: Output file path.
        quality: JPEG quality (1-100).
        bits: Output bit depth (8 or 16). 16-bit supported for PNG
              (grayscale via Pillow, RGB/RGBA via cv2 fallback).

    """
    arr = np.clip(arr, 0, 1)

    if arr.size == 0 or arr.ndim < 2:
        raise ValueError(
            f"Cannot save empty or degenerate array (shape={arr.shape}) to {path}"
        )

    ext = Path(path).suffix.lower()
    use_16bit = bits == 16 and ext == ".png"

    parent_dir = os.path.dirname(path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    # Atomic write: write to temp file, then os.replace to final path.
    # Keep original extension so Pillow/cv2 can infer the format.
    import threading as _th
    tmp_path = f"{path}.tmp.{os.getpid()}.{_th.get_ident()}{ext}"

    try:
        if use_16bit:
            arr_16 = np.round(arr * 65535.0).astype(np.uint16)
            if arr_16.ndim == 3 and arr_16.shape[-1] == 1:
                arr_16 = arr_16[:, :, 0]

            # Prefer cv2 for 16-bit PNG (grayscale/RGB/RGBA).
            try:
                import cv2 as _cv2

                if arr_16.ndim == 2:
                    png_data = arr_16
                elif arr_16.ndim == 3 and arr_16.shape[-1] == 4:
                    png_data = arr_16[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA
                elif arr_16.ndim == 3 and arr_16.shape[-1] >= 3:
                    png_data = arr_16[:, :, :3][:, :, ::-1]  # RGB -> BGR
                else:
                    raise ValueError(f"Unsupported shape for 16-bit PNG save: {arr_16.shape}")

                ok = _cv2.imwrite(tmp_path, np.ascontiguousarray(png_data))
                del png_data, arr_16
                if not ok:
                    raise IOError(f"cv2.imwrite failed for 16-bit PNG: {path}")
                os.replace(tmp_path, path)
                logger.debug(f"Saved: {path} (16bit)")
                return
            except ImportError:
                pass

            # Pillow fallback for grayscale only. Explicit mode avoids accidental
            # promotion to 32-bit signed integer mode.
            if arr_16.ndim == 2:
                byteorder = arr_16.dtype.byteorder
                if byteorder == ">":
                    mode = "I;16B"
                elif byteorder == "<":
                    mode = "I;16"
                elif byteorder == "=":
                    mode = "I;16" if sys.byteorder == "little" else "I;16B"
                else:
                    mode = "I;16"
                with Image.fromarray(arr_16, mode=mode) as img:
                    img.save(tmp_path)
                os.replace(tmp_path, path)
                logger.debug(f"Saved: {path} ({arr_16.shape}, 16bit grayscale)")
                return

            logger.warning(
                "cv2 unavailable for 16-bit multi-channel PNG; "
                f"falling back to 8-bit for {path}. "
                "Height/roughness/metalness maps will lose precision. "
                "Install opencv-python-headless to fix this."
            )
            # Raise an error so callers are aware of precision loss rather
            # than silently shipping degraded maps.
            raise ImportError(
                f"Cannot save 16-bit multi-channel PNG without cv2: {path}. "
                "Install opencv-python-headless: pip install opencv-python-headless"
            )

        # 8-bit path
        arr_out = np.round(arr * 255).astype(np.uint8)
        with Image.fromarray(arr_out) as img:
            if ext in (".jpg", ".jpeg"):
                if img.mode == "RGBA":
                    with img.convert("RGB") as converted:
                        converted.save(tmp_path, quality=quality)
                else:
                    img.save(tmp_path, quality=quality)
            elif ext == ".png":
                img.save(tmp_path, optimize=True)
            else:
                img.save(tmp_path)
        os.replace(tmp_path, path)
        logger.debug(f"Saved: {path} ({arr_out.shape}, 8bit)")
    finally:
        # Clean up temp file on any error
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (H, W, 3)."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        return arr[:, :, :3]
    if arr.shape[-1] == 1:
        return np.concatenate([arr] * 3, axis=-1)
    return arr


def extract_alpha(arr: np.ndarray) -> Optional[np.ndarray]:
    """Extract alpha channel if present."""
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return arr[:, :, 3]
    return None


def merge_alpha(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Merge RGB with alpha channel."""
    if alpha is None:
        return rgb
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(
            f"rgb must be HxWx3/4, got shape {rgb.shape}"
        )
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    h, w = rgb.shape[:2]
    if alpha.ndim == 2:
        alpha_2d = alpha
    elif alpha.ndim == 3 and alpha.shape[-1] == 1:
        alpha_2d = alpha[:, :, 0]
    else:
        raise ValueError(
            f"alpha must be HxW or HxWx1, got shape {alpha.shape}"
        )

    if alpha_2d.shape != (h, w):
        raise ValueError(
            f"alpha shape {alpha_2d.shape} does not match rgb shape {(h, w)}"
        )

    alpha_2d = np.clip(alpha_2d, 0, 1).astype(rgb.dtype, copy=False)
    return np.dstack([rgb, alpha_2d])


def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] values to linear RGB."""
    if np.isnan(arr).any():
        logger.warning("NaN detected in srgb_to_linear input; replacing with 0.0")
        arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(
        arr <= 0.04045,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    ).astype(np.float32, copy=False)


def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to sRGB [0,1]."""
    if np.isnan(arr).any():
        logger.warning("NaN detected in linear_to_srgb input; replacing with 0.0")
        arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(arr, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def luminance_bt709(arr: np.ndarray, assume_srgb: bool = False) -> np.ndarray:
    """Compute BT.709 luminance from an RGB-like array."""
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    rgb = ensure_rgb(arr).astype(np.float32, copy=False)
    if assume_srgb:
        rgb = srgb_to_linear(rgb)
    return (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    ).astype(np.float32, copy=False)
