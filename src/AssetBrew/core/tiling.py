"""Tiling helpers for seamless texture preservation."""

from typing import Tuple
import logging

import numpy as np

logger = logging.getLogger("asset_pipeline.tiling")


def _resize_float_with_pillow(
    arr: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Resize float image with Pillow when cv2 is unavailable."""
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).LANCZOS
    src = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    if src.ndim == 2:
        with Image.fromarray(src, mode="F") as img:
            with img.resize((target_w, target_h), resample=resample) as resized:
                return np.asarray(resized, dtype=np.float32)

    channels = []
    for idx in range(src.shape[2]):
        with Image.fromarray(src[:, :, idx], mode="F") as img:
            with img.resize((target_w, target_h), resample=resample) as resized:
                channels.append(np.asarray(resized, dtype=np.float32))
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def pad_for_tiling(arr: np.ndarray, pad_fraction: float = 0.25) -> Tuple[np.ndarray, dict]:
    """Pad image with wrapped borders to preserve seamless tiling after processing.

    Uses np.pad with "wrap" mode instead of a full 3x3 tile, reducing memory
    from ~9x to ~1.5x of the original image.

    Returns (padded_image, padding_info) where padding_info stores exact
    pixel counts for correct cropping later regardless of scale.
    """
    h, w = arr.shape[:2]
    pad_h = int(round(h * pad_fraction))
    pad_w = int(round(w * pad_fraction))

    # Use wrap padding (equivalent to tiling) with lower memory than np.tile.
    if arr.ndim == 3:
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="wrap")
    else:
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="wrap")

    info = {
        "original_h": h,
        "original_w": w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "padded_h": padded.shape[0],
        "padded_w": padded.shape[1],
    }
    return padded, info


def crop_from_padded(arr: np.ndarray, pad_info: dict, scale: int) -> np.ndarray:
    """Crop the center region from a padded+processed image.

    Uses exact pixel math based on the pad_info from pad_for_tiling.

    Args:
        arr: The processed (padded + upscaled) image
        pad_info: Dict from pad_for_tiling with original dims and pad amounts
        scale: The scale factor applied during processing (e.g. 4 for 4x upscale)

    """
    target_h = pad_info["original_h"] * scale
    target_w = pad_info["original_w"] * scale
    crop_top = pad_info["pad_h"] * scale
    crop_left = pad_info["pad_w"] * scale

    # Clamp to actual array bounds
    actual_h, actual_w = arr.shape[:2]
    end_h = min(crop_top + target_h, actual_h)
    end_w = min(crop_left + target_w, actual_w)

    # .copy() breaks the view into the full padded array so the caller
    # does not inadvertently keep the (much larger) padded buffer alive.
    cropped = arr[crop_top:end_h, crop_left:end_w].copy()

    if cropped.size == 0:
        logger.warning(
            "crop_from_padded produced empty crop (input shape=%s, expected=%s). "
            "Returning zero-filled array.",
            arr.shape, (target_h, target_w),
        )
        channels = arr.shape[2:] if arr.ndim > 2 else ()
        return np.zeros((target_h, target_w) + channels, dtype=arr.dtype)

    # If the crop is slightly off due to rounding, resize to exact target.
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        logger.debug(
            "Cropping mismatch (actual %dx%d, expected %dx%d); resizing mip seam crop.",
            cropped.shape[0], cropped.shape[1], target_h, target_w
        )
        src = np.clip(cropped, 0, 1).astype(np.float32, copy=False)
        try:
            import cv2 as _cv2

            resized = _cv2.resize(
                src,
                (target_w, target_h),
                interpolation=_cv2.INTER_LANCZOS4
            )
        except ImportError:
            logger.warning(
                "OpenCV unavailable in crop_from_padded; "
                "using Pillow Lanczos fallback."
            )
            resized = _resize_float_with_pillow(src, target_w, target_h)
        cropped = np.clip(resized, 0, 1).astype(np.float32, copy=False)

    return cropped
