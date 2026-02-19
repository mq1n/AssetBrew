"""Texture classification by filename suffix patterns and content analysis."""

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import TextureType, TEXTURE_PATTERNS, GLOSS_PATTERNS, PipelineConfig

logger = logging.getLogger("asset_pipeline")


def _infer_content_bit_depth(img: "Image.Image") -> int | None:
    """Infer source bit depth for content normalization."""
    mode = str(getattr(img, "mode", ""))
    if mode in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
        return 8
    if mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        return 16

    bits_info = img.info.get("bits")
    if isinstance(bits_info, int) and bits_info > 0:
        return int(bits_info)

    tag_v2 = getattr(img, "tag_v2", None)
    if tag_v2 is not None:
        bits_tag = tag_v2.get(258)
        if isinstance(bits_tag, tuple) and bits_tag:
            bits_tag = bits_tag[0]
        if isinstance(bits_tag, int) and bits_tag > 0:
            return int(bits_tag)
    return None


def _normalize_content_array(arr_raw: np.ndarray, bit_depth: int | None) -> np.ndarray:
    """Normalize sampled image data to [0, 1] for stable heuristics."""
    arr = arr_raw.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr

    if np.issubdtype(arr_raw.dtype, np.integer):
        if bit_depth is not None and 1 <= bit_depth <= 32:
            max_value = float((1 << bit_depth) - 1)
        else:
            max_value = float(np.iinfo(arr_raw.dtype).max)
        if max_value > 0:
            arr = arr / max_value
        return np.clip(arr, 0.0, 1.0)

    arr_max = float(arr.max())
    if arr_max <= 1.0:
        return np.clip(arr, 0.0, 1.0)

    if bit_depth in (8, 16):
        scale = float((1 << bit_depth) - 1)
        if scale > 0:
            arr = arr / scale
        return np.clip(arr, 0.0, 1.0)

    # Unknown float payload. Keep conservative legacy behavior.
    scale = 255.0 if arr_max <= 255.0 else 65535.0
    return np.clip(arr / scale, 0.0, 1.0)


def classify_texture(filepath: str) -> TextureType:
    """Classify texture type based on filename suffix patterns.

    Uses longest-match suffix strategy to avoid false positives from
    short patterns matching mid-word substrings.
    """
    name = Path(filepath).stem.lower()
    best_type = TextureType.UNKNOWN
    best_len = 0
    for tex_type, patterns in TEXTURE_PATTERNS.items():
        for pattern in patterns:
            if name.endswith(pattern) and len(pattern) > best_len:
                best_len = len(pattern)
                best_type = tex_type
    return best_type


def classify_texture_by_content(img: "Image.Image") -> TextureType:
    """Content-based texture classification fallback.

    Used when filename-based classification returns UNKNOWN. Analyzes
    pixel statistics to distinguish normal maps from other texture types.

    Conservative by design -- prefers DIFFUSE (safe default for the PBR
    pipeline) over more specific types to avoid misrouting assets like
    desaturated character textures into height-map processing.

    Heuristics:
    - Normal maps: RGB centered near (0.5, 0.5, 1.0), high B mean,
      low B variance, and characteristic normal-map channel statistics.
    - True grayscale (L/LA mode): Classified as HEIGHT only when the
      source image is natively single-channel.
    - Everything else (including near-grayscale RGB): Assumed DIFFUSE.
    """
    try:
        # Classify natively single-channel images as HEIGHT
        # (before thumbnail, which can change mode)
        if img.mode in ("L", "LA"):
            logger.debug("Content-based classification: single-channel mode %s -> HEIGHT", img.mode)
            return TextureType.HEIGHT

        # Sample at most 512x512 for speed — resize directly to avoid
        # allocating a full-resolution copy via img.copy().
        sample = img.resize(
            (min(img.width, 512), min(img.height, 512)), Image.LANCZOS
        )
        arr_raw = np.asarray(sample)
        bit_depth = _infer_content_bit_depth(img)

        if arr_raw.ndim == 2:
            # Pure grayscale -> likely height/displacement
            logger.debug("Content-based classification: 2D ndarray -> HEIGHT")
            return TextureType.HEIGHT

        if arr_raw.shape[2] < 3:
            logger.debug(
                "Content-based classification: low channel count %d -> HEIGHT",
                arr_raw.shape[2],
            )
            return TextureType.HEIGHT

        arr = _normalize_content_array(arr_raw, bit_depth)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        b_std = b.std()

        # Normal map detection: blue channel high (~0.75-1.0),
        # red and green centered around 0.5 with moderate spread
        if (b_mean > 0.7
                and abs(r_mean - 0.5) < 0.15
                and abs(g_mean - 0.5) < 0.15
                and b_std < 0.15):
            logger.debug(
                "Content-based classification: normal map detected "
                "(r_mean=%.4f, g_mean=%.4f, b_mean=%.4f, b_std=%.4f)",
                r_mean,
                g_mean,
                b_mean,
                b_std,
            )
            return TextureType.NORMAL

        # Default: treat as diffuse (safe for PBR pipeline).
        # Near-grayscale RGB images (desaturated diffuse, hair, skin)
        # are intentionally kept as DIFFUSE to avoid misrouting.
        logger.debug(
            "Content-based classification fallback: mode=%s -> DIFFUSE "
            "(r_mean=%.4f, g_mean=%.4f, b_mean=%.4f, b_std=%.4f)",
            img.mode,
            r_mean,
            g_mean,
            b_mean,
            b_std,
        )
        return TextureType.DIFFUSE

    except Exception as e:
        logger.debug("Content-based classification failed: %s", e)
        return TextureType.UNKNOWN


def is_gloss_texture(filepath: str) -> bool:
    """Check if texture is a glossiness map (needs inversion to roughness)."""
    name = Path(filepath).stem.lower()
    for pattern in GLOSS_PATTERNS:
        if name.endswith(pattern):
            return True
    return False


def detect_material_category(filepath: str) -> str:
    """Guess material category from filepath/filename.

    Uses word-boundary detection: a category matches only when it is
    surrounded by non-alpha characters (underscores, hyphens, path separators,
    digits, or string boundaries).  E.g. "ironsword" does NOT match "iron",
    but "iron_sword" and "iron-plate" do.
    """
    path_lower = filepath.lower()
    # Sorted longest-first to match more specific categories first
    categories = [
        "cobblestone", "concrete", "leather", "plastic", "ceramic",
        "copper", "silver", "fabric", "rubber", "glass",
        "brick", "metal", "stone", "grass", "paint", "cloth",
        "iron", "gold", "wood", "rock",
        "tile", "sand", "dirt", "skin",
    ]
    for cat in categories:
        # Match category at a word boundary (non-alpha on both sides)
        if re.search(r'(?<![a-z])' + re.escape(cat) + r'(?![a-z])', path_lower):
            return cat
    return "default"


def is_likely_tileable(filepath: str, config: PipelineConfig) -> bool:
    """Return whether the filepath matches configured tileable asset patterns."""
    path_lower = filepath.lower()
    for pattern in config.upscale.tiling_asset_patterns:
        if re.search(r'(?<![a-z])' + re.escape(pattern) + r'(?![a-z])', path_lower):
            return True
    return False


def is_hero_asset(filepath: str, config: PipelineConfig) -> bool:
    """Return whether the filepath matches configured hero asset patterns."""
    path_lower = filepath.lower()
    for pattern in config.upscale.hero_asset_patterns:
        if re.search(r'(?<![a-z])' + re.escape(pattern) + r'(?![a-z])', path_lower):
            return True
    return False
