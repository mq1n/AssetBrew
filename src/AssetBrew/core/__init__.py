"""Core utilities -- re-exports all public symbols for convenience."""

from .records import AssetRecord
from .io import (
    load_image,
    save_image,
    ensure_rgb,
    extract_alpha,
    merge_alpha,
    srgb_to_linear,
    linear_to_srgb,
    luminance_bt709,
)
from .classify import (
    classify_texture, classify_texture_by_content,
    is_gloss_texture, detect_material_category,
    is_likely_tileable, is_hero_asset,
)
from .tiling import pad_for_tiling, crop_from_padded
from .gpu import GPUMonitor
from .checkpoint import CheckpointManager, config_fingerprint
from .scanning import (
    file_hash,
    file_hash_sha256,
    file_hash_md5,
    scan_assets,
    save_manifest,
    load_manifest,
)
from .paths import get_output_path, get_intermediate_path
from .logging import setup_logging
from .compat import tqdm

__all__ = [
    "AssetRecord",
    "load_image", "save_image", "ensure_rgb", "extract_alpha", "merge_alpha",
    "srgb_to_linear", "linear_to_srgb", "luminance_bt709",
    "classify_texture", "classify_texture_by_content",
    "is_gloss_texture", "detect_material_category",
    "is_likely_tileable", "is_hero_asset",
    "pad_for_tiling", "crop_from_padded",
    "GPUMonitor",
    "CheckpointManager", "config_fingerprint",
    "file_hash", "file_hash_sha256", "file_hash_md5",
    "scan_assets", "save_manifest", "load_manifest",
    "get_output_path", "get_intermediate_path",
    "setup_logging",
    "tqdm",
]
