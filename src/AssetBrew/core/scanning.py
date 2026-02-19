"""Asset scanning, manifest I/O, and file hashing."""

import csv
import dataclasses
import hashlib
import logging
import os
import threading
import warnings
from pathlib import Path
from typing import List

from PIL import Image

from ..config import PipelineConfig, TextureType
from .records import AssetRecord
from .classify import (
    classify_texture, classify_texture_by_content,
    is_gloss_texture, detect_material_category,
    is_likely_tileable, is_hero_asset,
)
from .io import temporary_max_image_pixels

logger = logging.getLogger("asset_pipeline")


def file_hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash from in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for change detection.

    On I/O failure, returns a unique non-repeatable sentinel so that the
    checkpoint system never considers the file "completed" — this forces
    re-processing on subsequent runs (safe fallback).
    """
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Failed to hash %s: %s", filepath, exc)
        import uuid
        return f"unhashable-{uuid.uuid4().hex}"


def file_hash_sha256(filepath: str) -> str:
    """Explicit SHA-256-named helper for change detection."""
    return file_hash(filepath)


def file_hash_md5(filepath: str) -> str:
    """Deprecated backward-compat shim; this function returns SHA-256."""
    warnings.warn(
        "file_hash_md5() is deprecated and returns SHA-256, not MD5. "
        "Use file_hash() or file_hash_sha256().",
        DeprecationWarning,
        stacklevel=2,
    )
    return file_hash(filepath)


def scan_assets(input_dir: str, config: PipelineConfig) -> List[AssetRecord]:
    """Scan input directory and build asset records with file hashes."""
    records = []
    supported = set(config.supported_formats)
    input_root_real = os.path.realpath(input_dir)

    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext not in supported:
                continue

            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, input_dir)
            real_fpath = os.path.realpath(fpath)

            # Guard against symlink/path escapes outside input_dir.
            try:
                if os.path.commonpath([input_root_real, real_fpath]) != input_root_real:
                    logger.warning(
                        "Skipping file outside input root via symlink/path traversal: "
                        f"{fpath}"
                    )
                    continue
            except ValueError:
                logger.warning(
                    "Skipping file with incompatible path root: "
                    f"{fpath}"
                )
                continue

            try:
                with temporary_max_image_pixels(config.max_image_pixels):
                    with Image.open(fpath) as img:
                        w, h = img.size

                        # Memory guard — check BEFORE decode to avoid OOM
                        if config.max_image_pixels > 0 and w * h > config.max_image_pixels:
                            logger.warning(
                                f"Skipping {fname}: {w}x{h} = {w * h:,} pixels "
                                f"exceeds max_image_pixels ({config.max_image_pixels:,})"
                            )
                            continue

                        # Full decode deferred until content classification
                        # needs it. Header fields (size, mode, bands) are
                        # available without load().

                        min_dim = max(int(getattr(config, "min_texture_dim", 1)), 1)
                        if w < min_dim or h < min_dim:
                            logger.warning(
                                f"Skipping {fname}: dimensions {w}x{h} are below "
                                f"minimum supported size ({min_dim}x{min_dim})"
                            )
                            continue

                        channels = len(img.getbands())
                        has_alpha = _has_effective_alpha(img)
                        tex_type = classify_texture(fpath)
                        if tex_type == TextureType.UNKNOWN:
                            img.load()  # full decode needed for content analysis
                            tex_type = classify_texture_by_content(img)
                            if tex_type != TextureType.UNKNOWN:
                                logger.debug(
                                    f"Content-classified {fname} as {tex_type.value}"
                                )

                # Compute hash/size after image validation succeeds
                # to avoid wasted I/O on corrupt files.
                fhash = file_hash(fpath)
                fsize = os.path.getsize(fpath) / 1024.0

                mat_cat = detect_material_category(fpath)
                is_tile = is_likely_tileable(fpath, config)
                is_hero_flag = is_hero_asset(fpath, config)
                gloss_flag = is_gloss_texture(fpath)

                records.append(AssetRecord(
                    filepath=rel_path,
                    filename=fname,
                    texture_type=tex_type.value,
                    original_width=w,
                    original_height=h,
                    channels=channels,
                    has_alpha=has_alpha,
                    is_tileable=is_tile,
                    is_hero=is_hero_flag,
                    material_category=mat_cat,
                    file_size_kb=round(fsize, 1),
                    file_hash=fhash,
                    is_gloss=gloss_flag,
                ))
            except (OSError, IOError, ValueError, Image.DecompressionBombError) as e:
                logger.warning("Failed to read %s: %s", fpath, e, exc_info=True)
            except Exception as e:
                logger.error(
                    "Unexpected error reading %s: %s", fpath, e, exc_info=True
                )

    logger.info(f"Scanned {len(records)} assets from {input_dir}")
    return records


def _alpha_full_value(mode: str) -> int | None:
    """Return the numeric value for fully opaque alpha in a PIL alpha mode."""
    if mode == "1":
        return 1
    if mode in {"L", "P"}:
        return 255
    if mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        return 65535
    return None


def _has_effective_alpha(img: Image.Image) -> bool:
    """Return True only when alpha channel contains non-opaque pixels."""
    bands = img.getbands()
    if "A" not in bands:
        return False
    try:
        alpha = img.getchannel("A")
        try:
            extrema = alpha.getextrema()
            if not isinstance(extrema, tuple) or len(extrema) != 2:
                return True
            min_a, max_a = extrema
            if min_a is None or max_a is None:
                return True

            full = _alpha_full_value(alpha.mode)
            if full is not None:
                return int(min_a) < full or int(max_a) < full
            # Unknown alpha mode: treat as effective alpha unless the range is
            # exactly a single constant value.
            return int(min_a) != int(max_a)
        finally:
            alpha.close()
    except Exception:
        # Conservative fallback: if anything goes wrong, assume alpha is used.
        return True


def save_manifest(records: List[AssetRecord], path: str):
    """Write asset records to CSV manifest file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(AssetRecord.__dataclass_fields__.keys())
    import uuid as _uuid
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{_uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    logger.info(f"Manifest saved: {path} ({len(records)} entries)")


def load_manifest(path: str) -> List[AssetRecord]:
    """Load asset records from a CSV manifest file."""
    records = []

    def _parse_bool(value, field_name: str, row_idx: int) -> bool:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(
            f"field '{field_name}' has invalid boolean value '{value}' at row {row_idx}"
        )

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                try:
                    required = [
                        "filepath",
                        "filename",
                        "texture_type",
                        "original_width",
                        "original_height",
                        "channels",
                        "has_alpha",
                        "is_tileable",
                        "is_hero",
                        "material_category",
                        "file_size_kb",
                        "file_hash",
                    ]
                    missing = [name for name in required if name not in row or row[name] is None]
                    if missing:
                        raise KeyError(f"missing required columns: {', '.join(missing)}")

                    row["original_width"] = int(row["original_width"])
                    row["original_height"] = int(row["original_height"])
                    row["channels"] = int(row["channels"])
                    row["has_alpha"] = _parse_bool(row["has_alpha"], "has_alpha", row_idx)
                    row["is_tileable"] = _parse_bool(
                        row["is_tileable"], "is_tileable", row_idx
                    )
                    row["is_hero"] = _parse_bool(row["is_hero"], "is_hero", row_idx)
                    row["is_gloss"] = _parse_bool(
                        row.get("is_gloss", "false"), "is_gloss", row_idx
                    )
                    row["file_size_kb"] = float(row["file_size_kb"])
                    # Filter to known fields for forward compatibility
                    # with manifests written by newer pipeline versions.
                    known_fields = {
                        f.name for f in dataclasses.fields(AssetRecord)
                    }
                    filtered_row = {
                        k: v for k, v in row.items() if k in known_fields
                    }
                    records.append(AssetRecord(**filtered_row))
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse manifest '{path}' at row {row_idx}: {e}"
                    ) from e
    except OSError as e:
        raise OSError(f"Failed to read manifest '{path}': {e}") from e
    except csv.Error as e:
        raise ValueError(f"Malformed CSV manifest '{path}': {e}") from e
    return records
