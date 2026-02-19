"""Output and intermediate path helpers."""

import os
from pathlib import Path, PurePosixPath


def _normalize_rel_asset_path(input_rel_path: str) -> Path:
    """Normalize a relative asset path to a canonical, traversal-free form."""
    raw = str(input_rel_path).replace("\\", "/")
    p = PurePosixPath(raw)
    if p.is_absolute():
        raise ValueError(f"Asset path must be relative, got absolute path: {input_rel_path}")

    parts = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            else:
                raise ValueError(
                    f"Asset path escapes root via '..': {input_rel_path}"
                )
            continue
        parts.append(part)

    if not parts:
        raise ValueError(f"Asset path is empty after normalization: {input_rel_path}")
    return Path(*parts)


def get_output_path(input_rel_path: str, output_dir: str,
                    suffix: str = "", ext: str = None) -> str:
    """Return output path for an asset variant."""
    p = _normalize_rel_asset_path(input_rel_path)
    stem = p.stem + suffix
    extension = ext or p.suffix
    parent = "" if str(p.parent) == "." else str(p.parent)
    return os.path.join(output_dir, parent, stem + extension)


def get_intermediate_path(input_rel_path: str, phase: str,
                          intermediate_dir: str, suffix: str = "",
                          ext: str = None) -> str:
    """Return phase-specific intermediate path."""
    p = _normalize_rel_asset_path(input_rel_path)
    stem = p.stem + suffix
    extension = ext or p.suffix
    parent = "" if str(p.parent) == "." else str(p.parent)
    return os.path.join(intermediate_dir, phase, parent, stem + extension)
