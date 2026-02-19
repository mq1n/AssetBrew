"""Non-GUI helpers used by the desktop UI.

This module intentionally avoids PyQt imports so it can be unit tested
in headless environments (CI, servers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


MAP_OPTIONS: List[Tuple[str, str]] = [
    ("Base", ""),
    ("Albedo", "_albedo"),
    ("Roughness", "_roughness"),
    ("Gloss", "_gloss"),
    ("Metalness", "_metalness"),
    ("AO", "_ao"),
    ("Normal", "_normal"),
    ("Height", "_height"),
    ("ORM", "_orm"),
    ("Emissive", "_emissive"),
    ("Emissive Mask", "_emissive_mask"),
    ("Env Mask", "_envmask"),
    ("Zone Mask", "_zones"),
]

OUTPUT_EXT_PRIORITY: List[str] = [
    ".png",
    ".dds",
    ".tga",
    ".ktx2",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
]

RESULT_MAP_PATHS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "Base": (
        ("orm", "diffuse_alpha_packed"),
        ("seam_repair", "upscaled_repaired"),
        ("upscale", "upscaled"),
    ),
    "Albedo": (
        ("color_grading", "graded"),
        ("seam_repair", "albedo_repaired"),
        ("color_consistency", "corrected"),
        ("pbr", "albedo"),
    ),
    "Roughness": (
        ("specular_aa", "roughness_aa"),
        ("pbr", "roughness"),
    ),
    "Gloss": (
        ("pbr", "gloss"),
    ),
    "Metalness": (
        ("pbr", "metalness"),
    ),
    "AO": (
        ("pbr", "ao"),
    ),
    "Normal": (
        ("detail_map", "normal_detailed"),
        ("normal", "normal"),
    ),
    "Height": (
        ("pom", "height_refined"),
        ("normal", "height"),
    ),
    "ORM": (
        ("orm", "orm"),
    ),
    "Emissive": (
        ("emissive", "emissive"),
    ),
    "Emissive Mask": (
        ("emissive", "emissive_mask"),
    ),
    "Env Mask": (
        ("reflection_mask", "env_mask"),
    ),
    "Zone Mask": (
        ("pbr", "zone_mask"),
    ),
}


def value_to_text(value: Any) -> str:
    """Render a config value as editable text."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        dumped = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
        if dumped:
            return dumped
        return "[]" if isinstance(value, list) else "{}"
    return str(value)


def parse_typed_value(text: str, template: Any) -> Any:
    """Parse a text value into the same primitive shape as ``template``."""
    raw = text.strip()

    if template is None:
        lowered = raw.lower()
        if raw == "" or lowered in {"null", "none", "~"}:
            return None
        return yaml.safe_load(text)

    if isinstance(template, bool):
        lowered = raw.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean: {text!r}")

    if isinstance(template, int) and not isinstance(template, bool):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid integer: {text!r}") from exc

    if isinstance(template, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid float: {text!r}") from exc

    if isinstance(template, str):
        return text

    if isinstance(template, list):
        parsed = yaml.safe_load(text) if raw else []
        if parsed is None:
            parsed = []
        if not isinstance(parsed, list):
            raise ValueError("value must parse to a list")
        return parsed

    if isinstance(template, dict):
        parsed = yaml.safe_load(text) if raw else {}
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("value must parse to a mapping")
        return parsed

    return yaml.safe_load(text)


def set_dataclass_path(obj: Any, path: Sequence[str], value: Any) -> None:
    """Set nested dataclass attribute by path."""
    target = obj
    for key in path[:-1]:
        target = getattr(target, key)
    setattr(target, path[-1], value)


def find_output_map_file(
    rel_path: str,
    output_dir: str,
    suffix: str,
    ext_priority: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Find an output map file for ``rel_path`` using preferred extension order."""
    rel = Path(rel_path)
    stem = rel.stem + suffix
    parent = Path(output_dir) / rel.parent
    for ext in list(ext_priority or OUTPUT_EXT_PRIORITY):
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def _nested_get(data: Dict[str, Any], path: Sequence[str]) -> Optional[str]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, str) else None


def _first_existing_nested_path(
    data: Dict[str, Any],
    nested_paths: Sequence[Sequence[str]],
) -> Optional[str]:
    """Return the first existing on-disk path from a sequence of nested keys."""
    for path in nested_paths:
        candidate = _nested_get(data, path)
        if candidate and Path(candidate).exists():
            return candidate
    return None


def resolve_map_paths(
    rel_path: str,
    result_entry: Dict[str, Any],
    output_dir: str,
    map_options: Optional[Sequence[Tuple[str, str]]] = None,
    ext_priority: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Resolve available map files for an asset.

    Precedence:
    1. phase result paths when present and existing,
    2. fallback files in output directory based on suffix convention.
    """
    paths: Dict[str, str] = {}
    options = list(map_options or MAP_OPTIONS)

    for label, _suffix in options:
        nested_paths = RESULT_MAP_PATHS.get(label, ())
        candidate = _first_existing_nested_path(result_entry, nested_paths)
        if candidate:
            paths[label] = candidate

    for label, suffix in options:
        if label in paths:
            continue
        fallback = find_output_map_file(rel_path, output_dir, suffix, ext_priority=ext_priority)
        if fallback:
            paths[label] = fallback

    return paths
