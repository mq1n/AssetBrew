"""Provide package metadata and shared paths for `AssetBrew`."""

import logging as _logging
import os as _os
from pathlib import Path as _Path

__version__ = "1.2.0"
_logger = _logging.getLogger("asset_pipeline")


def _bin_dir_candidates():
    env = _os.environ.get("AssetBrew_BIN_DIR")
    if env:
        yield _Path(env).expanduser()

    pkg_dir = _Path(__file__).resolve().parent
    # Wheel/package-data layout (if bundled).
    yield pkg_dir / "bin"
    # Editable/repo layout: src/AssetBrew -> project_root/bin.
    yield pkg_dir.parent.parent / "bin"
    # Working-directory fallback for external deployments.
    yield _Path.cwd() / "bin"


def _resolve_bin_dir() -> _Path:
    seen = []
    for candidate in _bin_dir_candidates():
        seen.append(str(candidate))
        if candidate.is_dir():
            return candidate
    # Keep a deterministic fallback even when missing.
    fallback = _Path(__file__).resolve().parent / "bin"
    _logger.warning(
        "No valid tool directory found under AssetBrew_BIN_DIR/package/repo/cwd "
        "(checked: %s). Falling back to %s (tools may be unavailable).",
        ", ".join(seen),
        fallback,
    )
    return fallback


BIN_DIR = _resolve_bin_dir()


def _check_onnxruntime_conflict():
    """Warn if both onnxruntime and onnxruntime-gpu are installed."""
    try:
        import importlib.metadata
        installed = {
            d.metadata["Name"].lower()
            for d in importlib.metadata.distributions()
        }
        has_cpu = "onnxruntime" in installed
        has_gpu = "onnxruntime-gpu" in installed
        if has_cpu and has_gpu:
            import warnings
            warnings.warn(
                "Both 'onnxruntime' and 'onnxruntime-gpu' are installed. "
                "This causes conflicts. Uninstall one: "
                "pip uninstall onnxruntime  # if using GPU",
                RuntimeWarning,
                stacklevel=2,
            )
    except Exception:
        pass


_onnxruntime_checked = False


def _lazy_check_onnxruntime():
    global _onnxruntime_checked
    if not _onnxruntime_checked:
        _check_onnxruntime_conflict()
        _onnxruntime_checked = True


__all__ = ["__version__", "BIN_DIR"]
