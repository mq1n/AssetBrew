"""Shared test fixtures."""

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, save_image


_WORKSPACE_TMP_ROOT = Path(".tmp") / "pytest_tmp"
_ORIGINAL_MKDTEMP = tempfile.mkdtemp
_ORIGINAL_TEMPORARY_DIRECTORY = tempfile.TemporaryDirectory
_ORIGINAL_TEMPDIR = tempfile.tempdir
_ORIGINAL_TMP_ENV = {k: os.environ.get(k) for k in ("TMPDIR", "TMP", "TEMP")}
_TEMPFILE_COMPAT_SHIM_ACTIVE = False


def _probe_native_tempfile_writable() -> bool:
    """Return True when native tempfile directories are writable in this runtime."""
    path = None
    try:
        path = _ORIGINAL_MKDTEMP()
        probe_file = os.path.join(path, "__probe__.txt")
        with open(probe_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe_file)
        return True
    except Exception:
        return False
    finally:
        if path:
            shutil.rmtree(path, ignore_errors=True)


def _workspace_mkdtemp(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
):
    """Create a writable temp dir under a workspace-local root."""
    root = Path(dir) if dir else _WORKSPACE_TMP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    safe_prefix = prefix or "tmp"
    safe_suffix = suffix or ""
    while True:
        candidate = root / f"{safe_prefix}{uuid4().hex}{safe_suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return str(candidate)
        except FileExistsError:
            continue


class _WorkspaceTemporaryDirectory:
    """TemporaryDirectory shim for sandboxed environments with broken mkdtemp ACLs."""

    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
        *,
        delete: bool = True,
    ):
        self.name = _workspace_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def cleanup(self):
        if not self._delete:
            return
        if not self.name:
            return
        try:
            shutil.rmtree(
                self.name,
                ignore_errors=bool(self._ignore_cleanup_errors),
            )
        except FileNotFoundError:
            return


def _activate_tempfile_compat_shim():
    """Patch tempfile only when native temp dirs are not writable."""
    global _TEMPFILE_COMPAT_SHIM_ACTIVE
    _WORKSPACE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_abs = str(_WORKSPACE_TMP_ROOT.resolve())
    os.environ["TMPDIR"] = tmp_abs
    os.environ["TMP"] = tmp_abs
    os.environ["TEMP"] = tmp_abs
    tempfile.tempdir = tmp_abs
    tempfile.mkdtemp = _workspace_mkdtemp
    tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory
    _TEMPFILE_COMPAT_SHIM_ACTIVE = True


def pytest_sessionstart(session):  # noqa: ARG001
    """Use native tempfile behavior unless runtime ACLs make it unusable."""
    if not _probe_native_tempfile_writable():
        _activate_tempfile_compat_shim()


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Restore tempfile globals if a runtime shim was installed."""
    if _TEMPFILE_COMPAT_SHIM_ACTIVE:
        tempfile.mkdtemp = _ORIGINAL_MKDTEMP
        tempfile.TemporaryDirectory = _ORIGINAL_TEMPORARY_DIRECTORY
        tempfile.tempdir = _ORIGINAL_TEMPDIR
        for key, value in _ORIGINAL_TMP_ENV.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def default_config():
    return PipelineConfig()


@pytest.fixture
def sample_record():
    return AssetRecord(
        filepath="brick_diff.png",
        filename="brick_diff.png",
        texture_type="diffuse",
        original_width=256,
        original_height=256,
        channels=3,
        has_alpha=False,
        is_tileable=True,
        is_hero=False,
        material_category="brick",
        file_size_kb=120.5,
        file_hash="abc123", is_gloss=False,
    )


def save_test_png(path, width=64, height=64, channels=3):
    """Create a random test PNG image."""
    arr = np.random.rand(height, width, channels).astype(np.float32)
    save_image(arr, path)
    return arr
