"""Thread-safe checkpoint/resume system for crash recovery."""

import hashlib
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Set

from ..config import PipelineConfig

logger = logging.getLogger("asset_pipeline")
CHECKPOINT_SCHEMA_VERSION = 2


def config_fingerprint(config: "PipelineConfig") -> str:
    """Compute a short hash of pipeline configuration for change detection.

    Used by CheckpointManager to detect when config changes make cached
    results stale. Excludes fields that don't affect output quality
    (paths, workers, log level, checkpoint config itself).
    """
    import dataclasses as _dc
    d = _dc.asdict(config)
    # Remove fields that don't affect output quality
    for key in ('input_dir', 'output_dir', 'intermediate_dir',
                'comparison_dir', 'manifest_path', 'max_workers',
                'log_level', 'dry_run', 'checkpoint',
                'cleanup_intermediates'):
        d.pop(key, None)
    # Include configured device.  We intentionally do NOT call
    # resolve_device() here because it imports torch, which can trigger
    # a native DLL crash on Python 3.14 (MSVCP140.dll access violation
    # inside torch._load_dll_libraries).  The configured device string
    # ("auto"/"cuda"/"cpu") is sufficient for fingerprint purposes — if
    # the user changes the setting, the fingerprint changes too.
    d["device"] = config.device
    d["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    try:
        from .. import __version__ as pipeline_version
    except Exception:
        pipeline_version = "unknown"
    d["pipeline_version"] = pipeline_version
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _json_safe(obj):
    """Recursively keep only JSON-serializable values (str/int/float/bool/None/list/dict)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class CheckpointManager:
    """Thread-safe checkpoint manager for crash recovery and incremental runs."""

    def __init__(self, config: PipelineConfig):
        """Initialize checkpoint state and load existing data when available."""
        self.cfg = config.checkpoint
        self._config_fp = config_fingerprint(config)
        self._data: Dict = {}
        self._counter = 0
        self._lock = threading.RLock()
        self._last_save_error: Optional[str] = None
        self._consecutive_failures = 0

        self._cleanup_temp_files()

        if self.cfg.enabled and os.path.exists(self.cfg.checkpoint_path):
            self._load()

    def _cleanup_temp_files(self):
        """Remove orphaned temporary checkpoint files from previous crashes.

        Only removes temp files from the current PID or from PIDs that are
        no longer running, to avoid interfering with concurrent instances.
        """
        base = self.cfg.checkpoint_path
        parent = os.path.dirname(base) or "."
        prefix = os.path.basename(base) + ".tmp."
        current_pid = os.getpid()

        def _pid_alive(pid: int) -> bool:
            """Check if a process is still running."""
            if pid == current_pid:
                return True
            try:
                os.kill(pid, 0)
                return True
            except (OSError, PermissionError):
                return False

        try:
            for name in os.listdir(parent):
                if name.startswith(prefix):
                    # Try to extract PID from filename pattern: .tmp.<pid>.<tid>
                    parts = name[len(prefix):].split(".")
                    file_pid = None
                    if parts and parts[0].isdigit():
                        file_pid = int(parts[0])
                    # Only delete if PID is ours or the process is dead
                    if file_pid is None or file_pid == current_pid or not _pid_alive(file_pid):
                        tmp_path = os.path.join(parent, name)
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        except OSError:
            pass

    @staticmethod
    def _validate_schema(data: dict) -> bool:
        """Validate checkpoint JSON structure to fail fast on corrupt data.

        Returns True if valid, False if the structure is unusable.
        """
        if not isinstance(data, dict):
            return False
        completed = data.get("completed")
        if completed is not None:
            if not isinstance(completed, dict):
                return False
            for filepath, entry in completed.items():
                if not isinstance(filepath, str):
                    return False
                if not isinstance(entry, dict):
                    return False
                if "hash" in entry and not isinstance(entry["hash"], str):
                    return False
                if "phases" in entry and not isinstance(entry["phases"], list):
                    return False
                if "results" in entry and not isinstance(entry["results"], dict):
                    return False
        return True

    def _load(self):
        """Load existing checkpoint, invalidating if config changed."""
        try:
            with open(self.cfg.checkpoint_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

            if not self._validate_schema(self._data):
                logger.warning(
                    "Checkpoint file has invalid structure; discarding."
                )
                self._data = {}
                return

            saved_fp = self._data.get("config_fingerprint", "")
            if saved_fp and saved_fp != self._config_fp:
                count = len(self._data.get("completed", {}))
                logger.warning(
                    f"Pipeline config changed since last run "
                    f"(was {saved_fp}, now {self._config_fp}). "
                    f"Invalidating {count} cached results."
                )
                self._data = {}
                return
            count = len(self._data.get("completed", {}))
            logger.info(f"Loaded checkpoint: {count} completed assets")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            self._data = {}

    def save(self, _auto: bool = False):
        """Save checkpoint to disk (thread-safe, atomic write).

        Args:
            _auto: If True, this is an auto-save from mark_completed and
                   failures are logged but not raised. If False (explicit
                   save), OSError is re-raised after logging.

        """
        if not self.cfg.enabled:
            logger.debug("Skipping checkpoint save: checkpoint is disabled.")
            return
        with self._lock:
            os.makedirs(os.path.dirname(self.cfg.checkpoint_path) or ".", exist_ok=True)
            self._data["last_saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._data["config_fingerprint"] = self._config_fp
            tmp_path = (
                f"{self.cfg.checkpoint_path}.tmp."
                f"{os.getpid()}.{threading.get_ident()}"
            )
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.cfg.checkpoint_path)
                self._last_save_error = None
                self._consecutive_failures = 0
            except OSError as e:
                self._last_save_error = str(e)
                self._consecutive_failures += 1
                logger.error(f"Failed to save checkpoint: {e}")
                if not _auto:
                    raise
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def is_completed(self, filepath: str, file_hash: str, phases: List[str] = None) -> bool:
        """Check if an asset has already been fully processed (matching hash)."""
        if not self.cfg.enabled or not self.cfg.skip_completed:
            logger.debug(
                "Checkpoint completion check skipped for %s: enabled=%s, skip_completed=%s",
                filepath, self.cfg.enabled, self.cfg.skip_completed
            )
            return False
        with self._lock:
            completed = self._data.get("completed", {})
            entry = completed.get(filepath)
            if not entry:
                return False
            if entry.get("hash") != file_hash:
                return False
            if phases:
                done_phases = set(entry.get("phases", []))
                return all(p in done_phases for p in phases)
            return True

    def mark_completed(self, filepath: str, file_hash: str, phase: str,
                       result: dict = None):
        """Mark a phase as completed for an asset (thread-safe)."""
        if not self.cfg.enabled:
            logger.debug(
                "Skipping checkpoint write for %s (%s): checkpoint is disabled.",
                filepath, phase
            )
            return

        should_save = False
        with self._lock:
            if "completed" not in self._data:
                self._data["completed"] = {}
            if filepath not in self._data["completed"]:
                self._data["completed"][filepath] = {
                    "hash": file_hash,
                    "phases": [],
                    "results": {}
                }
            entry = self._data["completed"][filepath]
            entry["hash"] = file_hash
            if phase not in entry["phases"]:
                entry["phases"].append(phase)
            if result:
                entry["results"][phase] = _json_safe(result)

            self._counter += 1
            should_save = self._counter % self.cfg.save_interval == 0
            consecutive_failures = self._consecutive_failures

        # Save outside lock to avoid holding it during I/O
        if should_save:
            self.save(_auto=True)
            if consecutive_failures >= 3:
                logger.warning(
                    f"Checkpoint save has failed {consecutive_failures} "
                    f"consecutive times. Resume data may be lost. "
                    f"Last error: {self._last_save_error}"
                )

    def get_result(self, filepath: str, phase: str) -> Optional[dict]:
        """Retrieve cached result for an asset+phase."""
        with self._lock:
            completed = self._data.get("completed", {})
            entry = completed.get(filepath)
            if not entry:
                logger.debug("No cached result for %s (not recorded).", filepath)
                return None
            return entry.get("results", {}).get(phase)

    def prune(self, active_filepaths: Set[str]):
        """Remove checkpoint entries for files no longer in the input set."""
        if not self.cfg.enabled:
            logger.debug("Skipping checkpoint prune: checkpoint is disabled.")
            return
        stale_count = 0
        with self._lock:
            completed = self._data.get("completed", {})
            stale = [fp for fp in completed if fp not in active_filepaths]
            for fp in stale:
                del completed[fp]
            stale_count = len(stale)
        if stale_count:
            logger.info(f"Pruned {stale_count} stale checkpoint entries")
            self.save(_auto=True)

    def clear(self):
        """Clear all checkpoint data."""
        with self._lock:
            self._data = {}
            self._counter = 0
            try:
                os.remove(self.cfg.checkpoint_path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint file: {e}")
            self._cleanup_temp_files()
        logger.info("Checkpoint cleared")
