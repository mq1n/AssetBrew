"""Orchestrate all DDS preprocess pipeline phases end-to-end.

`AssetPipeline` coordinates scanning, processing phases, output assembly,
validation, and result persistence.
"""

import logging
import os
import json
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Iterable, List, Optional

import numpy as np
from PIL import Image

from .config import PipelineConfig, TextureType
from .core import (
    AssetRecord, scan_assets, save_manifest,
    get_output_path, CheckpointManager, GPUMonitor,
    load_image, save_image, extract_alpha, merge_alpha, tqdm,
)

# Phase processor imports are deferred to the phase methods that need them,
# so the pipeline can start (scan, dry-run) even when heavy dependencies
# like cv2, torch, or onnxruntime are not installed.


logger = logging.getLogger("asset_pipeline")

# Inter-phase data dependencies: phase -> list of phases it needs data from.
_PHASE_DEPS = {
    "upscale": [],
    "pbr": ["upscale"],
    "normal": ["upscale", "pbr"],
    "pom": ["normal"],
    "mipmap": ["upscale"],
    "postprocess": ["pbr", "normal"],
    "validate": [],
}

# Known DXGI format IDs that encode sRGB gamma:
# 72 = DXGI_FORMAT_BC1_UNORM_SRGB
# 75 = DXGI_FORMAT_BC2_UNORM_SRGB
# 78 = DXGI_FORMAT_BC3_UNORM_SRGB
# 99 = DXGI_FORMAT_BC7_UNORM_SRGB
_DXGI_SRGB_FORMATS = {72, 75, 78, 99}

# DDS codec mapping for container-level validation.
_DDS_CODEC_FROM_DXGI = {
    71: "bc1", 72: "bc1",
    74: "bc2", 75: "bc2",
    77: "bc3", 78: "bc3",
    80: "bc4", 81: "bc4",
    83: "bc5", 84: "bc5",
    95: "bc6h", 96: "bc6h",
    98: "bc7", 99: "bc7",
}

_DDS_CODEC_FROM_FOURCC = {
    b"DXT1": "bc1",
    b"DXT3": "bc2",
    b"DXT5": "bc3",
    b"ATI1": "bc4",
    b"BC4U": "bc4",
    b"BC4S": "bc4",
    b"ATI2": "bc5",
    b"BC5U": "bc5",
    b"BC5S": "bc5",
}


class PhaseTimeoutError(RuntimeError):
    """Raised when a phase exceeds timeout and the run should abort."""


class PipelineCancelledError(RuntimeError):
    """Raised when a user-requested pipeline cancellation is observed."""


class PhaseFailureThresholdError(RuntimeError):
    """Raised when per-phase failure ratio exceeds configured abort threshold."""


def _get_version() -> str:
    """Read version from package."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def _get_output_ext(record) -> str:
    """Return primary output extension matching the input file's format."""
    return Path(record.filepath).suffix.lower()


class AssetPipeline:
    """Master pipeline orchestrator.

    Phases:
    0. Scan & classify
    1. AI Upscaling
    2. PBR Material Conversion
    3. Normal/Height Map Generation
    4. POM Height Map Refinement
    5. Post-processing (seam repair, grading, packing, utility maps)
    6. Mipmap Generation
    7. Validation & Comparison
    """

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """Initialize pipeline state, checkpoints, and lazy phase processors."""
        self.config = config
        # Ensure runtime fixups (e.g. disabling half_precision on CPU)
        # are applied regardless of entry point (CLI, UI, or SDK).
        config.apply_runtime_fixups()
        self._progress_callback = progress_callback
        self.records: List[AssetRecord] = []
        self.results: dict = {}
        self._results_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._failed_assets = 0
        self._selected_asset_relpaths: Optional[set[str]] = None
        self._selected_map_suffixes: Optional[set[str]] = None

        # Systems
        self.checkpoint = CheckpointManager(config)
        self.gpu_monitor = GPUMonitor(config)

        # Phase processors (lazy init -- imported on first use)
        self._upscaler = None
        self._pbr_gen = None
        self._normal_gen = None
        self._pom_proc = None
        self._mipmap_gen = None
        self._orm_packer = None
        self._color_consistency = None
        self._color_grading = None
        self._seam_repair = None
        self._emissive_gen = None
        self._reflection_mask_gen = None
        self._specular_aa = None
        self._detail_overlay = None
        self._validator = None

        # Tracks phases skipped due to missing dependencies
        self._skipped_phases: List[str] = []
        # Tracks which phase names actually executed (for validation gating)
        self._executed_phases: set = set()

    # ------------------------------------------
    # Helpers
    # ------------------------------------------

    # Map import names to pip package names where they differ
    _PIP_NAMES = {"cv2": "opencv-python-headless"}
    _CACHED_PATH_EXTS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tga",
        ".tif",
        ".tiff",
        ".dds",
        ".ktx2",
        ".json",
        ".txt",
        ".glsl",
        ".hlsl",
        ".csv",
    }
    _CACHED_PATH_SKIP_KEYS = {"error", "errors", "reason", "fallback"}

    @classmethod
    def _is_probable_cached_path(cls, value: str) -> bool:
        """Return True when a cached string value appears to be a file path."""
        candidate = str(value or "").strip()
        if not candidate:
            return False
        if "://" in candidate or candidate.startswith("data:"):
            return False
        if "\n" in candidate or "\r" in candidate:
            return False

        suffix = Path(candidate).suffix.lower()
        if suffix not in cls._CACHED_PATH_EXTS:
            return False

        if os.path.isabs(candidate):
            return True
        if candidate.startswith("./") or candidate.startswith(".\\"):
            return True
        if "/" in candidate or "\\" in candidate:
            return True
        return False

    @classmethod
    def _find_missing_cached_paths(
        cls,
        payload,
        key_path: str = "",
    ) -> list[tuple[str, str]]:
        """Recursively collect missing file paths from cached phase payloads."""
        missing: list[tuple[str, str]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in cls._CACHED_PATH_SKIP_KEYS:
                    continue
                child_path = f"{key_path}.{key}" if key_path else key
                missing.extend(cls._find_missing_cached_paths(value, child_path))
            return missing
        if isinstance(payload, (list, tuple)):
            for idx, value in enumerate(payload):
                child_path = f"{key_path}[{idx}]"
                missing.extend(cls._find_missing_cached_paths(value, child_path))
            return missing
        if isinstance(payload, str) and cls._is_probable_cached_path(payload):
            if not os.path.exists(payload):
                missing.append((key_path or "<root>", payload))
        return missing

    @staticmethod
    def _is_failed_phase_result(phase_name: str, phase_result) -> bool:
        """Return True when a phase result should count as a failure."""
        if not isinstance(phase_result, dict):
            return False
        if phase_result.get("error"):
            return True
        if phase_name == "validate":
            if phase_result.get("passed") is False:
                return True
            errors = phase_result.get("errors")
            if isinstance(errors, list) and len(errors) > 0:
                return True
        if phase_name == "output_validation":
            issues = phase_result.get("issues")
            if isinstance(issues, list) and len(issues) > 0:
                return True
        return False

    @staticmethod
    def _normalize_relpath(rel_path: str) -> str:
        """Normalize a relative path for cross-platform matching."""
        return os.path.normpath(str(rel_path or ""))

    @classmethod
    def _normalized_relpath_set(cls, relpaths: Iterable[str]) -> set[str]:
        """Build a normalized set of non-empty relative paths."""
        normalized: set[str] = set()
        for rel_path in relpaths:
            norm = cls._normalize_relpath(rel_path)
            if norm:
                normalized.add(norm)
        return normalized

    @staticmethod
    def _normalize_map_suffix(suffix: str) -> str:
        """Normalize output map suffix identifiers (e.g. 'normal' -> '_normal')."""
        text = str(suffix or "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered in {"base", "_base"}:
            return ""
        if not text.startswith("_"):
            text = f"_{text}"
        return text.lower()

    @classmethod
    def _normalized_map_suffix_set(cls, suffixes: Iterable[str]) -> set[str]:
        """Build normalized suffix set for map filtering."""
        normalized: set[str] = set()
        for suffix in suffixes:
            normalized.add(cls._normalize_map_suffix(suffix))
        return normalized

    def _is_selected_map_suffix(self, suffix: str) -> bool:
        """Return whether a map suffix is enabled for this run."""
        if self._selected_map_suffixes is None:
            return True
        return self._normalize_map_suffix(suffix) in self._selected_map_suffixes

    def _report_progress(self, phase_name: str, done: int, total: int) -> None:
        """Publish per-phase progress to logs and optional callback."""
        done_i = int(done)
        total_i = max(int(total), 1)
        logger.info(
            "[progress] phase=%s done=%d total=%d",
            phase_name,
            done_i,
            total_i,
        )
        if self._progress_callback is not None:
            try:
                self._progress_callback(phase_name, done_i, total_i)
            except Exception:
                logger.debug("Progress callback failed.", exc_info=True)

    def _check_deps(self, phase_label: str, *module_names: str) -> list:
        """Pre-check that required modules are importable.

        Returns list of missing module names.  Logs a warning and returns
        the list so the caller can ``return`` early (skip the phase) when
        dependencies are absent.
        """
        import importlib.util as importlib_util

        missing = []
        for name in module_names:
            try:
                if importlib_util.find_spec(name) is None:
                    missing.append(name)
            except (ImportError, ModuleNotFoundError):
                missing.append(name)
            except Exception as exc:
                missing.append(name)
                logger.warning(
                    "Dependency probe failed for '%s': %s: %s",
                    name,
                    exc.__class__.__name__,
                    exc,
                )
        if missing:
            _actual_phases = {
                "upscale", "pbr", "normal", "pom",
                "mipmap", "postprocess", "validate",
            }
            if phase_label in _actual_phases:
                self._skipped_phases.append(phase_label)
            pip_pkgs = [self._PIP_NAMES.get(m, m) for m in missing]
            logger.warning(
                f"{phase_label} skipped: missing dependencies: "
                f"{', '.join(missing)}.  "
                f"Install with: pip install {' '.join(pip_pkgs)}"
            )
        return missing

    def _should_skip(self, record: AssetRecord, phase: str) -> bool:
        """Check if asset should be skipped (already completed, unchanged)."""
        return self.checkpoint.is_completed(record.filepath, record.file_hash, [phase])

    def _get_cached_phase_result(self, record: AssetRecord, phase_name: str):
        """Return cached phase result or ``None`` when cache should be invalidated."""
        cached = self.checkpoint.get_result(record.filepath, phase_name)
        if cached is None or not isinstance(cached, dict):
            return cached

        stale_paths = self._find_missing_cached_paths(cached)
        if stale_paths:
            preview = ", ".join(path for _, path in stale_paths[:3])
            if len(stale_paths) > 3:
                preview = f"{preview}, ..."
            logger.info(
                "Stale cached paths for %s/%s (%d missing): %s -- will reprocess",
                record.filename,
                phase_name,
                len(stale_paths),
                preview,
            )
            return None
        return cached

    @staticmethod
    def _run_record_with_timeout(
        phase_name: str,
        record: AssetRecord,
        process_fn,
        timeout_seconds: int,
    ):
        """Run one record in a daemon worker thread with a hard timeout."""
        result_queue: Queue = Queue(maxsize=1)

        def _worker():
            try:
                result_queue.put(("ok", process_fn(record)))
            except BaseException as exc:  # pragma: no cover - exercised via caller path
                result_queue.put(("err", exc))

        worker = threading.Thread(
            target=_worker,
            name=f"{phase_name}:{record.filename}",
            daemon=True,
        )
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            raise PhaseTimeoutError(
                f"Timed out after {timeout_seconds}s on {record.filename}"
            )

        try:
            status, payload = result_queue.get_nowait()
        except Empty as exc:
            raise RuntimeError(
                f"{phase_name} worker finished without returning a result for {record.filename}"
            ) from exc

        if status == "err":
            raise payload
        return payload

    def request_cancel(self):
        """Request cooperative cancellation for the active pipeline run."""
        self._cancel_event.set()

    def _run_parallel(self, phase_name: str, records: List[AssetRecord],
                      process_fn, desc: str, max_workers: int = None):
        """Run a phase with progress tracking, error handling, and checkpoint saves.

        process_fn(record) -> dict with result

        NOTE: process_fn closures read self.results from *previous* phases
        without holding _results_lock. This is safe only because phases run
        sequentially (previous-phase writes are fully complete before the
        next phase starts) and CPython's GIL protects dict reads.  Do NOT
        run multiple phases concurrently without adding locking to the
        cross-phase result reads inside process_fn closures.
        """
        workers = max_workers or self.config.max_workers
        phase_timeout = max(int(self.config.phase_timeout_seconds), 1)
        failure_abort_ratio = float(self.config.phase_failure_abort_ratio)
        failure_abort_min = int(self.config.phase_failure_abort_min_processed)

        def _failure_ratio_exceeded(failed: int, processed: int) -> bool:
            if processed < failure_abort_min:
                return False
            return (failed / max(processed, 1)) >= failure_abort_ratio

        # GPU-bound phases must be sequential (model not thread-safe)
        gpu_phases = {"upscale"}
        if phase_name in gpu_phases:
            workers = 1

        if self.config.dry_run:
            skipped = 0
            would_process = 0
            for r in records:
                if self._should_skip(r, phase_name):
                    skipped += 1
                else:
                    would_process += 1
            logger.info(
                f"[DRY RUN] {phase_name}: would process {would_process}, "
                f"skip {skipped} (cached)"
            )
            return

        if workers <= 1:
            # Sequential -- lock not needed but consistent API
            total_items = len(records)
            done_items = 0
            processed_items = 0
            failed_items = 0
            for record in tqdm(records, desc=desc):
                if self._cancel_event.is_set():
                    raise PipelineCancelledError("Pipeline cancelled by user request")
                if self._should_skip(record, phase_name):
                    cached = self._get_cached_phase_result(record, phase_name)
                    if cached is not None:
                        with self._results_lock:
                            self.results[record.filepath][phase_name] = cached
                        logger.debug(f"Skipping (cached): {record.filename}")
                        done_items += 1
                        self._report_progress(phase_name, done_items, total_items)
                        continue

                processed_items += 1
                record_failed = False
                try:
                    result = self._run_record_with_timeout(
                        phase_name,
                        record,
                        process_fn,
                        phase_timeout,
                    )
                    with self._results_lock:
                        self.results[record.filepath][phase_name] = result
                    if not (isinstance(result, dict) and result.get("error")):
                        self.checkpoint.mark_completed(
                            record.filepath, record.file_hash, phase_name, result
                        )
                    else:
                        record_failed = True
                        logger.warning(
                            f"[{phase_name}] {record.filename} returned error, "
                            f"not caching: {result['error']}"
                        )
                except TimeoutError as e:
                    record_failed = True
                    msg = str(e)
                    logger.error(f"[{phase_name}] {record.filename}: {msg}")
                    with self._results_lock:
                        self.results[record.filepath][phase_name] = {"error": msg}
                    # Abort the phase after timeout to avoid continuing while
                    # a potentially hung worker thread is still running.
                    raise PhaseTimeoutError(
                        f"{phase_name} timed out on {record.filename}: {msg}"
                    ) from e
                except PhaseTimeoutError as e:
                    record_failed = True
                    self._cancel_event.set()
                    msg = str(e)
                    logger.error(f"[{phase_name}] {record.filename}: {msg}")
                    with self._results_lock:
                        self.results[record.filepath][phase_name] = {"error": msg}
                    raise
                except PipelineCancelledError as e:
                    record_failed = True
                    msg = str(e)
                    logger.warning(
                        f"[{phase_name}] Cancellation requested during "
                        f"{record.filename}"
                    )
                    with self._results_lock:
                        self.results[record.filepath][phase_name] = {"error": msg}
                    raise
                except Exception as e:
                    record_failed = True
                    logger.error(
                        "[%s] Failed %s: %s",
                        phase_name, record.filename, e, exc_info=True,
                    )
                    with self._results_lock:
                        self.results[record.filepath][phase_name] = {"error": str(e)}
                finally:
                    if record_failed:
                        failed_items += 1
                    done_items += 1
                    self._report_progress(phase_name, done_items, total_items)
                if _failure_ratio_exceeded(failed_items, processed_items):
                    ratio = failed_items / max(processed_items, 1)
                    raise PhaseFailureThresholdError(
                        f"{phase_name} aborted: failure ratio {ratio:.1%} "
                        f"exceeded configured threshold "
                        f"{failure_abort_ratio:.1%} "
                        f"({failed_items}/{processed_items})"
                    )
        else:
            # Parallel
            to_process = []
            for record in records:
                if self._should_skip(record, phase_name):
                    cached = self._get_cached_phase_result(record, phase_name)
                    if cached is not None:
                        with self._results_lock:
                            self.results[record.filepath][phase_name] = cached
                        continue
                    # else: stale cache — fall through to reprocess
                to_process.append(record)

            if not to_process:
                logger.info(f"All assets already processed for {phase_name}")
                return

            # Parallel execution without nested daemon timeout wrappers.
            # We enforce timeout by watching task runtime and aborting the phase
            # when a task exceeds the limit.
            executor = ThreadPoolExecutor(max_workers=workers)
            start_lock = threading.Lock()
            start_times: dict[str, float] = {}

            def _wrapped_process(rec):
                with start_lock:
                    start_times[rec.filepath] = time.monotonic()
                return process_fn(rec)
            phase_abort_error: Exception | None = None
            futures = {}
            try:
                futures = {
                    executor.submit(
                        _wrapped_process,
                        rec,
                    ): rec
                    for rec in to_process
                }
                with tqdm(total=len(futures), desc=desc) as pbar:
                    pending = set(futures)
                    total_items = len(futures)
                    done_items = 0
                    processed_items = 0
                    failed_items = 0

                    while pending:
                        if self._cancel_event.is_set():
                            cancel_msg = f"Cancelled by user in {phase_name}"
                            cancelled_count = 0
                            for future in list(pending):
                                pending_record = futures[future]
                                future.cancel()
                                with start_lock:
                                    start_times.pop(pending_record.filepath, None)
                                with self._results_lock:
                                    self.results[pending_record.filepath][phase_name] = {
                                        "error": cancel_msg
                                    }
                                cancelled_count += 1
                            pbar.update(cancelled_count)
                            done_items += cancelled_count
                            self._report_progress(phase_name, done_items, total_items)
                            phase_abort_error = PipelineCancelledError(
                                "Pipeline cancelled by user request"
                            )
                            break

                        done, pending = wait(
                            pending,
                            timeout=0.2,
                            return_when=FIRST_COMPLETED,
                        )

                        now = time.monotonic()
                        timed_out_future = None
                        timed_out_elapsed = 0.0
                        for future in pending:
                            record = futures[future]
                            with start_lock:
                                start_time = start_times.get(record.filepath)
                            if start_time is None:
                                continue
                            elapsed = now - start_time
                            if elapsed > phase_timeout and elapsed >= timed_out_elapsed:
                                timed_out_future = future
                                timed_out_elapsed = elapsed

                        if timed_out_future is not None:
                            # Signal cooperative cancellation so that
                            # still-running worker threads stop writing
                            # outputs as soon as they check the flag.
                            self._cancel_event.set()
                            record = futures[timed_out_future]
                            with start_lock:
                                start_times.pop(record.filepath, None)
                            # Remove the timed-out future from pending so it
                            # is not iterated over after the break.
                            pending.discard(timed_out_future)
                            msg = f"Timed out after {phase_timeout}s"
                            logger.error(f"[{phase_name}] {record.filename}: {msg}")
                            with self._results_lock:
                                self.results[record.filepath][phase_name] = {"error": msg}
                            processed_items += 1
                            failed_items += 1

                            cancel_msg = (
                                f"Cancelled after timeout in {phase_name} "
                                f"({record.filename})"
                            )
                            cancelled_count = 0
                            for future in list(pending):
                                pending_record = futures[future]
                                future.cancel()
                                with start_lock:
                                    start_times.pop(pending_record.filepath, None)
                                with self._results_lock:
                                    self.results[pending_record.filepath][phase_name] = {
                                        "error": cancel_msg
                                    }
                                cancelled_count += 1

                            pbar.update(1 + cancelled_count)
                            done_items += 1 + cancelled_count
                            self._report_progress(phase_name, done_items, total_items)
                            phase_abort_error = PhaseTimeoutError(
                                f"{phase_name} timed out on {record.filename}: {msg}"
                            )
                            break

                        for future in done:
                            record = futures.pop(future)
                            with start_lock:
                                start_times.pop(record.filepath, None)
                            processed_items += 1
                            record_failed = False
                            try:
                                result = future.result()
                                with self._results_lock:
                                    self.results[record.filepath][phase_name] = result
                                if not (isinstance(result, dict) and result.get("error")):
                                    self.checkpoint.mark_completed(
                                        record.filepath, record.file_hash,
                                        phase_name, result
                                    )
                                else:
                                    record_failed = True
                                    logger.warning(
                                        f"[{phase_name}] {record.filename} returned "
                                        f"error, not caching: {result['error']}"
                                    )
                            except Exception as e:
                                record_failed = True
                                logger.error(f"[{phase_name}] Failed {record.filename}: {e}")
                                with self._results_lock:
                                    self.results[record.filepath][phase_name] = {"error": str(e)}
                            if record_failed:
                                failed_items += 1
                            pbar.update(1)
                            done_items += 1
                            self._report_progress(phase_name, done_items, total_items)
                            if _failure_ratio_exceeded(failed_items, processed_items):
                                ratio = failed_items / max(processed_items, 1)
                                msg = (
                                    f"{phase_name} aborted: failure ratio "
                                    f"{ratio:.1%} exceeded configured threshold "
                                    f"{failure_abort_ratio:.1%} "
                                    f"({failed_items}/{processed_items})"
                                )
                                logger.error(msg)
                                cancel_msg = (
                                    f"Cancelled after failure threshold in "
                                    f"{phase_name}"
                                )
                                cancelled_count = 0
                                for pending_future in list(pending):
                                    pending_record = futures[pending_future]
                                    pending_future.cancel()
                                    with start_lock:
                                        start_times.pop(pending_record.filepath, None)
                                    with self._results_lock:
                                        self.results[pending_record.filepath][phase_name] = {
                                            "error": cancel_msg
                                        }
                                    cancelled_count += 1
                                if cancelled_count:
                                    pbar.update(cancelled_count)
                                    done_items += cancelled_count
                                    self._report_progress(
                                        phase_name, done_items, total_items
                                    )
                                pending.clear()
                                phase_abort_error = PhaseFailureThresholdError(msg)
                                break
                        if phase_abort_error is not None:
                            break
                if phase_abort_error is not None:
                    raise phase_abort_error
            finally:
                if phase_abort_error is not None:
                    # Abort quickly after timeout. Pending futures were cancelled.
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
                # Release GPU cache to prevent orphaned workers holding VRAM.
                # NOTE: upscaler cleanup is handled by phase1_upscale's own
                # finally block — doing it here would cause double-cleanup and
                # cross-phase teardown when an unrelated phase times out.
                if phase_abort_error is not None:
                    try:
                        if self.gpu_monitor is not None:
                            self.gpu_monitor.clear_cache()
                    except Exception:
                        pass

    def _log_phase_summary(self, phase_name: str):
        processed = skipped = errors = 0
        with self._results_lock:
            for filepath, res in self.results.items():
                if phase_name not in res:
                    continue
                d = res[phase_name]
                if isinstance(d, dict):
                    if d.get("skipped"):
                        skipped += 1
                    elif self._is_failed_phase_result(phase_name, d):
                        errors += 1
                    else:
                        processed += 1
        logger.info(f"{phase_name}: processed={processed}, skipped={skipped}, errors={errors}")

    @staticmethod
    def _map_name_from_suffix(suffix: str) -> str:
        return "base" if suffix == "" else suffix.lstrip("_")

    @staticmethod
    def _safe_getsize(path: str) -> int:
        if not path:
            logger.debug("Skipping size lookup for empty path.")
            return -1
        try:
            return os.path.getsize(path)
        except OSError:
            logger.debug("Size lookup failed (path may be missing): %s", path)
            return -1

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes < 0:
            return "n/a"
        size = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size < 1024.0 or unit == "TiB":
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{int(num_bytes)} B"

    @classmethod
    def _format_size_change(cls, before_bytes: int, after_bytes: int) -> str:
        if before_bytes < 0 or after_bytes < 0:
            return "size=n/a"
        delta = after_bytes - before_bytes
        delta_sign = "+" if delta >= 0 else "-"
        delta_text = f"{delta_sign}{cls._format_bytes(abs(delta))}"
        before_text = cls._format_bytes(before_bytes)
        after_text = cls._format_bytes(after_bytes)
        if before_bytes > 0:
            pct = (delta / before_bytes) * 100.0
            return (
                f"size={before_text}->{after_text} "
                f"({delta_text}, {pct:+.1f}%)"
            )
        return f"size={before_text}->{after_text} ({delta_text})"

    def _compression_source_size(
        self, png_path: str, mip_paths: list, use_mipchain: bool
    ) -> int:
        if not use_mipchain:
            return self._safe_getsize(png_path)
        total = 0
        have_any = False
        for mip_info in mip_paths:
            mip_path = ""
            if isinstance(mip_info, dict):
                mip_path = str(mip_info.get("path", ""))
            size = self._safe_getsize(mip_path)
            if size >= 0:
                total += size
                have_any = True
        return total if have_any else -1

    def _log_compression_start(
        self,
        fmt: str,
        scope: str,
        record: AssetRecord,
        suffix: str,
        mode: str,
        src_path: str,
        dst_path: str,
        codec: str = "",
    ) -> None:
        map_name = self._map_name_from_suffix(suffix)
        codec_part = f", codec={codec}" if codec else ""
        logger.info(
            "[compress][%s][%s] start asset=%s, map=%s, mode=%s%s, src=%s, dst=%s",
            fmt,
            scope,
            record.filepath,
            map_name,
            mode,
            codec_part,
            src_path,
            dst_path,
        )

    def _log_compression_result(
        self,
        fmt: str,
        scope: str,
        record: AssetRecord,
        suffix: str,
        status: str,
        src_size: int,
        dst_path: str,
        reason: str = "",
    ) -> int:
        map_name = self._map_name_from_suffix(suffix)
        dst_size = self._safe_getsize(dst_path)
        size_text = self._format_size_change(src_size, dst_size)
        reason_text = f", reason={reason}" if reason else ""
        line = (
            f"[compress][{fmt}][{scope}] {status.upper()} asset={record.filepath}, "
            f"map={map_name}, dst={dst_path}, {size_text}{reason_text}"
        )
        if status.lower() in {"failed", "error", "skipped"}:
            logger.warning(line)
        else:
            logger.info(line)
        return dst_size

    def _record_copy_failure(self, record: AssetRecord, message: str) -> None:
        """Record copy/compression failure in results for failed-asset accounting."""
        with self._results_lock:
            record_results = self.results.setdefault(record.filepath, {})
            copy_entry = record_results.setdefault("copy", {})
            prev = copy_entry.get("error")
            if prev:
                if message not in prev:
                    copy_entry["error"] = f"{prev}; {message}"
            else:
                copy_entry["error"] = message
            errors = copy_entry.setdefault("errors", [])
            if message not in errors:
                errors.append(message)

    # ------------------------------------------
    # Phase 0: Scan
    # ------------------------------------------

    def phase0_scan(self):
        """Scan input assets, classify them, and initialize result entries."""
        logger.info("=" * 60)
        logger.info("PHASE 0: Scanning & Classifying Assets")
        logger.info("=" * 60)

        scanned_records = scan_assets(self.config.input_dir, self.config)

        if self._selected_asset_relpaths is not None:
            selected = self._selected_asset_relpaths
            unmatched = set(selected)
            filtered_records = []
            for rec in scanned_records:
                rec_norm = self._normalize_relpath(rec.filepath)
                if rec_norm in selected:
                    filtered_records.append(rec)
                    unmatched.discard(rec_norm)
            self.records = filtered_records
            logger.info(
                "Asset selection active: processing %d/%d scanned assets.",
                len(self.records),
                len(scanned_records),
            )
            if unmatched:
                logger.warning(
                    "Selected assets not found in scan (%d): %s",
                    len(unmatched),
                    ", ".join(sorted(unmatched)[:10])
                    + (" ..." if len(unmatched) > 10 else ""),
                )
        else:
            self.records = scanned_records

        type_counts = {}
        for r in self.records:
            type_counts[r.texture_type] = type_counts.get(r.texture_type, 0) + 1

        logger.info(f"Found {len(self.records)} assets:")
        for t, c in sorted(type_counts.items()):
            logger.info(f"  {t}: {c}")

        tileable = sum(1 for r in self.records if r.is_tileable)
        hero = sum(1 for r in self.records if r.is_hero)
        logger.info(f"  Tileable: {tileable}, Hero: {hero}")

        if self.config.dry_run:
            logger.info("[DRY RUN] Manifest write skipped")
        else:
            save_manifest(self.records, self.config.manifest_path)

        for r in self.records:
            self.results[r.filepath] = {}

        # Prune stale checkpoint entries for files no longer present
        active = {r.filepath for r in scanned_records}
        self.checkpoint.prune(active)

    # ------------------------------------------
    # Phase 1: Upscale
    # ------------------------------------------

    def phase1_upscale(self):
        """Execute phase 1 upscaling for all scanned assets."""
        if not self.config.upscale.enabled:
            logger.info("Phase 1 (Upscale) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 1: AI Upscaling")
        logger.info("=" * 60)

        if self.config.dry_run:
            self._run_parallel(
                "upscale", self.records, None,
                desc="Upscaling", max_workers=1
            )
            return

        missing_upscale_deps = self._check_deps(
            "Phase 1 (Upscale)", "cv2", "torch", "realesrgan", "basicsr"
        )
        if missing_upscale_deps:
            reason = (
                "missing AI dependencies: "
                + ", ".join(missing_upscale_deps)
            )
            if self.config.upscale.require_ai:
                raise RuntimeError(
                    f"AI upscale is required (upscale.require_ai=true) "
                    f"but dependencies are missing: "
                    f"{', '.join(missing_upscale_deps)}. "
                    f"Install with: pip install realesrgan basicsr"
                )
            logger.warning(
                "Phase 1 fallback active: AI upscale unavailable (%s). "
                "Base outputs will reuse original input textures.",
                reason,
            )
            with self._results_lock:
                for rec in self.records:
                    rec_results = self.results.setdefault(rec.filepath, {})
                    if "upscale" not in rec_results:
                        rec_results["upscale"] = {
                            "upscaled": None,
                            "skipped": True,
                            "fallback": "original_input",
                            "reason": reason,
                        }
            return

        from .phases.upscale import TextureUpscaler
        self._upscaler = TextureUpscaler(self.config)
        self.gpu_monitor.log_usage("before_upscale")

        try:
            self._run_parallel(
                "upscale", self.records,
                lambda rec: self._upscaler.process(rec),
                desc="Upscaling",
                max_workers=1  # GPU-bound, always sequential
            )
        finally:
            if self._upscaler is not None:
                try:
                    self._upscaler.cleanup()
                except Exception:
                    logger.debug("Upscaler cleanup error (ignored).", exc_info=True)
                self._upscaler = None
        self._log_phase_summary("upscale")

    # ------------------------------------------
    # Phase 2: PBR
    # ------------------------------------------

    def phase2_pbr(self):
        """Execute phase 2 PBR map generation for all scanned assets."""
        if not self.config.pbr.enabled:
            logger.info("Phase 2 (PBR) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 2: PBR Material Conversion")
        logger.info("=" * 60)

        if self.config.dry_run:
            self._run_parallel("pbr", self.records, None, desc="PBR Generation")
            return

        if self._check_deps("Phase 2 (PBR)", "cv2", "scipy"):
            logger.warning("Phase 2 (PBR) skipped: missing dependencies.")
            return

        from .phases.pbr import PBRGenerator
        self._pbr_gen = PBRGenerator(self.config)

        def process(rec):
            upscale_res = self.results.get(rec.filepath, {}).get("upscale", {})
            if not isinstance(upscale_res, dict):
                upscale_res = {}
            upscaled = upscale_res.get("upscaled")
            if upscaled is None and self.config.upscale.enabled:
                logger.debug(
                    "PBR: no upscaled path for %s; using original input.",
                    rec.filename,
                )
            return self._pbr_gen.process(rec, upscaled)

        self._run_parallel("pbr", self.records, process, desc="PBR Generation")
        self._log_phase_summary("pbr")

    # ------------------------------------------
    # Phase 3: Normal Maps
    # ------------------------------------------

    def phase3_normals(self):
        """Execute phase 3 normal/height generation for all scanned assets."""
        if not self.config.normal.enabled:
            logger.info("Phase 3 (Normals) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 3: Normal & Height Map Generation")
        logger.info("=" * 60)

        if self.config.dry_run:
            self._run_parallel("normal", self.records, None, desc="Normal Generation")
            return

        if self._check_deps("Phase 3 (Normal)", "cv2", "scipy"):
            logger.warning("Phase 3 (Normals) skipped: missing dependencies.")
            return

        from .phases.normal import NormalMapGenerator
        self._normal_gen = NormalMapGenerator(self.config)

        def process(rec):
            res = self.results.get(rec.filepath, {})
            upscale_res = res.get("upscale", {})
            if not isinstance(upscale_res, dict):
                upscale_res = {}
            upscaled = upscale_res.get("upscaled")
            albedo = res.get("pbr", {}).get("albedo")
            if upscaled is None and self.config.upscale.enabled:
                logger.debug(
                    "Normal: no upscaled path for %s; using original input.",
                    rec.filename,
                )
            return self._normal_gen.process(rec, upscaled, albedo)

        try:
            self._run_parallel("normal", self.records, process, desc="Normal Generation")
        finally:
            if self._normal_gen is not None:
                try:
                    self._normal_gen.cleanup()
                except Exception:
                    logger.debug("Normal generator cleanup error (ignored).", exc_info=True)
                self._normal_gen = None
        self._log_phase_summary("normal")

    # ------------------------------------------
    # Phase 4: POM
    # ------------------------------------------

    def phase4_pom(self):
        """Execute phase 4 height-map refinement for parallax mapping."""
        if not self.config.pom.enabled:
            logger.info("Phase 4 (POM) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 4: POM Height Map Refinement")
        logger.info("=" * 60)

        if self.config.dry_run:
            self._run_parallel("pom", self.records, None, desc="POM Processing")
            return

        if self._check_deps("Phase 4 (POM)", "cv2", "scipy"):
            logger.warning("Phase 4 (POM) skipped: missing dependencies.")
            return

        from .phases.pom import POMProcessor
        self._pom_proc = POMProcessor(self.config)

        def process(rec):
            height = self.results.get(rec.filepath, {}).get("normal", {}).get("height")
            if not height:
                return {"skipped": True}
            return self._pom_proc.process(rec, height)

        self._run_parallel("pom", self.records, process, desc="POM Processing")

        # Only export reference shaders if at least one asset was processed
        with self._results_lock:
            any_processed = any(
                isinstance(self.results.get(r.filepath, {}).get("pom"), dict)
                and not self.results[r.filepath]["pom"].get("skipped")
                and not self.results[r.filepath]["pom"].get("error")
                for r in self.records
            )
        if any_processed:
            shader_dir = os.path.join(self.config.output_dir, "_shaders")
            self._pom_proc.export_shaders(shader_dir)
        self._log_phase_summary("pom")

    # ------------------------------------------
    # Phase 6: Mipmaps
    # ------------------------------------------

    def phase6_mipmaps(self):
        """Execute phase 6 mipmap generation for eligible outputs."""
        if not self.config.mipmap.enabled:
            logger.info("Phase 6 (Mipmaps) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 6: Mipmap Generation")
        logger.info("=" * 60)

        if self.config.dry_run:
            self._run_parallel("mipmap", self.records, None, desc="Mipmap Generation")
            return

        if self._check_deps("Phase 6 (Mipmap)", "cv2", "scipy"):
            logger.warning("Phase 6 (Mipmaps) skipped: missing dependencies.")
            return

        from .phases.mipmap import MipmapGenerator
        self._mipmap_gen = MipmapGenerator(self.config)

        def process(rec):
            res = self.results.get(rec.filepath, {})
            mip_results = {}

            # Main texture (prefer post-processed packed/repaired base)
            main_path = (
                res.get("orm", {}).get("diffuse_alpha_packed")
                or res.get("seam_repair", {}).get("upscaled_repaired")
                or res.get("upscale", {}).get("upscaled")
                or os.path.join(self.config.input_dir, rec.filepath)
            )
            if self._is_selected_map_suffix("") and os.path.exists(main_path):
                mip_results["main"] = self._mipmap_gen.process(
                    rec, main_path, TextureType(rec.texture_type), name_tag="main"
                )

            # Normal map (prefer post-processed detail overlay)
            normal_path = (
                res.get("detail_map", {}).get("normal_detailed") or
                res.get("normal", {}).get("normal")
            )
            if (
                self._is_selected_map_suffix("_normal")
                and normal_path
                and os.path.exists(normal_path)
            ):
                mip_results["normal"] = self._mipmap_gen.process(
                    rec, normal_path, TextureType.NORMAL, name_tag="normal"
                )

            # Roughness (prefer post-processed specular AA)
            rough_path = (
                res.get("specular_aa", {}).get("roughness_aa") or
                res.get("pbr", {}).get("roughness")
            )
            if (
                self._is_selected_map_suffix("_roughness")
                and rough_path
                and os.path.exists(rough_path)
            ):
                mip_results["roughness"] = self._mipmap_gen.process(
                    rec, rough_path, TextureType.ROUGHNESS, name_tag="roughness"
                )

            # Albedo (prefer seam-repaired + color-corrected + graded output)
            albedo_path = (
                res.get("color_grading", {}).get("graded")
                or res.get("color_consistency", {}).get("corrected")
                or res.get("seam_repair", {}).get("albedo_repaired")
                or
                res.get("pbr", {}).get("albedo")
            )
            if (
                self._is_selected_map_suffix("_albedo")
                and albedo_path
                and os.path.exists(albedo_path)
            ):
                mip_results["albedo"] = self._mipmap_gen.process(
                    rec, albedo_path, TextureType.ALBEDO, name_tag="albedo"
                )

            # Metalness
            metalness_path = res.get("pbr", {}).get("metalness")
            if (
                self._is_selected_map_suffix("_metalness")
                and metalness_path
                and os.path.exists(metalness_path)
            ):
                mip_results["metalness"] = self._mipmap_gen.process(
                    rec, metalness_path, TextureType.METALNESS, name_tag="metalness"
                )

            # Ambient Occlusion
            ao_path = res.get("pbr", {}).get("ao")
            if self._is_selected_map_suffix("_ao") and ao_path and os.path.exists(ao_path):
                mip_results["ao"] = self._mipmap_gen.process(
                    rec, ao_path, TextureType.AO, name_tag="ao"
                )

            # Gloss/smoothness (optional roughness inverse output)
            gloss_path = res.get("pbr", {}).get("gloss")
            if self._is_selected_map_suffix("_gloss") and gloss_path and os.path.exists(gloss_path):
                mip_results["gloss"] = self._mipmap_gen.process(
                    rec, gloss_path, TextureType.ROUGHNESS, name_tag="gloss"
                )

            # Height (prefer POM-refined, fall back to raw)
            height_path = (
                res.get("pom", {}).get("height_refined") or
                res.get("normal", {}).get("height")
            )
            if (
                self._is_selected_map_suffix("_height")
                and height_path
                and os.path.exists(height_path)
            ):
                mip_results["height"] = self._mipmap_gen.process(
                    rec, height_path, TextureType.HEIGHT, name_tag="height"
                )

            # ORM packed map
            orm_path = res.get("orm", {}).get("orm")
            if self._is_selected_map_suffix("_orm") and orm_path and os.path.exists(orm_path):
                mip_results["orm"] = self._mipmap_gen.process(
                    rec, orm_path, TextureType.ORM, name_tag="orm"
                )

            # Emissive
            emissive_path = res.get("emissive", {}).get("emissive")
            if (
                self._is_selected_map_suffix("_emissive")
                and emissive_path
                and os.path.exists(emissive_path)
            ):
                mip_results["emissive"] = self._mipmap_gen.process(
                    rec, emissive_path, TextureType.EMISSIVE, name_tag="emissive"
                )

            # Emissive mask
            emissive_mask_path = res.get("emissive", {}).get("emissive_mask")
            if (
                self._is_selected_map_suffix("_emissive_mask")
                and emissive_mask_path
                and os.path.exists(emissive_mask_path)
            ):
                mip_results["emissive_mask"] = self._mipmap_gen.process(
                    rec, emissive_mask_path, TextureType.MASK,
                    name_tag="emissive_mask",
                )

            # Reflection/environment mask
            envmask_path = res.get("reflection_mask", {}).get("env_mask")
            if (
                self._is_selected_map_suffix("_envmask")
                and envmask_path
                and os.path.exists(envmask_path)
            ):
                mip_results["envmask"] = self._mipmap_gen.process(
                    rec, envmask_path, TextureType.MASK, name_tag="envmask"
                )

            # Material zone mask (RGBA channels: metal/cloth/leather/skin)
            zone_path = res.get("pbr", {}).get("zone_mask")
            if self._is_selected_map_suffix("_zones") and zone_path and os.path.exists(zone_path):
                mip_results["zone_mask"] = self._mipmap_gen.process(
                    rec, zone_path, TextureType.MASK, name_tag="zones"
                )

            return mip_results

        self._run_parallel("mipmap", self.records, process, desc="Mipmap Generation")
        self._log_phase_summary("mipmap")

    # ------------------------------------------
    # Phase 5: Post-Processing
    # ------------------------------------------

    def phase5_postprocess(self):
        """Execute phase 5 post-processing passes."""
        enabled_subpasses = []
        if self.config.seam_repair.enabled:
            enabled_subpasses.append("seam_repair")
        if self.config.color_consistency.enabled:
            enabled_subpasses.append("color_consistency")
        if self.config.color_grading.enabled:
            enabled_subpasses.append("color_grading")
        if self.config.specular_aa.enabled:
            enabled_subpasses.append("specular_aa")
        if self.config.detail_map.enabled:
            enabled_subpasses.append("detail_map")
        if self.config.emissive.enabled:
            enabled_subpasses.append("emissive")
        if self.config.reflection_mask.enabled:
            enabled_subpasses.append("reflection_mask")
        if self.config.orm_packing.enabled:
            enabled_subpasses.append("orm")
        if not enabled_subpasses:
            logger.info("Phase 5 (Post-Processing) disabled (all sub-passes are disabled).")
            return

        logger.info("=" * 60)
        logger.info("PHASE 5: Post-Processing")
        logger.info("=" * 60)

        if self.config.dry_run:
            for sub in enabled_subpasses:
                self._run_parallel(sub, self.records, None, desc="Post-Processing")
            return

        if self._check_deps("Phase 5 (Post-process)", "cv2", "scipy"):
            logger.warning("Phase 5 (Post-Processing) skipped: missing dependencies.")
            return

        from .phases.postprocess import (
            ColorConsistencyPass,
            ColorGradingPass,
            DetailMapOverlay,
            EmissiveMapGenerator,
            ORMPacker,
            ReflectionMaskGenerator,
            SeamRepairProcessor,
            SpecularAAProcessor,
        )

        def _get_albedo_source(rec):
            res = self.results.get(rec.filepath, {})
            return (
                res.get("seam_repair", {}).get("albedo_repaired")
                or res.get("pbr", {}).get("albedo")
                or res.get("seam_repair", {}).get("upscaled_repaired")
                or res.get("upscale", {}).get("upscaled")
            )

        # 5a. Seam repair
        if self.config.seam_repair.enabled:
            logger.info("--- Seam Repair ---")
            self._seam_repair = SeamRepairProcessor(self.config)

            def seam_process(rec):
                res = self.results.get(rec.filepath, {})
                upscaled = res.get("upscale", {}).get("upscaled")
                albedo = res.get("pbr", {}).get("albedo")
                return self._seam_repair.process(rec, upscaled, albedo)

            self._run_parallel("seam_repair", self.records, seam_process, desc="Seam Repair")

        # 5b. Color consistency
        if self.config.color_consistency.enabled:
            logger.info("--- Color Consistency Pass ---")
            self._color_consistency = ColorConsistencyPass(self.config)
            self._color_consistency.build_references(self.records, _get_albedo_source)

            def cc_process(rec):
                albedo = _get_albedo_source(rec)
                if not albedo:
                    return {"skipped": True}
                return self._color_consistency.process(rec, albedo)

            self._run_parallel(
                "color_consistency", self.records, cc_process,
                desc="Color Consistency",
            )

        # 5c. Color grading
        if self.config.color_grading.enabled:
            logger.info("--- Color Grading ---")
            self._color_grading = ColorGradingPass(self.config)

            def cg_process(rec):
                res = self.results.get(rec.filepath, {})
                albedo = (
                    res.get("color_consistency", {}).get("corrected")
                    or _get_albedo_source(rec)
                )
                if not albedo:
                    return {"skipped": True}
                return self._color_grading.process(rec, albedo)

            self._run_parallel("color_grading", self.records, cg_process, desc="Color Grading")

        # 5d. Specular AA
        if self.config.specular_aa.enabled:
            logger.info("--- Specular Anti-Aliasing ---")
            self._specular_aa = SpecularAAProcessor(self.config)

            def saa_process(rec):
                res = self.results.get(rec.filepath, {})
                normal = res.get("normal", {}).get("normal")
                roughness = res.get("pbr", {}).get("roughness")
                return self._specular_aa.process(rec, normal, roughness)

            self._run_parallel("specular_aa", self.records, saa_process, desc="Specular AA")

        # 5e. Detail normal overlay
        if self.config.detail_map.enabled:
            logger.info("--- Detail Map Overlay ---")
            self._detail_overlay = DetailMapOverlay(self.config)

            def detail_process(rec):
                res = self.results.get(rec.filepath, {})
                normal = res.get("normal", {}).get("normal")
                return self._detail_overlay.process(rec, normal)

            self._run_parallel("detail_map", self.records, detail_process, desc="Detail Maps")

        # 5f. Emissive detection
        if self.config.emissive.enabled:
            logger.info("--- Emissive Detection ---")
            self._emissive_gen = EmissiveMapGenerator(self.config)

            def emissive_process(rec):
                res = self.results.get(rec.filepath, {})
                albedo = (
                    res.get("color_grading", {}).get("graded")
                    or res.get("color_consistency", {}).get("corrected")
                    or _get_albedo_source(rec)
                )
                return self._emissive_gen.process(rec, albedo)

            self._run_parallel("emissive", self.records, emissive_process, desc="Emissive Maps")

        # 5g. Reflection mask generation
        if self.config.reflection_mask.enabled:
            logger.info("--- Reflection Mask Generation ---")
            self._reflection_mask_gen = ReflectionMaskGenerator(self.config)

            def reflection_process(rec):
                res = self.results.get(rec.filepath, {})
                roughness = (
                    res.get("specular_aa", {}).get("roughness_aa")
                    or res.get("pbr", {}).get("roughness")
                )
                metalness = res.get("pbr", {}).get("metalness")
                return self._reflection_mask_gen.process(rec, roughness, metalness)

            self._run_parallel(
                "reflection_mask", self.records,
                reflection_process, desc="Env Masks",
            )

        # 5h. Channel packing
        if self.config.orm_packing.enabled:
            logger.info("--- Channel Packing ---")
            self._orm_packer = ORMPacker(self.config)

            def orm_process(rec):
                res = self.results.get(rec.filepath, {})
                ao = res.get("pbr", {}).get("ao")
                roughness = (
                    res.get("specular_aa", {}).get("roughness_aa")
                    or res.get("pbr", {}).get("roughness")
                )
                metalness = res.get("pbr", {}).get("metalness")
                diffuse = (
                    res.get("seam_repair", {}).get("upscaled_repaired")
                    or res.get("upscale", {}).get("upscaled")
                )
                gloss = res.get("pbr", {}).get("gloss")
                return self._orm_packer.process(
                    rec,
                    ao,
                    roughness,
                    metalness,
                    diffuse_path=diffuse,
                    gloss_path=gloss,
                )

            self._run_parallel("orm", self.records, orm_process, desc="Channel Packing")

    # Backward-compatible wrappers (old phase-numbered method names).
    def phase5_mipmaps(self):
        """Backward-compatible alias for `phase6_mipmaps`."""
        return self.phase6_mipmaps()

    def phase6_postprocess(self):
        """Backward-compatible alias for `phase5_postprocess`."""
        return self.phase5_postprocess()

    # ------------------------------------------
    # Phase 7: Validate
    # ------------------------------------------

    def phase7_validate(self):
        """Execute phase 7 validation and write validation reports."""
        if not self.config.validation.enabled:
            logger.info("Phase 7 (Validation) disabled.")
            return

        logger.info("=" * 60)
        logger.info("PHASE 7: Validation & Output")
        logger.info("=" * 60)

        if self.config.dry_run:
            logger.info("[DRY RUN] validate: would validate %d assets", len(self.records))
            return

        if self._check_deps("Phase 7 (Validate)", "cv2"):
            logger.warning("Phase 7 (Validation) skipped: missing dependencies.")
            return

        from .phases.validate import Validator
        self._validator = Validator(self.config)

        def validate_process(rec):
            res = self.results.get(rec.filepath, {})
            processed = {
                "upscaled": (
                    res.get("seam_repair", {}).get("upscaled_repaired")
                    or res.get("upscale", {}).get("upscaled")
                ),
                "albedo": (
                    res.get("color_grading", {}).get("graded")
                    or res.get("color_consistency", {}).get("corrected")
                    or res.get("seam_repair", {}).get("albedo_repaired")
                    or res.get("pbr", {}).get("albedo")
                ),
                "roughness": (res.get("specular_aa", {}).get("roughness_aa") or
                              res.get("pbr", {}).get("roughness")),
                "metalness": res.get("pbr", {}).get("metalness"),
                "ao": res.get("pbr", {}).get("ao"),
                "gloss": res.get("pbr", {}).get("gloss"),
                "normal": (res.get("detail_map", {}).get("normal_detailed") or
                           res.get("normal", {}).get("normal")),
                "height": (res.get("pom", {}).get("height_refined") or
                          res.get("normal", {}).get("height")),
                "orm": res.get("orm", {}).get("orm"),
                "emissive": res.get("emissive", {}).get("emissive"),
                "env_mask": res.get("reflection_mask", {}).get("env_mask"),
                "zone_mask": res.get("pbr", {}).get("zone_mask"),
            }
            return self._validator.validate_asset(rec, processed)

        self._run_parallel("validate", self.records, validate_process,
                           desc="Validating")

        # Collect reports from results and write summary
        all_reports = []
        with self._results_lock:
            for rec in self.records:
                val_result = self.results.get(rec.filepath, {}).get("validate")
                if val_result and not isinstance(val_result, dict):
                    all_reports.append(val_result)
                elif (isinstance(val_result, dict)
                      and not val_result.get("error")
                      and not val_result.get("skipped")):
                    all_reports.append(val_result)

        if all_reports:
            report_path = os.path.join(self.config.output_dir, "validation_report.txt")
            self._validator.generate_report(all_reports, report_path)
            if getattr(self.config.tiling_quality, "enabled", False):
                tiling_path = os.path.join(self.config.output_dir, "tiling_quality_report.json")
                self._validator.generate_tiling_quality_report(all_reports, tiling_path)
        self._log_phase_summary("validate")

    # ------------------------------------------
    # Copy to Output
    # ------------------------------------------

    def copy_to_output(self):
        """Copy final artifacts to output, matching each input's original format.

        Primary output format is determined by the input file's extension:
        - .dds  -> DDS compressed via generate_dds/generate_dds_mipchain
        - .ktx2 -> KTX2 compressed via generate_ktx2
        - .tga  -> TGA via Pillow save_image
        - .png  -> PNG (current default behavior)
        - other -> saved via Pillow (jpg, bmp, tiff, etc.)

        Additional formats (generate_dds, generate_ktx2, generate_tga config
        flags) are still honored but skip when the primary already matches.
        """
        logger.info("=" * 60)
        logger.info("Copying final assets to output directory")
        logger.info("=" * 60)

        self._copy_load_image = load_image
        self._copy_save_image = save_image

        output_count = 0
        dds_count = 0
        dds_attempted = 0
        ktx2_count = 0
        ktx2_attempted = 0
        dds_degraded = 0
        tga_count = 0
        compression_fallbacks = 0
        dds_bytes_before = 0
        dds_bytes_after = 0
        ktx2_bytes_before = 0
        ktx2_bytes_after = 0
        generate_dds = (
            self.config.compression.enabled and
            self.config.compression.generate_dds
        )
        generate_ktx2 = (
            self.config.compression.enabled and
            self.config.compression.generate_ktx2
        )
        generate_tga = self.config.compression.generate_tga

        if not self.records:
            logger.warning("No records to copy")
            return

        # Pre-init MipmapGenerator once (needed for DDS/KTX2 compression).
        # Also required when input files are DDS/KTX2 so we can write the
        # primary output in the matching compressed format.
        has_compressed_input = any(
            Path(r.filepath).suffix.lower() in (".dds", ".ktx2")
            for r in self.records
        )
        need_mipmap_gen = generate_dds or generate_ktx2 or has_compressed_input

        if need_mipmap_gen and self._mipmap_gen is None:
            if self._check_deps("DDS/KTX2 generation", "cv2", "scipy"):
                logger.warning(
                    "DDS/KTX2 output generation disabled for this run: "
                    "missing compressor dependencies."
                )
                generate_dds = False
                generate_ktx2 = False
            else:
                from .phases.mipmap import MipmapGenerator
                self._mipmap_gen = MipmapGenerator(self.config)

        # Map suffixes to texture type keys for DDS compression format lookup
        suffix_to_type = {
            "": "unknown",  # Updated per-record in _copy_single_record
            "_albedo": "albedo",
            "_roughness": "roughness",
            "_metalness": "metalness",
            "_ao": "ao",
            "_gloss": "roughness",
            "_normal": "normal",
            "_height": "height",
            "_orm": "orm",
            "_emissive": "emissive",
            "_envmask": "mask",
            "_zones": "mask",
            "_emissive_mask": "mask",
        }

        # Map output suffix -> mipmap result key (Phase 6 data)
        suffix_to_mip_key = {
            "": "main",
            "_albedo": "albedo",
            "_normal": "normal",
            "_roughness": "roughness",
            "_metalness": "metalness",
            "_ao": "ao",
            "_gloss": "gloss",
            "_height": "height",
            "_orm": "orm",
            "_emissive": "emissive",
            "_envmask": "envmask",
            "_zones": "zone_mask",
            "_emissive_mask": "emissive_mask",
        }
        base_fallbacks = []
        stats_lock = threading.Lock()

        copy_workers = min(self.config.max_workers, 8)
        executor = ThreadPoolExecutor(max_workers=copy_workers)
        try:
            futures = {
                executor.submit(
                    self._copy_single_record,
                    record, generate_dds, generate_ktx2, generate_tga,
                    suffix_to_type, suffix_to_mip_key,
                ): record
                for record in self.records
            }
            for future in as_completed(futures):
                try:
                    local_stats = future.result()
                except Exception as e:
                    record = futures[future]
                    logger.error(
                        "Unexpected error copying record %s: %s",
                        record.filename, e,
                    )
                    continue
                with stats_lock:
                    output_count += local_stats["output_count"]
                    dds_count += local_stats["dds_count"]
                    dds_attempted += local_stats["dds_attempted"]
                    ktx2_count += local_stats["ktx2_count"]
                    ktx2_attempted += local_stats["ktx2_attempted"]
                    dds_degraded += local_stats["dds_degraded"]
                    tga_count += local_stats["tga_count"]
                    compression_fallbacks += local_stats["compression_fallbacks"]
                    dds_bytes_before += local_stats["dds_bytes_before"]
                    dds_bytes_after += local_stats["dds_bytes_after"]
                    ktx2_bytes_before += local_stats["ktx2_bytes_before"]
                    ktx2_bytes_after += local_stats["ktx2_bytes_after"]
                    base_fallbacks.extend(local_stats["base_fallbacks"])
        except BaseException:
            # Catch KeyboardInterrupt — cancel pending futures immediately
            # instead of blocking in shutdown(wait=True).
            for fut in futures:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        # Explicit cleanup after all records to reduce peak RSS for large batches
        import gc
        gc.collect()

        if base_fallbacks:
            examples = ", ".join(fp for fp, _ in base_fallbacks[:5])
            if len(base_fallbacks) > 5:
                examples = f"{examples}, ..."
            reasons = "; ".join(sorted({reason for _, reason in base_fallbacks}))
            logger.warning(
                "Base texture fallback used original inputs for %d asset(s). "
                "Examples: %s. Reason(s): %s",
                len(base_fallbacks), examples, reasons
            )

        logger.info(
            f"Copied {output_count} output files to {self.config.output_dir}"
        )
        if dds_attempted > 0:
            dds_failed = max(dds_attempted - dds_count - dds_degraded, 0)
            logger.info(f"DDS generation: {dds_count}/{dds_attempted} succeeded")
            if dds_count > 0:
                logger.info(
                    "DDS compression delta: %s",
                    self._format_size_change(dds_bytes_before, dds_bytes_after),
                )
            if dds_degraded > 0:
                logger.warning(
                    f"DDS quality degraded for {dds_degraded} file(s): "
                    f"mipchain assembly failed, fell back to base-only DDS "
                    f"(no mip levels)"
                )
            if dds_failed > 0:
                logger.warning(
                    f"DDS generation failed for {dds_failed}/{dds_attempted} "
                    f"textures. Check compression tool configuration."
                )
        if ktx2_attempted > 0:
            ktx2_failed = ktx2_attempted - ktx2_count
            logger.info(f"KTX2 generation: {ktx2_count}/{ktx2_attempted} succeeded")
            if ktx2_count > 0:
                logger.info(
                    "KTX2 compression delta: %s",
                    self._format_size_change(ktx2_bytes_before, ktx2_bytes_after),
                )
            if ktx2_failed > 0:
                logger.warning(
                    f"KTX2 generation failed for {ktx2_failed}/{ktx2_attempted} "
                    f"textures. Install toktx or basisu."
                )
        combined_before = dds_bytes_before + ktx2_bytes_before
        combined_after = dds_bytes_after + ktx2_bytes_after
        if (dds_count + ktx2_count) > 0:
            logger.info(
                "Compression total delta: %s",
                self._format_size_change(combined_before, combined_after),
            )
        if tga_count > 0:
            logger.info(f"TGA generation: {tga_count} files")
        if compression_fallbacks > 0:
            logger.warning(
                f"Compression fallback: {compression_fallbacks} file(s) "
                f"shipped as PNG because DDS/KTX2 compression failed. "
                f"These outputs are NOT in the expected compressed format."
            )

    def _copy_single_record(self, record, generate_dds, generate_ktx2,
                            generate_tga, suffix_to_type, suffix_to_mip_key):
        """Process a single record for output copying. Returns a stats dict.

        This is the per-record body extracted from copy_to_output for parallel
        execution via ThreadPoolExecutor.
        """
        stats = {
            "output_count": 0,
            "dds_count": 0,
            "dds_attempted": 0,
            "ktx2_count": 0,
            "ktx2_attempted": 0,
            "dds_degraded": 0,
            "tga_count": 0,
            "compression_fallbacks": 0,
            "dds_bytes_before": 0,
            "dds_bytes_after": 0,
            "ktx2_bytes_before": 0,
            "ktx2_bytes_after": 0,
            "base_fallbacks": [],
        }

        def _accumulate(fmt, src_size, dst_size):
            if src_size < 0 or dst_size < 0:
                return
            if fmt == "DDS":
                stats["dds_bytes_before"] += src_size
                stats["dds_bytes_after"] += dst_size
            elif fmt == "KTX2":
                stats["ktx2_bytes_before"] += src_size
                stats["ktx2_bytes_after"] += dst_size

        res = self.results.get(record.filepath, {})
        output_ext = _get_output_ext(record)

        # Base texture: prefer upscaled, fall back to original input
        upscale_result = res.get("upscale", {})
        if not isinstance(upscale_result, dict):
            upscale_result = {}
        packed_base = res.get("orm", {}).get("diffuse_alpha_packed")
        seam_base = res.get("seam_repair", {}).get("upscaled_repaired")
        upscaled_path = upscale_result.get("upscaled")
        base_texture = packed_base or seam_base or upscaled_path
        if not base_texture or not os.path.exists(base_texture):
            base_texture = os.path.join(self.config.input_dir, record.filepath)
            if self.config.upscale.enabled:
                reason = upscale_result.get("reason") or "upscale output missing"
                stats["base_fallbacks"].append((record.filepath, reason))

        copies = {
            "": base_texture if os.path.exists(base_texture) else None,
            "_albedo": (
                res.get("color_grading", {}).get("graded")
                or res.get("color_consistency", {}).get("corrected")
                or res.get("seam_repair", {}).get("albedo_repaired")
                or res.get("pbr", {}).get("albedo")
            ),
            "_roughness": (res.get("specular_aa", {}).get("roughness_aa") or
                           res.get("pbr", {}).get("roughness")),
            "_metalness": res.get("pbr", {}).get("metalness"),
            "_ao": res.get("pbr", {}).get("ao"),
            "_gloss": res.get("pbr", {}).get("gloss"),
            "_normal": (res.get("detail_map", {}).get("normal_detailed") or
                        res.get("normal", {}).get("normal")),
            "_height": (res.get("pom", {}).get("height_refined") or
                        res.get("normal", {}).get("height")),
            "_orm": res.get("orm", {}).get("orm"),
            "_emissive": res.get("emissive", {}).get("emissive"),
            "_envmask": res.get("reflection_mask", {}).get("env_mask"),
            "_zones": res.get("pbr", {}).get("zone_mask"),
            "_emissive_mask": res.get("emissive", {}).get("emissive_mask"),
        }

        # Mip chain data from Phase 6 (if available)
        mip_data = res.get("mipmap", {})

        # Per-record copy of suffix_to_type with base type set for this record
        local_suffix_to_type = dict(suffix_to_type)
        local_suffix_to_type[""] = TextureType(record.texture_type).value

        for suffix, src_path in copies.items():
            if not self._is_selected_map_suffix(suffix):
                continue
            if not src_path or not os.path.exists(src_path):
                continue

            try:
                tex_type = local_suffix_to_type.get(suffix, "unknown")
                is_srgb = tex_type in set(self.config.compression.srgb_texture_types)
                bc_format = self.config.compression.format_map.get(tex_type, "bc7")
                # Offer BC3 path for base RGBA textures when explicitly configured
                # or when no alpha-specific override is provided.
                if (
                    suffix == ""
                    and (
                        record.has_alpha
                        or src_path == res.get("orm", {}).get("diffuse_alpha_packed")
                    )
                ):
                    alpha_key = f"{tex_type}_alpha"
                    fallback_alpha_bc = (
                        "bc3"
                        if tex_type in {
                            "diffuse", "albedo", "specular",
                            "emissive", "ui", "unknown",
                        }
                        else bc_format
                    )
                    bc_format = self.config.compression.format_map.get(
                        alpha_key,
                        fallback_alpha_bc,
                    )

                # Always write a PNG intermediate (needed as source for
                # DDS/KTX2 compression and as the base for conversion).
                png_dst = get_output_path(
                    record.filepath, self.config.output_dir,
                    suffix=suffix, ext=".png"
                )
                os.makedirs(os.path.dirname(png_dst) or ".", exist_ok=True)
                conversion_img = None

                # Alpha propagation: re-attach original alpha channel to
                # the base texture output when the input had meaningful
                # alpha. Processing phases work on RGB only; this ensures
                # the alpha survives to the final output.
                need_alpha_merge = (
                    suffix == ""
                    and record.has_alpha
                    and src_path != res.get("orm", {}).get("diffuse_alpha_packed")
                )
                if need_alpha_merge:
                    out_img = self._copy_load_image(
                        src_path,
                        max_pixels=self.config.max_image_pixels,
                    )
                    # Prefer alpha from the upscaled output (the upscale
                    # phase already handles alpha with adaptive
                    # interpolation).  Only fall back to the original
                    # input's alpha when the processed output lost it
                    # (e.g. phase skipped or RGB-only intermediate).
                    alpha = extract_alpha(out_img)
                    if alpha is None:
                        original_input = os.path.join(
                            self.config.input_dir, record.filepath
                        )
                        orig_img = self._copy_load_image(
                            original_input,
                            max_pixels=self.config.max_image_pixels,
                        )
                        alpha = extract_alpha(orig_img)
                        del orig_img
                    if alpha is not None:
                        # Resize alpha if dimensions changed (e.g. upscale)
                        oh, ow = out_img.shape[:2]
                        if alpha.shape[:2] != (oh, ow):
                            try:
                                import cv2 as _cv2
                                # Adaptive interpolation: NEAREST for binary
                                # cutouts, AREA for downscale, CUBIC for upscale
                                _is_binary = (np.unique(alpha).size <= 2)
                                if _is_binary:
                                    _alpha_interp = _cv2.INTER_NEAREST
                                elif oh < alpha.shape[0] or ow < alpha.shape[1]:
                                    _alpha_interp = _cv2.INTER_AREA
                                else:
                                    _alpha_interp = _cv2.INTER_CUBIC
                                alpha = _cv2.resize(
                                    alpha, (ow, oh),
                                    interpolation=_alpha_interp,
                                )
                            except ImportError:
                                _a_img = Image.fromarray(
                                    (np.clip(alpha, 0, 1) * 255).astype(
                                        np.uint8
                                    )
                                )
                                _a_img = _a_img.resize(
                                    (ow, oh), Image.BILINEAR
                                )
                                alpha = (
                                    np.asarray(_a_img, dtype=np.float32)
                                    / 255.0
                                )
                        conversion_img = merge_alpha(out_img, alpha)
                    else:
                        conversion_img = out_img
                    del out_img, alpha
                    self._copy_save_image(conversion_img, png_dst)
                elif src_path.lower().endswith(".png"):
                    # Do not preserve source mtime; output rewrite should
                    # reflect the current run timestamp.
                    shutil.copyfile(src_path, png_dst)
                else:
                    conversion_img = self._copy_load_image(
                        src_path,
                        max_pixels=self.config.max_image_pixels
                    )
                    self._copy_save_image(conversion_img, png_dst)

                # Track whether PNG is just a temp file to clean up
                png_is_temp = (output_ext != ".png")

                # ----- Write primary output matching input format -----
                if output_ext == ".png":
                    # PNG already written above
                    stats["output_count"] += 1

                elif output_ext == ".dds":
                    dds_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".dds"
                    )
                    if self._mipmap_gen is not None:
                        mip_key = suffix_to_mip_key.get(suffix)
                        mip_info = (
                            mip_data.get(mip_key, {})
                            if isinstance(mip_data, dict) else {}
                        )
                        mip_paths = mip_info.get("mips", [])
                        use_mipchain = bool(mip_paths and not mip_info.get("error"))
                        source_size = self._compression_source_size(
                            png_dst, mip_paths, use_mipchain
                        )
                        mode = "mipchain" if use_mipchain else "single"

                        stats["dds_attempted"] += 1
                        self._log_compression_start(
                            "DDS",
                            "primary",
                            record,
                            suffix,
                            mode,
                            png_dst,
                            dds_dst,
                            bc_format,
                        )
                        degraded = False
                        if use_mipchain:
                            ok, degraded = self._mipmap_gen.generate_dds_mipchain(
                                mip_paths, dds_dst, bc_format, srgb=is_srgb
                            )
                        else:
                            ok = self._mipmap_gen.generate_dds(
                                png_dst, dds_dst, bc_format, srgb=is_srgb
                            )
                        if ok:
                            stats["output_count"] += 1
                            status = "degraded" if degraded else "ok"
                            if status == "ok":
                                stats["dds_count"] += 1
                            dst_size = self._log_compression_result(
                                "DDS",
                                "primary",
                                record,
                                suffix,
                                status,
                                source_size,
                                dds_dst,
                            )
                            _accumulate("DDS", source_size, dst_size)
                            if status == "degraded":
                                stats["dds_degraded"] += 1
                                self._record_copy_failure(
                                    record,
                                    f"DDS primary mipchain degraded for "
                                    f"{record.filename}{suffix} "
                                    f"(base-only fallback)",
                                )
                        else:
                            self._log_compression_result(
                                "DDS",
                                "primary",
                                record,
                                suffix,
                                "failed",
                                source_size,
                                dds_dst,
                                "tool_failed",
                            )
                            if self.config.compression.fail_on_fallback:
                                self._record_copy_failure(
                                    record,
                                    f"DDS primary compression failed for "
                                    f"{record.filename}{suffix} "
                                    f"(fail_on_fallback=True, PNG not shipped)",
                                )
                                logger.error(
                                    f"DDS compression failed for "
                                    f"{record.filename}{suffix} and "
                                    f"fail_on_fallback is enabled — "
                                    f"not shipping PNG fallback"
                                )
                            else:
                                self._record_copy_failure(
                                    record,
                                    f"DDS primary compression failed for "
                                    f"{record.filename}{suffix}",
                                )
                                logger.warning(
                                    f"DDS compression failed for "
                                    f"{record.filename}{suffix}, "
                                    f"shipped as PNG fallback"
                                )
                                png_is_temp = False
                                stats["output_count"] += 1
                                stats["compression_fallbacks"] += 1
                    else:
                        self._log_compression_result(
                            "DDS",
                            "primary",
                            record,
                            suffix,
                            "skipped",
                            self._safe_getsize(png_dst),
                            dds_dst,
                            "compression_tool_unavailable",
                        )
                        if self.config.compression.fail_on_fallback:
                            self._record_copy_failure(
                                record,
                                f"DDS primary output unavailable for "
                                f"{record.filename}{suffix} "
                                f"(compression tool missing, "
                                f"fail_on_fallback=True)",
                            )
                            logger.error(
                                f"Cannot generate DDS for "
                                f"{record.filename}{suffix} "
                                f"(no compression tool) and "
                                f"fail_on_fallback is enabled"
                            )
                        else:
                            self._record_copy_failure(
                                record,
                                f"DDS primary output unavailable for "
                                f"{record.filename}{suffix} "
                                f"(compression tool missing)",
                            )
                            logger.warning(
                                f"Cannot generate DDS for "
                                f"{record.filename}{suffix} "
                                f"(no compression tool), shipped as PNG"
                            )
                            png_is_temp = False
                            stats["output_count"] += 1
                            stats["compression_fallbacks"] += 1

                elif output_ext == ".ktx2":
                    ktx2_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".ktx2"
                    )
                    if self._mipmap_gen is not None:
                        mip_key = suffix_to_mip_key.get(suffix)
                        mip_info = (
                            mip_data.get(mip_key, {})
                            if isinstance(mip_data, dict) else {}
                        )
                        mip_paths = mip_info.get("mips", [])
                        use_mipchain = bool(mip_paths and not mip_info.get("error"))
                        source_size = self._compression_source_size(
                            png_dst, mip_paths, use_mipchain
                        )
                        mode = "mipchain" if use_mipchain else "single"

                        stats["ktx2_attempted"] += 1
                        self._log_compression_start(
                            "KTX2",
                            "primary",
                            record,
                            suffix,
                            mode,
                            png_dst,
                            ktx2_dst,
                        )
                        if use_mipchain:
                            ok = self._mipmap_gen.generate_ktx2_mipchain(
                                mip_paths, ktx2_dst, srgb=is_srgb
                            )
                        else:
                            ok = self._mipmap_gen.generate_ktx2(
                                png_dst, ktx2_dst, srgb=is_srgb
                            )
                        if ok:
                            stats["ktx2_count"] += 1
                            stats["output_count"] += 1
                            dst_size = self._log_compression_result(
                                "KTX2",
                                "primary",
                                record,
                                suffix,
                                "ok",
                                source_size,
                                ktx2_dst,
                            )
                            _accumulate("KTX2", source_size, dst_size)
                        else:
                            self._log_compression_result(
                                "KTX2",
                                "primary",
                                record,
                                suffix,
                                "failed",
                                source_size,
                                ktx2_dst,
                                "tool_failed",
                            )
                            if self.config.compression.fail_on_fallback:
                                self._record_copy_failure(
                                    record,
                                    f"KTX2 primary compression failed for "
                                    f"{record.filename}{suffix} "
                                    f"(fail_on_fallback=True, PNG not shipped)",
                                )
                                logger.error(
                                    f"KTX2 compression failed for "
                                    f"{record.filename}{suffix} and "
                                    f"fail_on_fallback is enabled — "
                                    f"not shipping PNG fallback"
                                )
                            else:
                                self._record_copy_failure(
                                    record,
                                    f"KTX2 primary compression failed for "
                                    f"{record.filename}{suffix}",
                                )
                                logger.warning(
                                    f"KTX2 compression failed for "
                                    f"{record.filename}{suffix}, "
                                    f"shipped as PNG fallback"
                                )
                                png_is_temp = False
                                stats["output_count"] += 1
                                stats["compression_fallbacks"] += 1
                    else:
                        self._log_compression_result(
                            "KTX2",
                            "primary",
                            record,
                            suffix,
                            "skipped",
                            self._safe_getsize(png_dst),
                            ktx2_dst,
                            "compression_tool_unavailable",
                        )
                        if self.config.compression.fail_on_fallback:
                            self._record_copy_failure(
                                record,
                                f"KTX2 primary output unavailable for "
                                f"{record.filename}{suffix} "
                                f"(compression tool missing, "
                                f"fail_on_fallback=True)",
                            )
                            logger.error(
                                f"Cannot generate KTX2 for "
                                f"{record.filename}{suffix} "
                                f"(no compression tool) and "
                                f"fail_on_fallback is enabled"
                            )
                        else:
                            self._record_copy_failure(
                                record,
                                f"KTX2 primary output unavailable for "
                                f"{record.filename}{suffix} "
                                f"(compression tool missing)",
                            )
                            logger.warning(
                                f"Cannot generate KTX2 for "
                                f"{record.filename}{suffix} "
                                f"(no compression tool), shipped as PNG"
                            )
                            png_is_temp = False
                            stats["output_count"] += 1
                            stats["compression_fallbacks"] += 1

                elif output_ext == ".tga":
                    tga_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".tga"
                    )
                    if conversion_img is None:
                        conversion_img = self._copy_load_image(
                            src_path,
                            max_pixels=self.config.max_image_pixels
                        )
                    self._copy_save_image(conversion_img, tga_dst)
                    stats["tga_count"] += 1
                    stats["output_count"] += 1

                else:
                    # Other formats (jpg, bmp, tiff, etc.)
                    other_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=output_ext
                    )
                    if conversion_img is None:
                        conversion_img = self._copy_load_image(
                            src_path,
                            max_pixels=self.config.max_image_pixels
                        )
                    self._copy_save_image(conversion_img, other_dst)
                    stats["output_count"] += 1

                # ----- Additional format generation -----
                # Skip when primary output already matches the format.

                # Additional DDS
                if generate_dds and output_ext != ".dds":
                    dds_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".dds"
                    )
                    mip_key = suffix_to_mip_key.get(suffix)
                    mip_info = (
                        mip_data.get(mip_key, {})
                        if isinstance(mip_data, dict) else {}
                    )
                    mip_paths = mip_info.get("mips", [])
                    use_mipchain = bool(mip_paths and not mip_info.get("error"))
                    source_size = self._compression_source_size(
                        png_dst, mip_paths, use_mipchain
                    )
                    mode = "mipchain" if use_mipchain else "single"

                    stats["dds_attempted"] += 1
                    self._log_compression_start(
                        "DDS",
                        "extra",
                        record,
                        suffix,
                        mode,
                        png_dst,
                        dds_dst,
                        bc_format,
                    )
                    if self._mipmap_gen is None:
                        self._log_compression_result(
                            "DDS",
                            "extra",
                            record,
                            suffix,
                            "skipped",
                            source_size,
                            dds_dst,
                            "compression_tool_unavailable",
                        )
                        self._record_copy_failure(
                            record,
                            f"DDS extra generation unavailable for "
                            f"{record.filename}{suffix} "
                            f"(compression tool missing)",
                        )
                    else:
                        degraded = False
                        if use_mipchain:
                            ok, degraded = self._mipmap_gen.generate_dds_mipchain(
                                mip_paths, dds_dst, bc_format, srgb=is_srgb
                            )
                        else:
                            ok = self._mipmap_gen.generate_dds(
                                png_dst, dds_dst, bc_format, srgb=is_srgb
                            )
                        if ok:
                            status = "degraded" if degraded else "ok"
                            if status == "ok":
                                stats["dds_count"] += 1
                            dst_size = self._log_compression_result(
                                "DDS",
                                "extra",
                                record,
                                suffix,
                                status,
                                source_size,
                                dds_dst,
                            )
                            _accumulate("DDS", source_size, dst_size)
                            if status == "degraded":
                                stats["dds_degraded"] += 1
                                self._record_copy_failure(
                                    record,
                                    f"DDS extra mipchain degraded for "
                                    f"{record.filename}{suffix} "
                                    f"(base-only fallback)",
                                )
                        else:
                            self._log_compression_result(
                                "DDS",
                                "extra",
                                record,
                                suffix,
                                "failed",
                                source_size,
                                dds_dst,
                                "tool_failed",
                            )
                            self._record_copy_failure(
                                record,
                                f"DDS extra generation failed for "
                                f"{record.filename}{suffix}",
                            )

                # Additional KTX2
                if generate_ktx2 and output_ext != ".ktx2":
                    ktx2_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".ktx2"
                    )
                    mip_key = suffix_to_mip_key.get(suffix)
                    mip_info = (
                        mip_data.get(mip_key, {})
                        if isinstance(mip_data, dict) else {}
                    )
                    mip_paths = mip_info.get("mips", [])
                    use_mipchain = bool(mip_paths and not mip_info.get("error"))
                    source_size = self._compression_source_size(
                        png_dst, mip_paths, use_mipchain
                    )
                    mode = "mipchain" if use_mipchain else "single"

                    stats["ktx2_attempted"] += 1
                    self._log_compression_start(
                        "KTX2",
                        "extra",
                        record,
                        suffix,
                        mode,
                        png_dst,
                        ktx2_dst,
                    )
                    if self._mipmap_gen is None:
                        self._log_compression_result(
                            "KTX2",
                            "extra",
                            record,
                            suffix,
                            "skipped",
                            source_size,
                            ktx2_dst,
                            "compression_tool_unavailable",
                        )
                        self._record_copy_failure(
                            record,
                            f"KTX2 extra generation unavailable for "
                            f"{record.filename}{suffix} "
                            f"(compression tool missing)",
                        )
                    else:
                        if use_mipchain:
                            ok = self._mipmap_gen.generate_ktx2_mipchain(
                                mip_paths, ktx2_dst, srgb=is_srgb
                            )
                        else:
                            ok = self._mipmap_gen.generate_ktx2(
                                png_dst, ktx2_dst, srgb=is_srgb
                            )
                        if ok:
                            stats["ktx2_count"] += 1
                            dst_size = self._log_compression_result(
                                "KTX2",
                                "extra",
                                record,
                                suffix,
                                "ok",
                                source_size,
                                ktx2_dst,
                            )
                            _accumulate("KTX2", source_size, dst_size)
                        else:
                            self._log_compression_result(
                                "KTX2",
                                "extra",
                                record,
                                suffix,
                                "failed",
                                source_size,
                                ktx2_dst,
                                "tool_failed",
                            )
                            self._record_copy_failure(
                                record,
                                f"KTX2 extra generation failed for "
                                f"{record.filename}{suffix}",
                            )

                # Additional TGA
                if generate_tga and output_ext != ".tga":
                    tga_dst = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".tga"
                    )
                    if conversion_img is None:
                        conversion_img = self._copy_load_image(
                            src_path,
                            max_pixels=self.config.max_image_pixels
                        )
                    self._copy_save_image(conversion_img, tga_dst)
                    stats["tga_count"] += 1

                # Clean up temp PNG if it's not the primary format
                if png_is_temp and os.path.exists(png_dst):
                    os.remove(png_dst)

            except Exception as e:
                # Clean up orphaned temp PNG on error
                try:
                    if png_is_temp and png_dst and os.path.exists(png_dst):
                        os.remove(png_dst)
                except (NameError, OSError):
                    pass
                logger.error(
                    f"Failed to copy {record.filename}{suffix}: {e}"
                )
                # Record copy error so it's reflected in _failed_assets
                with self._results_lock:
                    if record.filepath not in self.results:
                        self.results[record.filepath] = {}
                    self.results[record.filepath]["copy"] = {
                        "error": str(e)
                    }

        return stats

    # ------------------------------------------
    # Output Validation
    # ------------------------------------------

    def _validate_outputs(self):
        """Validate final output files after copy_to_output.

        Checks PNG files for corruption, degenerate content, and consistency.
        Checks DDS/KTX2 files for correct magic bytes.
        Gated by config.validation.enabled and dry_run.
        """
        if not self.config.validation.enabled or self.config.dry_run:
            if not self.config.validation.enabled:
                logger.debug("Output validation skipped (pipeline validation disabled).")
            else:
                logger.debug("Output validation skipped (dry-run mode).")
            return

        logger.info("=" * 60)
        logger.info("OUTPUT VALIDATION: Checking final output files")
        logger.info("=" * 60)

        all_issues = []
        total_checked = 0
        total_passed = 0

        suffixes = [
            "",
            "_albedo",
            "_roughness",
            "_metalness",
            "_ao",
            "_gloss",
            "_normal",
            "_height",
            "_orm",
            "_emissive",
            "_emissive_mask",
            "_envmask",
            "_zones",
        ]

        for record in self.records:
            asset_dims = {}
            asset_issues = []
            output_ext = _get_output_ext(record)
            checked_paths = set()

            for suffix in suffixes:
                # Check primary output format for this record
                primary_path = get_output_path(
                    record.filepath, self.config.output_dir,
                    suffix=suffix, ext=output_ext
                )
                if (os.path.exists(primary_path)
                        and primary_path not in checked_paths):
                    checked_paths.add(primary_path)
                    total_checked += 1
                    if output_ext in (".dds", ".ktx2"):
                        fmt = "dds" if output_ext == ".dds" else "ktx2"
                        issues = self._validate_compressed_output(
                            primary_path, fmt, suffix, record, asset_dims
                        )
                    else:
                        issues = self._validate_single_output(
                            primary_path, suffix, record, asset_dims
                        )
                    if issues:
                        asset_issues.extend(issues)
                    else:
                        total_passed += 1

                # Check additional PNG (if not the primary format)
                if output_ext != ".png":
                    png_path = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".png"
                    )
                    if (os.path.exists(png_path)
                            and png_path not in checked_paths):
                        checked_paths.add(png_path)
                        total_checked += 1
                        issues = self._validate_single_output(
                            png_path, suffix, record, asset_dims
                        )
                        if issues:
                            asset_issues.extend(issues)
                        else:
                            total_passed += 1

                # Check additional DDS (if not the primary format)
                if output_ext != ".dds":
                    dds_path = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".dds"
                    )
                    if (os.path.exists(dds_path)
                            and dds_path not in checked_paths):
                        checked_paths.add(dds_path)
                        total_checked += 1
                        issues = self._validate_compressed_output(
                            dds_path, "dds", suffix, record, asset_dims
                        )
                        if issues:
                            asset_issues.extend(issues)
                        else:
                            total_passed += 1

                # Check additional KTX2 (if not the primary format)
                if output_ext != ".ktx2":
                    ktx2_path = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".ktx2"
                    )
                    if (os.path.exists(ktx2_path)
                            and ktx2_path not in checked_paths):
                        checked_paths.add(ktx2_path)
                        total_checked += 1
                        issues = self._validate_compressed_output(
                            ktx2_path, "ktx2", suffix, record, asset_dims
                        )
                        if issues:
                            asset_issues.extend(issues)
                        else:
                            total_passed += 1

                # Check additional TGA (if not the primary format)
                if output_ext != ".tga":
                    tga_path = get_output_path(
                        record.filepath, self.config.output_dir,
                        suffix=suffix, ext=".tga"
                    )
                    if (os.path.exists(tga_path)
                            and tga_path not in checked_paths):
                        checked_paths.add(tga_path)
                        total_checked += 1
                        issues = self._validate_single_output(
                            tga_path, suffix, record, asset_dims
                        )
                        if issues:
                            asset_issues.extend(issues)
                        else:
                            total_passed += 1

            # Cross-suffix dimension consistency
            dim_issues = self._check_dimension_consistency(record, asset_dims)
            asset_issues.extend(dim_issues)

            # Store per-asset results
            with self._results_lock:
                self.results[record.filepath]["output_validation"] = {
                    "checked": len([s for s in suffixes
                                    if os.path.exists(get_output_path(
                                        record.filepath, self.config.output_dir,
                                        suffix=s, ext=output_ext))]),
                    "issues": asset_issues,
                    "passed": len(asset_issues) == 0,
                }

            all_issues.extend(asset_issues)

            for issue in asset_issues:
                logger.warning(
                    f"[output_validation] {issue['file']}: "
                    f"{issue['check']} - {issue['message']}"
                )

        # Write report
        report = {
            "total_files_checked": total_checked,
            "passed": total_passed,
            "failed": total_checked - total_passed,
            "issues": all_issues,
        }
        report_path = os.path.join(
            self.config.output_dir, "output_validation.json"
        )
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        tmp_report_path = (
            f"{report_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_report_path, report_path)
        finally:
            if os.path.exists(tmp_report_path):
                try:
                    os.remove(tmp_report_path)
                except OSError:
                    pass

        affected = sum(
            1 for r in self.records
            if self.results.get(r.filepath, {})
            .get("output_validation", {}).get("issues")
        )
        logger.info(
            f"Output validation: {total_passed}/{total_checked} files passed, "
            f"{len(all_issues)} issue(s) across {affected} asset(s)"
        )
        logger.info(f"Report: {report_path}")

    def _validate_single_output(self, path, suffix, record, asset_dims):
        """Validate a single image output file (PNG, TGA, JPG, BMP, etc.).

        Returns list of issue dicts (empty if all checks pass).
        """
        issues = []
        filename = os.path.basename(path)
        ext = Path(path).suffix.lower()

        # 1. Non-zero file size
        if os.path.getsize(path) == 0:
            issues.append({
                "file": filename,
                "check": "non_zero_size",
                "message": "File is zero bytes",
            })
            return issues  # Can't do further checks

        # 2. Format-specific magic bytes check
        with open(path, "rb") as f:
            magic = f.read(8)
        if ext == ".png" and magic[:4] != b"\x89PNG":
            issues.append({
                "file": filename,
                "check": "magic_bytes",
                "message": f"Expected PNG magic bytes, got {magic[:4]!r}",
            })
            return issues
        elif ext in (".jpg", ".jpeg") and magic[:2] != b"\xff\xd8":
            issues.append({
                "file": filename,
                "check": "magic_bytes",
                "message": f"Expected JPEG magic bytes, got {magic[:2]!r}",
            })
            return issues
        elif ext == ".bmp" and magic[:2] != b"BM":
            issues.append({
                "file": filename,
                "check": "magic_bytes",
                "message": f"Expected BMP magic bytes, got {magic[:2]!r}",
            })
            return issues
        # TGA has no reliable magic bytes - rely on Pillow load check below

        # 3. Loadable by Pillow
        try:
            with Image.open(path) as img:
                img.load()
                w, h = img.size
                arr_raw = np.array(img)
        except Exception as e:
            issues.append({
                "file": filename,
                "check": "loadable",
                "message": f"Failed to load: {e}",
            })
            return issues

        # 4. Power-of-two dimensions (only flag when PoT is enforced)
        if self.config.upscale.enforce_power_of_two:
            if w > 0 and (w & (w - 1)) != 0:
                issues.append({
                    "file": filename,
                    "check": "power_of_two",
                    "message": f"Width {w} is not a power of two",
                })
            if h > 0 and (h & (h - 1)) != 0:
                issues.append({
                    "file": filename,
                    "check": "power_of_two",
                    "message": f"Height {h} is not a power of two",
                })

        # 5. Not solid color (std dev > 0.005)
        # Skip for suffixes that are often intentionally flat/uniform
        flat_ok_suffixes = {
            "_roughness", "_metalness", "_ao", "_orm", "_opacity",
            "_mask", "_height", "_gloss", "_envmask", "_zones",
            "_emissive_mask", "_emissive",
        }
        arr = arr_raw.astype(np.float32)
        if arr.size == 0:
            arr_norm = arr
        elif np.issubdtype(arr_raw.dtype, np.integer):
            max_val = float(np.iinfo(arr_raw.dtype).max)
            arr_norm = arr / max_val if max_val > 0 else arr
        else:
            arr_max = float(arr.max())
            if arr_max > 1.0:
                arr_norm = arr / (65535.0 if arr_max > 255.0 else 255.0)
            else:
                arr_norm = arr
        if suffix not in flat_ok_suffixes and np.std(arr_norm) < 0.005:
            issues.append({
                "file": filename,
                "check": "solid_color",
                "message": f"Image appears to be solid color (std={np.std(arr_norm):.6f})",
            })

        # Track dimensions for consistency check
        asset_dims[suffix] = (w, h)

        # 6. Normal integrity (suffix _normal only)
        if suffix == "_normal" and len(arr.shape) >= 3 and arr.shape[2] >= 3:
            rgb = arr_norm
            if len(rgb.shape) == 3:
                normal_issues = self._check_output_normal_integrity(
                    rgb[:, :, :3], filename
                )
                issues.extend(normal_issues)

        return issues

    def _validate_compressed_output(
        self, path, fmt, suffix, record, asset_dims=None
    ):
        """Validate a DDS or KTX2 compressed output file.

        Returns list of issue dicts.
        """
        issues = []
        filename = os.path.basename(path)
        file_size = self._safe_getsize(path)

        # 1. Non-zero size
        if file_size == 0:
            issues.append({
                "file": filename,
                "check": "non_zero_size",
                "message": "File is zero bytes",
            })
            return issues

        # 2. Magic bytes + container-level structural checks.
        raw = b""
        with open(path, "rb") as f:
            if fmt == "dds":
                raw = f.read(148)  # DDS + optional DX10 header
                magic = raw[:4]
                expected = b"DDS "
            else:  # ktx2
                # KTX2 fixed header is 80 bytes; level index follows.
                raw = f.read(80)
                if len(raw) >= 44:
                    level_count = self._u32_le(raw, 40)
                    index_bytes = max(24 * max(level_count, 0), 0)
                    available = max(file_size - len(raw), 0)
                    to_read = min(index_bytes, available)
                    if to_read > 0:
                        raw += f.read(to_read)
                magic = raw[:12]
                expected = b"\xabKTX 20\xbb\r\n\x1a\n"

        if magic != expected:
            issues.append({
                "file": filename,
                "check": "magic_bytes",
                "message": f"Expected {fmt.upper()} magic bytes, got {magic!r}",
            })
            return issues

        if fmt == "dds":
            width = self._u32_le(raw, 16)
            height = self._u32_le(raw, 12)
            require_mips = (
                bool(self.config.validation.require_full_mipchain)
                and bool(self.config.mipmap.enabled)
                and "mipmap" in self._executed_phases
                and max(width, height) > 1
            )
            if isinstance(asset_dims, dict) and width > 0 and height > 0:
                asset_dims[suffix] = (width, height)
            issues.extend(
                self._validate_dds_header(
                    raw,
                    file_size,
                    filename,
                    require_mips=require_mips,
                )
            )

            expected_codec = self._expected_dds_codec_for_output(suffix, record)
            actual_codec = self._dds_codec_from_header(raw)
            if expected_codec and expected_codec != "none":
                if actual_codec is None:
                    issues.append({
                        "file": filename,
                        "check": "dds_codec",
                        "message": (
                            f"DDS codec could not be determined; expected "
                            f"{expected_codec.upper()}"
                        ),
                    })
                elif actual_codec != expected_codec:
                    issues.append({
                        "file": filename,
                        "check": "dds_codec",
                        "message": (
                            f"DDS codec mismatch: expected "
                            f"{expected_codec.upper()}, got {actual_codec.upper()}"
                        ),
                    })

            # 3. sRGB expectation check via DX10 header DXGI format
            fourcc = raw[84:88] if len(raw) >= 88 else b""
            if fourcc == b"DX10" and len(raw) >= 148:
                dxgi_format = self._u32_le(raw, 128)
                # Determine sRGB expectation from suffix -> texture type
                _suffix_type_map = {
                    "": "unknown",
                    "_albedo": "albedo",
                    "_roughness": "roughness",
                    "_metalness": "metalness",
                    "_ao": "ao",
                    "_gloss": "roughness",
                    "_normal": "normal",
                    "_height": "height",
                    "_orm": "orm",
                    "_emissive": "emissive",
                    "_envmask": "mask",
                    "_zones": "mask",
                    "_emissive_mask": "mask",
                }
                tex_type = _suffix_type_map.get(suffix, "unknown")
                if suffix == "" and record is not None:
                    try:
                        tex_type = TextureType(record.texture_type).value
                    except (ValueError, KeyError):
                        pass
                srgb_types = set(self.config.compression.srgb_texture_types)
                expect_srgb = tex_type in srgb_types
                is_srgb_format = dxgi_format in _DXGI_SRGB_FORMATS
                if expect_srgb and not is_srgb_format:
                    logger.warning(
                        "DDS sRGB mismatch for %s: texture type '%s' expects "
                        "sRGB but DXGI format %d is not sRGB.",
                        filename, tex_type, dxgi_format,
                    )
                elif not expect_srgb and is_srgb_format:
                    logger.warning(
                        "DDS sRGB mismatch for %s: texture type '%s' does not "
                        "expect sRGB but DXGI format %d is sRGB.",
                        filename, tex_type, dxgi_format,
                    )
        else:
            width = self._u32_le(raw, 20)
            height = self._u32_le(raw, 24)
            require_mips = (
                bool(self.config.validation.require_full_mipchain)
                and bool(self.config.mipmap.enabled)
                and "mipmap" in self._executed_phases
                and max(width, height) > 1
            )
            if isinstance(asset_dims, dict) and width > 0 and height > 0:
                asset_dims[suffix] = (width, height)
            issues.extend(
                self._validate_ktx2_header(
                    raw,
                    file_size,
                    filename,
                    require_mips=require_mips,
                )
            )

        return issues

    def _expected_dds_codec_for_output(self, suffix: str, record) -> Optional[str]:
        """Resolve expected DDS codec (bc1..bc7) for a given output suffix."""
        suffix_type_map = {
            "": "unknown",
            "_albedo": "albedo",
            "_roughness": "roughness",
            "_metalness": "metalness",
            "_ao": "ao",
            "_gloss": "roughness",
            "_normal": "normal",
            "_height": "height",
            "_orm": "orm",
            "_emissive": "emissive",
            "_envmask": "mask",
            "_zones": "mask",
            "_emissive_mask": "mask",
        }
        tex_type = suffix_type_map.get(suffix, "unknown")
        if suffix == "" and record is not None:
            try:
                tex_type = TextureType(record.texture_type).value
            except (ValueError, KeyError):
                tex_type = "unknown"

        bc_format = self.config.compression.format_map.get(tex_type, "bc7")
        if suffix == "" and record is not None and getattr(record, "has_alpha", False):
            alpha_key = f"{tex_type}_alpha"
            fallback_alpha_bc = (
                "bc3"
                if tex_type in {
                    "diffuse", "albedo", "specular", "emissive", "ui", "unknown",
                }
                else bc_format
            )
            bc_format = self.config.compression.format_map.get(
                alpha_key, fallback_alpha_bc
            )
        return str(bc_format).lower()

    def _dds_codec_from_header(self, raw: bytes) -> Optional[str]:
        """Return DDS codec id (bc1..bc7) from header, or None if unknown."""
        if len(raw) < 88:
            return None
        fourcc = raw[84:88]
        if fourcc == b"DX10":
            if len(raw) < 148:
                return None
            dxgi = self._u32_le(raw, 128)
            return _DDS_CODEC_FROM_DXGI.get(dxgi)
        return _DDS_CODEC_FROM_FOURCC.get(fourcc)

    @staticmethod
    def _u32_le(raw: bytes, offset: int) -> int:
        import struct

        if offset + 4 > len(raw):
            return 0
        return int(struct.unpack_from("<I", raw, offset)[0])

    @staticmethod
    def _u64_le(raw: bytes, offset: int) -> int:
        import struct

        if offset + 8 > len(raw):
            return 0
        return int(struct.unpack_from("<Q", raw, offset)[0])

    def _validate_dds_header(
        self, raw: bytes, file_size: int, filename: str, require_mips: bool = False
    ) -> list:
        """Perform structural validation of DDS header fields."""
        issues = []
        if len(raw) < 128:
            issues.append({
                "file": filename,
                "check": "dds_header",
                "message": f"DDS header truncated (size={len(raw)} bytes)",
            })
            return issues

        dds_header_size = self._u32_le(raw, 4)
        dds_flags = self._u32_le(raw, 8)
        height = self._u32_le(raw, 12)
        width = self._u32_le(raw, 16)
        mip_count = self._u32_le(raw, 28)
        pf_size = self._u32_le(raw, 76)
        # raw includes the 4-byte "DDS " magic, so dwCaps (header offset 104)
        # lives at absolute offset 108.
        caps = self._u32_le(raw, 108)
        fourcc = raw[84:88]

        if dds_header_size != 124:
            issues.append({
                "file": filename,
                "check": "dds_header",
                "message": f"Invalid DDS header size: {dds_header_size} (expected 124)",
            })
        if pf_size != 32:
            issues.append({
                "file": filename,
                "check": "dds_pixel_format",
                "message": f"Invalid DDS pixel format size: {pf_size} (expected 32)",
            })
        required_flags = 0x1 | 0x2 | 0x4 | 0x1000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT
        missing_flags = required_flags & ~dds_flags
        if missing_flags:
            issues.append({
                "file": filename,
                "check": "dds_flags",
                "message": (
                    f"DDS header flags missing required bits: "
                    f"flags=0x{dds_flags:08X}, missing=0x{missing_flags:08X}"
                ),
            })
        if not (caps & 0x00001000):  # DDSCAPS_TEXTURE
            issues.append({
                "file": filename,
                "check": "dds_caps_texture",
                "message": (
                    f"DDS caps missing DDSCAPS_TEXTURE bit: caps=0x{caps:08X}"
                ),
            })
        if width <= 0 or height <= 0:
            issues.append({
                "file": filename,
                "check": "dds_dimensions",
                "message": f"Invalid DDS dimensions: {width}x{height}",
            })

        has_mip_flag = bool(dds_flags & 0x00020000)
        if has_mip_flag and mip_count == 0:
            issues.append({
                "file": filename,
                "check": "dds_mip_count",
                "message": "DDS declares mipmap flag but mip count is 0",
            })
        if mip_count > 0 and max(width, height) > 0:
            max_possible = int(max(width, height)).bit_length()
            if mip_count > max_possible:
                issues.append({
                    "file": filename,
                    "check": "dds_mip_count",
                    "message": (
                        f"DDS mip count {mip_count} exceeds plausible max "
                        f"{max_possible} for {width}x{height}"
                    ),
                })
        if require_mips and mip_count <= 1:
            issues.append({
                "file": filename,
                "check": "dds_mipchain",
                "message": (
                    "DDS output is missing full mip chain "
                    "(mip_count <= 1 while mipmaps are required)"
                ),
            })

        if (caps & 0x00400000) and mip_count <= 1:
            issues.append({
                "file": filename,
                "check": "dds_caps_mipmap",
                "message": "DDS mipmap caps set but mip count <= 1",
            })

        if fourcc == b"DX10" and file_size < 148:
            issues.append({
                "file": filename,
                "check": "dds_dx10_header",
                "message": f"DDS DX10 header missing/truncated (size={file_size} bytes)",
            })

        return issues

    def _validate_ktx2_header(
        self, raw: bytes, file_size: int, filename: str, require_mips: bool = False
    ) -> list:
        """Perform structural validation of KTX2 header and level index."""
        issues = []
        if len(raw) < 80:
            issues.append({
                "file": filename,
                "check": "ktx2_header",
                "message": f"KTX2 header truncated (size={len(raw)} bytes)",
            })
            return issues

        width = self._u32_le(raw, 20)
        height = self._u32_le(raw, 24)
        face_count = self._u32_le(raw, 36)
        level_count = self._u32_le(raw, 40)

        if width <= 0:
            issues.append({
                "file": filename,
                "check": "ktx2_dimensions",
                "message": f"Invalid KTX2 pixel width: {width}",
            })
        if height <= 0:
            issues.append({
                "file": filename,
                "check": "ktx2_dimensions",
                "message": f"Invalid KTX2 pixel height: {height}",
            })
        if face_count not in (1, 6):
            issues.append({
                "file": filename,
                "check": "ktx2_face_count",
                "message": f"Invalid KTX2 face count: {face_count}",
            })
        if level_count <= 0:
            issues.append({
                "file": filename,
                "check": "ktx2_level_count",
                "message": f"Invalid KTX2 level count: {level_count}",
            })
            return issues
        if require_mips and level_count <= 1:
            issues.append({
                "file": filename,
                "check": "ktx2_mipchain",
                "message": (
                    "KTX2 output is missing full mip chain "
                    "(level_count <= 1 while mipmaps are required)"
                ),
            })

        index_end = 80 + (24 * level_count)
        if file_size < index_end:
            issues.append({
                "file": filename,
                "check": "ktx2_level_index",
                "message": (
                    f"KTX2 level index truncated: requires at least {index_end} bytes, "
                    f"file is {file_size} bytes"
                ),
            })
            return issues

        if len(raw) < index_end:
            issues.append({
                "file": filename,
                "check": "ktx2_level_index",
                "message": (
                    f"KTX2 header read is shorter than required level index: "
                    f"read {len(raw)} bytes, need {index_end} bytes"
                ),
            })
            return issues

        for level in range(level_count):
            offset = 80 + (24 * level)
            level_offset = self._u64_le(raw, offset)
            level_length = self._u64_le(raw, offset + 8)
            if level_length == 0:
                issues.append({
                    "file": filename,
                    "check": "ktx2_level_index",
                    "message": f"KTX2 level {level} has zero byte length",
                })
                continue
            if level_offset + level_length > file_size:
                issues.append({
                    "file": filename,
                    "check": "ktx2_level_index",
                    "message": (
                        f"KTX2 level {level} range [{level_offset}, "
                        f"{level_offset + level_length}) exceeds file size {file_size}"
                    ),
                })

        return issues

    def _check_output_normal_integrity(self, rgb, filename):
        """Check normal map pixel integrity.

        Decodes [0,1] -> [-1,1] and flags if >5% negative-Z or >5% non-unit vectors.
        """
        import numpy as np

        issues = []
        decoded = rgb * 2.0 - 1.0
        total_pixels = decoded.shape[0] * decoded.shape[1]

        # Check Z-channel (should be positive for outward-facing normals)
        z = decoded[:, :, 2]
        negative_z_ratio = np.sum(z < 0) / total_pixels
        if negative_z_ratio > 0.05:
            issues.append({
                "file": filename,
                "check": "normal_integrity",
                "message": f"{negative_z_ratio:.1%} of normals have negative Z",
            })

        # Check unit length
        lengths = np.sqrt(np.sum(decoded ** 2, axis=-1))
        non_unit = np.sum(np.abs(lengths - 1.0) > 0.1) / total_pixels
        if non_unit > 0.05:
            issues.append({
                "file": filename,
                "check": "normal_integrity",
                "message": f"{non_unit:.1%} of normals are non-unit length",
            })

        return issues

    def _check_dimension_consistency(self, record, asset_dims):
        """Check that all output suffixes for an asset have matching dimensions."""
        issues = []
        if len(asset_dims) <= 1:
            return issues

        dims_set = set(asset_dims.values())
        if len(dims_set) > 1:
            detail = ", ".join(
                f"{s or '(base)'}={w}x{h}" for s, (w, h) in asset_dims.items()
            )
            issues.append({
                "file": record.filename,
                "check": "dimension_consistency",
                "message": f"Mismatched dimensions: {detail}",
            })

        return issues

    # ------------------------------------------
    # Cleanup Intermediates
    # ------------------------------------------

    def _cleanup_intermediates(self):
        """Remove intermediate directory if configured."""
        if not self.config.cleanup_intermediates:
            logger.debug("Skipping intermediate cleanup (disabled in config).")
            return
        inter_dir = self.config.intermediate_dir
        if os.path.isdir(inter_dir):
            shutil.rmtree(inter_dir, ignore_errors=True)
            logger.info(f"Cleaned up intermediate directory: {inter_dir}")

    # ------------------------------------------
    # Run
    # ------------------------------------------

    def run(
        self,
        phases: List[str] = None,
        selected_assets: Optional[List[str]] = None,
        selected_maps: Optional[List[str]] = None,
    ):
        """Run the pipeline for all phases or a selected phase subset.

        Args:
            phases: Optional phase subset to execute.
            selected_assets: Optional relative asset paths to process.
                When provided, scan still traverses ``input_dir`` but only
                matching scanned records are kept for downstream phases.
            selected_maps: Optional output map suffixes to process.
                Examples: ``""`` (base), ``"_albedo"``, ``"_normal"``.
                When provided, non-selected maps are skipped in mipmap
                generation and output copy/compression stages.
        """
        self._cancel_event.clear()
        # Reset per-run state to prevent cross-run accumulation (UI mode).
        self.results = {}
        self._failed_assets = 0
        self._executed_phases = set()
        self._skipped_phases = []
        start_time = time.time()
        requested_set = set(phases) if phases else None
        scan_only_mode = requested_set == {"scan"}
        self._selected_asset_relpaths = (
            None
            if selected_assets is None
            else self._normalized_relpath_set(selected_assets)
        )
        self._selected_map_suffixes = (
            None
            if selected_maps is None
            else self._normalized_map_suffix_set(selected_maps)
        )

        logger.info("=" * 60)
        logger.info(f"GAME ASSET VISUAL UPGRADE PIPELINE v{_get_version()}")
        logger.info("=" * 60)
        logger.info(f"Input:      {self.config.input_dir}")
        logger.info(f"Output:     {self.config.output_dir}")
        logger.info(f"Device:     {self.config.resolve_device()}")
        logger.info(f"Workers:    {self.config.max_workers}")
        logger.info(f"Checkpoint: {'enabled' if self.config.checkpoint.enabled else 'disabled'}")
        if self._selected_asset_relpaths is not None:
            logger.info(
                "Selected assets requested: %d",
                len(self._selected_asset_relpaths),
            )
        if self._selected_map_suffixes is not None:
            selected_map_names = [
                "base" if suffix == "" else suffix
                for suffix in sorted(self._selected_map_suffixes)
            ]
            logger.info(
                "Selected output maps requested: %d (%s)",
                len(self._selected_map_suffixes),
                ", ".join(selected_map_names),
            )

        if self.config.dry_run:
            logger.info("*** DRY RUN MODE ***")

        self.gpu_monitor.log_usage("startup")

        # Create directories only when writing outputs.
        if not self.config.dry_run and not scan_only_mode:
            for d in [self.config.output_dir, self.config.intermediate_dir,
                      self.config.comparison_dir]:
                os.makedirs(d, exist_ok=True)

            # Disk space pre-check: warn if output volume has <1 GB free
            try:
                import shutil as _shutil_disk
                disk = _shutil_disk.disk_usage(self.config.output_dir)
                free_gb = disk.free / (1024 ** 3)
                if free_gb < 1.0:
                    logger.warning(
                        "Low disk space on output volume: %.2f GB free. "
                        "Pipeline may fail during write operations.",
                        free_gb,
                    )
                else:
                    logger.debug("Output disk free: %.1f GB", free_gb)
            except OSError:
                pass  # Non-fatal — skip check on unsupported FS

        # Early warning about missing compression tools when DDS/KTX2 is enabled
        if not self.config.dry_run and self.config.compression.enabled:
            import shutil as _shutil_tools
            import platform as _plat_tools
            from AssetBrew import BIN_DIR as _bin_dir

            def _has_bundled(name: str) -> bool:
                """Check if a bundled binary exists in BIN_DIR."""
                exe = ".exe" if _plat_tools.system() == "Windows" else ""
                if name == "compressonatorcli":
                    for d in sorted(_bin_dir.glob("compressonatorcli*"),
                                    reverse=True):
                        if d.is_dir() and (d / f"{name}{exe}").is_file():
                            return True
                    return (_bin_dir / f"{name}{exe}").is_file()
                return (_bin_dir / f"{name}{exe}").is_file()

            if self.config.compression.generate_dds:
                tool = self.config.compression.tool
                tool_path = self.config.compression.tool_path
                if not tool_path:
                    bin_name = (
                        "compressonatorcli"
                        if tool == "compressonator" else tool
                    )
                    if (not _shutil_tools.which(bin_name)
                            and not _has_bundled(bin_name)):
                        logger.warning(
                            "DDS compression is enabled but '%s' was not "
                            "found on PATH or in bundled tools (%s).",
                            bin_name, _bin_dir,
                        )
            if self.config.compression.generate_ktx2:
                if (not _shutil_tools.which("toktx")
                        and not _has_bundled("toktx")):
                    logger.warning(
                        "KTX2 compression is enabled but 'toktx' was not "
                        "found on PATH or in bundled tools (%s).",
                        _bin_dir,
                    )

        all_phases = {
            "scan": self.phase0_scan,
            "upscale": self.phase1_upscale,
            "pbr": self.phase2_pbr,
            "normal": self.phase3_normals,
            "pom": self.phase4_pom,
            "postprocess": self.phase5_postprocess,
            "mipmap": self.phase6_mipmaps,
            "validate": self.phase7_validate,
        }

        # Warn about missing inter-phase dependencies when a subset is selected.
        if requested_set:
            for phase_name in requested_set:
                deps = _PHASE_DEPS.get(phase_name, [])
                missing = [d for d in deps if d not in requested_set]
                if missing:
                    logger.warning(
                        "Phase '%s' depends on %s, but %s not in selected phases. "
                        "Output quality may be affected.",
                        phase_name,
                        ", ".join(f"'{d}'" for d in missing),
                        "it is" if len(missing) == 1 else "they are",
                    )

        fatal_error = None
        try:
            if phases:
                # Always run scan first, then execute requested phases
                # in canonical order (not user-specified order) to satisfy
                # inter-phase dependencies (e.g. PBR must run before normal).
                requested = [
                    name for name in all_phases
                    if name != "scan" and name in requested_set
                ]
                if requested:
                    logger.info(
                        "Running selected phases: scan, %s",
                        ", ".join(requested),
                    )
                else:
                    logger.info("Running scan phase only (no additional phases selected).")
                self.phase0_scan()
                requested = requested_set
                for name, fn in all_phases.items():
                    if name != "scan" and name in requested:
                        self._executed_phases.add(name)
                        fn()
            else:
                logger.info("Running all phases.")
                for name, fn in all_phases.items():
                    self._executed_phases.add(name)
                    fn()
        except Exception as exc:
            fatal_error = exc
            logger.exception("Pipeline execution aborted: %s", exc)
        finally:
            if not self.config.dry_run:
                if scan_only_mode:
                    logger.info(
                        "Scan-only selection: skipped copy_to_output(), "
                        "output validation, and results write."
                    )
                else:
                    if fatal_error is not None:
                        logger.error(
                            "Skipping output copy/validation after fatal error: %s",
                            type(fatal_error).__name__,
                        )
                        try:
                            self._save_results()
                        except Exception as exc:
                            logger.exception("Failed during _save_results(): %s", exc)
                    else:
                        try:
                            self.copy_to_output()
                        except Exception as exc:
                            logger.exception("Failed during copy_to_output(): %s", exc)
                            if fatal_error is None:
                                fatal_error = exc
                        try:
                            self._validate_outputs()
                        except Exception as exc:
                            logger.exception("Failed during _validate_outputs(): %s", exc)
                            if fatal_error is None:
                                fatal_error = exc
                        try:
                            self._save_results()
                        except Exception as exc:
                            logger.exception("Failed during _save_results(): %s", exc)
                            if fatal_error is None:
                                fatal_error = exc
            else:
                logger.info(
                    "Dry-run mode: skipped copy_to_output(), "
                    "output validation, and results write."
                )

            try:
                self.checkpoint.save()  # Final checkpoint save on all exits
            except Exception as exc:
                logger.exception("Failed to save checkpoint at shutdown: %s", exc)
                if fatal_error is None:
                    fatal_error = exc

            if fatal_error is None:
                self._cleanup_intermediates()

            # Release all phase processors to free memory (ONNX sessions,
            # cached models, internal buffers).  The upscaler and normal
            # generator have their own try/finally cleanup in their phase
            # methods; this sweep catches the rest and acts as a safety net.
            for attr in (
                "_pbr_gen", "_pom_proc", "_mipmap_gen", "_orm_packer",
                "_color_consistency", "_color_grading", "_seam_repair",
                "_emissive_gen", "_reflection_mask_gen", "_specular_aa",
                "_detail_overlay", "_validator",
                # Safety net for upscaler/normal_gen in case phase-level
                # cleanup was bypassed (e.g. exception before the phase ran).
                "_upscaler", "_normal_gen",
            ):
                proc = getattr(self, attr, None)
                if proc is not None:
                    cleanup_fn = getattr(proc, "cleanup", None)
                    if callable(cleanup_fn):
                        try:
                            cleanup_fn()
                        except Exception:
                            pass
                    setattr(self, attr, None)

        # Count assets with at least one phase error or output validation issue
        self._failed_assets = 0
        for fp, phases_dict in self.results.items():
            has_error = False
            for phase_name, phase_result in phases_dict.items():
                if self._is_failed_phase_result(phase_name, phase_result):
                    has_error = True
                    break
            if has_error:
                self._failed_assets += 1

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        warnings = []
        if not self.records:
            warnings.append(
                "No assets found -- check input directory and "
                "filename patterns"
            )
        if self._skipped_phases:
            warnings.append(
                f"{len(self._skipped_phases)} phase(s) skipped: "
                f"{', '.join(self._skipped_phases)}"
            )
        if self._failed_assets:
            warnings.append(
                f"{self._failed_assets}/{len(self.records)} asset(s) "
                f"had errors"
            )
        if fatal_error is not None:
            warnings.append(f"fatal error: {fatal_error}")
        if warnings:
            logger.warning(
                f"PIPELINE FINISHED WITH WARNINGS in {elapsed:.1f}s -- "
                + "; ".join(warnings)
            )
        else:
            logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"Processed {len(self.records)} assets")
        # Count heuristic PBR outputs
        heuristic_pbr_count = sum(
            1 for r in self.results.values()
            if isinstance(r.get("pbr"), dict) and r["pbr"].get("is_heuristic")
        )
        if heuristic_pbr_count:
            logger.info(
                f"Heuristic PBR maps: {heuristic_pbr_count} asset(s) "
                f"received heuristic roughness/metalness/AO (not physically "
                f"measured). Review before production use."
            )
        logger.info(f"Output: {self.config.output_dir}")
        logger.info("=" * 60)

        if fatal_error is not None:
            raise fatal_error

    def _save_results(self):
        path = os.path.join(self.config.output_dir, "pipeline_results.json")

        def _make_serializable(obj):
            """Recursively convert non-JSON-safe values to strings."""
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            return str(obj)

        with self._results_lock:
            serializable = _make_serializable(self.results)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        logger.info(f"Results saved: {path}")
