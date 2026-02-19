"""Logging setup for the asset pipeline."""

import logging
import logging.handlers
import os
import threading

logger = logging.getLogger("asset_pipeline")

# 10 MB max per log file, keep 3 rotated backups
_LOG_MAX_BYTES = 10 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_setup_lock = threading.Lock()


def setup_logging(level: str = "INFO", log_file: str = None, force: bool = False):
    """Configure logging without clobbering host-app handlers by default."""
    with _setup_lock:
        _setup_logging_impl(level, log_file, force)


def _setup_logging_impl(level: str, log_file: str, force: bool):
    """Internal implementation of setup_logging (called under _setup_lock)."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s [T%(thread)d]: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_file, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        ))
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{level}', defaulting to INFO")
        numeric_level = logging.INFO

    root = logging.getLogger()
    if force:
        logger.debug("Setting logging with force=%s and handlers=%d", force, len(handlers))
        logging.basicConfig(
            level=numeric_level,
            format=fmt,
            handlers=handlers,
            force=True,
        )
        return

    if not root.handlers:
        logger.debug("No root handlers found; initializing logging with default handlers.")
        logging.basicConfig(
            level=numeric_level,
            format=fmt,
            handlers=handlers,
        )
        return

    # Embedded mode: only update the asset_pipeline logger hierarchy so we
    # don't affect unrelated libraries that share the root logger.
    pipeline_logger = logging.getLogger("asset_pipeline")
    pipeline_logger.setLevel(numeric_level)
    if log_file:
        existing_files = {
            getattr(h, "baseFilename", None)
            for h in pipeline_logger.handlers
            if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler))
        }
        file_handler = handlers[1]
        if getattr(file_handler, "baseFilename", None) not in existing_files:
            logger.info("Adding file handler: %s", file_handler.baseFilename)
            file_handler.setFormatter(logging.Formatter(fmt))
            pipeline_logger.addHandler(file_handler)
