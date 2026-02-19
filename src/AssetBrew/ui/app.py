"""PyQt desktop UI for the asset preprocess pipeline.

Features:
- Runtime config editing (full dataclass tree, live validation)
- Pipeline run controls (phases, device, workers, dry-run, checkpoint reset)
- Asset scanning + selection
- Interactive before/after preview (split slider, zoom, pan, per-map toggles)
- Live pipeline log streaming from a background worker thread
"""

from __future__ import annotations

import copy
import logging
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PyQt6.QtCore import (
        QObject,
        QPointF,
        QRegularExpression,
        QRectF,
        QSettings,
        QTimer,
        Qt,
        QThread,
        pyqtSignal,
        pyqtSlot,
    )
    from PyQt6.QtGui import (
        QColor,
        QImage,
        QKeySequence,
        QMouseEvent,
        QPainter,
        QPen,
        QRegularExpressionValidator,
        QShortcut,
        QWheelEvent,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListView,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PyQt6 is required for the UI.\nInstall with: pip install PyQt6"
    ) from exc

from ..config import PipelineConfig
from ..pipeline import AssetPipeline, PipelineCancelledError
from ..core import CheckpointManager, load_image, scan_assets, setup_logging
from .core import (
    MAP_OPTIONS,
    OUTPUT_EXT_PRIORITY,
    find_output_map_file,
    parse_typed_value,
    resolve_map_paths,
    set_dataclass_path,
    value_to_text,
)

PHASE_OPTIONS: List[str] = [
    "upscale",
    "pbr",
    "normal",
    "pom",
    "mipmap",
    "postprocess",
    "validate",
]
PHASE_EXECUTION_ORDER: List[str] = [
    "upscale",
    "pbr",
    "normal",
    "pom",
    "postprocess",
    "mipmap",
    "validate",
]

PREVIEW_FAST_PHASES = {"upscale", "pbr", "normal", "pom", "postprocess"}
CORE_MAP_LABELS = {"Base", "Albedo", "Normal", "Roughness", "ORM"}
LOG_LEVEL_FILTERS: Tuple[str, ...] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "OTHER",
)



def _image_to_qimage(path: str, max_pixels: int = 67_108_864) -> QImage:
    arr = load_image(path, max_pixels=max_pixels)
    arr = np.clip(arr, 0.0, 1.0)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.shape[2] >= 4:
        rgba = np.ascontiguousarray((arr[:, :, :4] * 255.0).astype(np.uint8))
        image = QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        return image.copy()

    rgb = np.ascontiguousarray((arr[:, :, :3] * 255.0).astype(np.uint8))
    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    )
    return image.copy()


class ConfigTreeEditor(QWidget):
    """Render editable `PipelineConfig` values in a hierarchical tree."""

    ROLE_PATH = int(Qt.ItemDataRole.UserRole)
    ROLE_TEMPLATE = ROLE_PATH + 1

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the config tree editor widget."""
        super().__init__(parent)
        self._config_snapshot = PipelineConfig()
        self._leaf_items: Dict[Tuple[str, ...], QTreeWidgetItem] = {}

        layout = QVBoxLayout(self)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(
            "Filter settings (example: color_grading.lut or emissive.threshold)"
        )
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)

    def set_config(self, config: PipelineConfig) -> None:
        """Populate the tree with values from a pipeline configuration."""
        self._config_snapshot = copy.deepcopy(config)
        self._leaf_items.clear()
        self.tree.clear()
        self._populate(self._config_snapshot, parent_item=None)
        self.tree.expandToDepth(1)
        self.tree.resizeColumnToContents(0)
        self._apply_filter(self.filter_edit.text())

    def build_config(self) -> PipelineConfig:
        """Build a `PipelineConfig` from edited tree values."""
        config = copy.deepcopy(self._config_snapshot)
        errors: List[str] = []

        for path, item in self._leaf_items.items():
            template = item.data(0, self.ROLE_TEMPLATE)
            value_text = item.text(1)
            try:
                parsed = parse_typed_value(value_text, template)
                set_dataclass_path(config, path, parsed)
            except Exception as exc:
                joined = ".".join(path)
                errors.append(f"{joined}: {exc}")

        if errors:
            raise ValueError("Invalid config edits:\n- " + "\n- ".join(errors))

        return config

    def _populate(
        self,
        obj: Any,
        parent_item: Optional[QTreeWidgetItem],
        prefix: Tuple[str, ...] = (),
    ) -> None:
        if not is_dataclass(obj):
            return

        for f in fields(obj):
            value = getattr(obj, f.name)
            path = prefix + (f.name,)

            if is_dataclass(value):
                group = QTreeWidgetItem([f.name, ""])
                if parent_item is None:
                    self.tree.addTopLevelItem(group)
                else:
                    parent_item.addChild(group)
                self._populate(value, group, path)
                continue

            leaf = QTreeWidgetItem([f.name, value_to_text(value)])
            leaf.setData(0, self.ROLE_PATH, path)
            leaf.setData(0, self.ROLE_TEMPLATE, copy.deepcopy(value))
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsEditable)

            if parent_item is None:
                self.tree.addTopLevelItem(leaf)
            else:
                parent_item.addChild(leaf)

            self._leaf_items[path] = leaf

    def _apply_filter(self, raw_text: str) -> None:
        """Filter visible config rows by key path or value text."""
        needle = (raw_text or "").strip().lower()

        def visit(item: QTreeWidgetItem, prefix: Tuple[str, ...] = ()) -> bool:
            key = item.text(0)
            path = prefix + (key,)
            own_text = f"{'.'.join(path)} {item.text(1)}".lower()
            own_match = (not needle) or (needle in own_text)

            if item.childCount() == 0:
                item.setHidden(not own_match)
                return own_match

            child_visible = False
            for idx in range(item.childCount()):
                child = item.child(idx)
                child_visible = visit(child, path) or child_visible

            visible = own_match or child_visible
            item.setHidden(not visible)
            if needle and child_visible:
                item.setExpanded(True)
            return visible

        for idx in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(idx))


class SplitImageViewer(QWidget):
    """Display before/after images with split, zoom, and pan controls."""

    zoom_changed = pyqtSignal(float)
    split_ratio_changed = pyqtSignal(float)
    VALID_PREVIEW_MODES = {"split", "before", "after"}

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize interactive preview viewer state."""
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)

        self._before: Optional[QImage] = None
        self._after: Optional[QImage] = None
        self._before_label = ""
        self._after_label = ""
        self._split_ratio = 0.5
        self._split_swapped = False
        self._preview_mode = "split"
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._dragging_pan = False
        self._dragging_split = False
        self._split_hit_radius = 8.0
        self._last_pos = QPointF(0.0, 0.0)

    def set_images(
        self,
        before: Optional[QImage],
        after: Optional[QImage],
        before_label: str = "",
        after_label: str = "",
    ) -> None:
        """Set preview images and optional labels, then schedule repaint."""
        self._before = before
        self._after = after
        self._before_label = before_label
        self._after_label = after_label
        self.update()

    def set_split_ratio(self, ratio: float) -> None:
        """Set split ratio in `[0, 1]` and emit change signal when updated."""
        clamped = max(0.0, min(1.0, ratio))
        if abs(clamped - self._split_ratio) < 1e-6:
            return
        self._split_ratio = clamped
        self.split_ratio_changed.emit(self._split_ratio)
        self.update()

    def set_split_swapped(self, swapped: bool) -> None:
        """Swap split-mode left/right image assignment when enabled."""
        normalized = bool(swapped)
        if normalized == self._split_swapped:
            return
        self._split_swapped = normalized
        self.update()

    def set_preview_mode(self, mode: str) -> None:
        """Switch preview mode among `split`, `before`, and `after`."""
        normalized = mode.strip().lower()
        if normalized not in self.VALID_PREVIEW_MODES:
            normalized = "split"
        if normalized == self._preview_mode:
            return
        self._preview_mode = normalized
        self._dragging_split = False
        if self._preview_mode != "split" and not self._dragging_pan:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_zoom(self, zoom: float) -> None:
        """Set zoom factor and emit zoom-change signal when updated."""
        clamped = max(0.1, min(16.0, zoom))
        if abs(clamped - self._zoom) < 1e-6:
            return
        self._zoom = clamped
        self.zoom_changed.emit(self._zoom)
        self.update()

    def reset_view(self) -> None:
        """Reset zoom and pan to their default values."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse-wheel zoom interactions."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.set_zoom(self._zoom * factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start split-handle drag or viewport panning on left click."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_split_interaction_enabled() and self._is_split_handle_hit(event.position()):
                self._dragging_split = True
                self._last_pos = event.position()
                self.setCursor(Qt.CursorShape.SplitHCursor)
                event.accept()
                return
            self._dragging_pan = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update split position or pan offset while dragging."""
        if self._dragging_split:
            self._update_split_from_pos(event.position())
            event.accept()
            return

        if self._dragging_pan:
            delta = event.position() - self._last_pos
            self._pan += delta
            self._last_pos = event.position()
            self.update()
            event.accept()
            return

        if self._is_split_interaction_enabled() and self._is_split_handle_hit(event.position()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End active dragging interactions on mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_pan = False
            self._dragging_split = False
            if self._is_split_interaction_enabled() and self._is_split_handle_hit(event.position()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """Render current preview state, split line, and overlay metadata."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#181818"))

        if self._preview_mode == "before":
            draw_base = self._before or self._after
        elif self._preview_mode == "after":
            draw_base = self._after or self._before
        else:
            draw_base = self._after or self._before

        if draw_base is None:
            painter.setPen(QColor("#A0A0A0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview loaded")
            return

        target = self._target_rect(draw_base.width(), draw_base.height())

        if self._preview_mode == "before":
            if self._before is not None:
                painter.drawImage(target, self._before)
            elif self._after is not None:
                painter.drawImage(target, self._after)
        elif self._preview_mode == "after":
            if self._after is not None:
                painter.drawImage(target, self._after)
            elif self._before is not None:
                painter.drawImage(target, self._before)
        else:
            right_image = self._before if self._split_swapped else self._after
            left_image = self._after if self._split_swapped else self._before

            if right_image is not None:
                painter.drawImage(target, right_image)

            if left_image is not None:
                if right_image is None:
                    painter.drawImage(target, left_image)
                else:
                    split_x = target.left() + target.width() * self._split_ratio
                    clip = QRectF(
                        target.left(),
                        target.top(),
                        max(0.0, split_x - target.left()),
                        target.height(),
                    )
                    painter.save()
                    painter.setClipRect(clip)
                    painter.drawImage(target, left_image)
                    painter.restore()

                    painter.setPen(QPen(QColor("#FFC857"), 2))
                    painter.drawLine(
                        int(split_x),
                        int(target.top()),
                        int(split_x),
                        int(target.bottom()),
                    )

        self._draw_overlay(painter)

    def _target_rect(self, img_w: int, img_h: int) -> QRectF:
        view_w = max(1, self.width())
        view_h = max(1, self.height())
        fit = min(view_w / img_w, view_h / img_h)
        scale = fit * self._zoom
        draw_w = img_w * scale
        draw_h = img_h * scale
        left = (view_w - draw_w) * 0.5 + self._pan.x()
        top = (view_h - draw_h) * 0.5 + self._pan.y()
        return QRectF(left, top, draw_w, draw_h)

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(QColor("#F0F0F0"))
        mode = self._preview_mode.capitalize()
        swap_state = (
            " | Sides: swapped"
            if self._preview_mode == "split" and self._split_swapped
            else ""
        )
        info = (
            f"Mode: {mode} | Zoom: {self._zoom * 100:.0f}% "
            f"| Split: {self._split_ratio * 100:.0f}%"
            f"{swap_state}"
        )
        painter.drawText(10, 20, info)

        left_text = f"Before: {self._before_label or '-'}"
        right_text = f"After: {self._after_label or '-'}"
        if self._preview_mode == "split" and self._split_swapped:
            left_text, right_text = right_text, left_text

        metrics = painter.fontMetrics()
        line_h = metrics.height()
        baseline_y = self.height() - 12
        left_x = 16
        right_w = metrics.horizontalAdvance(right_text)
        right_x = max(left_x + 24, self.width() - right_w - 16)

        left_w = metrics.horizontalAdvance(left_text)
        pad_x = 8
        pad_y = 4
        left_bg = QRectF(
            left_x - pad_x,
            baseline_y - line_h - pad_y,
            left_w + (pad_x * 2),
            line_h + (pad_y * 2),
        )
        right_bg = QRectF(
            right_x - pad_x,
            baseline_y - line_h - pad_y,
            right_w + (pad_x * 2),
            line_h + (pad_y * 2),
        )
        painter.fillRect(left_bg, QColor(0, 0, 0, 120))
        painter.fillRect(right_bg, QColor(0, 0, 0, 120))

        painter.drawText(left_x, baseline_y, left_text)
        painter.drawText(right_x, baseline_y, right_text)

    def _is_split_interaction_enabled(self) -> bool:
        return (
            self._preview_mode == "split"
            and self._before is not None
            and self._after is not None
        )

    def _current_target_rect(self) -> Optional[QRectF]:
        draw_base = self._after or self._before
        if draw_base is None:
            return None
        return self._target_rect(draw_base.width(), draw_base.height())

    def _is_split_handle_hit(self, pos: QPointF) -> bool:
        if not self._is_split_interaction_enabled():
            return False
        target = self._current_target_rect()
        if target is None or not target.contains(pos):
            return False
        split_x = target.left() + target.width() * self._split_ratio
        return abs(pos.x() - split_x) <= self._split_hit_radius

    def _update_split_from_pos(self, pos: QPointF) -> None:
        target = self._current_target_rect()
        if target is None or target.width() <= 0:
            return
        ratio = (pos.x() - target.left()) / target.width()
        self.set_split_ratio(ratio)


class QtSignalLogHandler(logging.Handler):
    """Forward Python log records to a Qt signal callback."""

    def __init__(self, callback):
        """Initialize handler with a callable receiving formatted lines."""
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """Emit one formatted log record to the registered callback."""
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            pass


class ScanWorker(QObject):
    """Scan assets on a background thread to keep UI responsive."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: PipelineConfig):
        """Capture scan config for asynchronous execution."""
        super().__init__()
        self.config = copy.deepcopy(config)

    @pyqtSlot()
    def run(self) -> None:
        """Execute scan and emit record dicts."""
        try:
            records = scan_assets(self.config.input_dir, self.config)
            self.finished.emit([r.to_dict() for r in records])
        except Exception as exc:
            self.failed.emit(str(exc))


class PreviewImageLoadWorker(QObject):
    """Load preview QImages in a background thread."""

    finished = pyqtSignal(object)

    def __init__(self, token: int, before_path: str, after_path: str):
        """Capture image load request data."""
        super().__init__()
        self.token = int(token)
        self.before_path = before_path or ""
        self.after_path = after_path or ""

    @pyqtSlot()
    def run(self) -> None:
        """Load image paths into QImage payload for main-thread consumption."""
        payload: Dict[str, Any] = {
            "token": self.token,
            "before_path": self.before_path,
            "after_path": self.after_path,
            "before_img": None,
            "after_img": None,
            "errors": [],
        }
        for key, path in (("before_img", self.before_path), ("after_img", self.after_path)):
            if not path or not os.path.exists(path):
                continue
            try:
                payload[key] = _image_to_qimage(path)
            except Exception as exc:
                payload["errors"].append((path, str(exc)))
        self.finished.emit(payload)


class PipelineWorker(QObject):
    """Run the full pipeline in a background Qt worker thread."""

    log_line = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object, object, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: PipelineConfig,
        phases: Optional[List[str]],
        reset_checkpoint: bool,
        selected_relpaths: Optional[List[str]] = None,
        selected_map_suffixes: Optional[List[str]] = None,
    ):
        """Capture run parameters for asynchronous pipeline execution."""
        super().__init__()
        self.config = copy.deepcopy(config)
        self.phases = phases
        self.reset_checkpoint = reset_checkpoint
        self.selected_relpaths = list(selected_relpaths) if selected_relpaths else None
        self.selected_map_suffixes = (
            list(selected_map_suffixes) if selected_map_suffixes else None
        )
        self._pipeline: Optional[AssetPipeline] = None
        self._cancel_requested = False

    @pyqtSlot()
    def request_cancel(self) -> None:
        """Request cooperative pipeline cancellation."""
        self._cancel_requested = True
        if self._pipeline is not None:
            self._pipeline.request_cancel()

    @pyqtSlot()
    def run(self) -> None:
        """Execute pipeline run and emit completion or failure signals."""
        root = logging.getLogger()
        log_handler = QtSignalLogHandler(self.log_line.emit)
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        try:
            log_file = os.path.join(self.config.output_dir, "pipeline.log")
            setup_logging(self.config.log_level, log_file)
            root = logging.getLogger()
            root.addHandler(log_handler)

            if self.reset_checkpoint:
                CheckpointManager(self.config).clear()

            pipeline = AssetPipeline(self.config, progress_callback=self.progress.emit)
            self._pipeline = pipeline
            if self._cancel_requested:
                pipeline.request_cancel()
            pipeline.run(
                self.phases,
                selected_assets=self.selected_relpaths,
                selected_maps=self.selected_map_suffixes,
            )
            records = [r.to_dict() for r in pipeline.records]
            failed_count = int(getattr(pipeline, "_failed_assets", 0))
            self.finished.emit(pipeline.results, records, failed_count)
        except PipelineCancelledError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            detail = f"{exc}\n\n{traceback.format_exc()}"
            self.failed.emit(detail)
        finally:
            self._pipeline = None
            try:
                root.removeHandler(log_handler)
            except Exception:
                pass


class PreviewRenderWorker(QObject):
    """Render a staged single-asset preview in a background worker thread."""

    log_line = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: PipelineConfig,
        phases: Optional[List[str]],
        rel_path: str,
    ):
        """Capture preview inputs for asynchronous rendering."""
        super().__init__()
        self.config = copy.deepcopy(config)
        self.phases = list(phases) if phases else None
        self.rel_path = rel_path
        self._pipeline: Optional[AssetPipeline] = None
        self._cancel_requested = False

    @pyqtSlot()
    def request_cancel(self) -> None:
        """Request cooperative cancellation for the preview render."""
        self._cancel_requested = True
        if self._pipeline is not None:
            self._pipeline.request_cancel()

    @pyqtSlot()
    def run(self) -> None:
        """Run preview pipeline on a temporary staged copy of one asset."""
        root = logging.getLogger()
        log_handler = QtSignalLogHandler(self.log_line.emit)
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

        staging_root = ""
        try:
            source_path = os.path.join(self.config.input_dir, self.rel_path)
            if not os.path.isfile(source_path):
                raise FileNotFoundError(f"Selected asset not found: {source_path}")

            staging_root = tempfile.mkdtemp(prefix="dds_preview_")
            staging_input = os.path.join(staging_root, "input")

            staged_asset_path = os.path.join(staging_input, self.rel_path)
            os.makedirs(os.path.dirname(staged_asset_path), exist_ok=True)
            shutil.copy2(source_path, staged_asset_path)

            preview_cfg = copy.deepcopy(self.config)
            preview_cfg.input_dir = staging_input
            preview_cfg.dry_run = False
            preview_cfg.max_workers = max(1, min(preview_cfg.max_workers, 2))
            preview_cfg.checkpoint.enabled = False
            preview_cfg.intermediate_dir = os.path.join(staging_root, "intermediate")
            preview_cfg.comparison_dir = os.path.join(staging_root, "comparisons")
            preview_cfg.manifest_path = os.path.join(staging_root, "manifest.csv")

            selected = set(self.phases or [])
            # Preview runs often skip expensive phases like mipmap/validate.
            # Align validation strictness with selected phases to avoid
            # false-failure summaries for intentionally partial renders.
            if self.phases is not None and "validate" not in selected:
                preview_cfg.validation.enabled = False
            if self.phases is not None and "mipmap" not in selected:
                preview_cfg.validation.require_full_mipchain = False

            preview_log = os.path.join(staging_root, "pipeline_preview.log")
            setup_logging(preview_cfg.log_level, preview_log)
            root = logging.getLogger()
            root.addHandler(log_handler)

            pipeline = AssetPipeline(preview_cfg)
            self._pipeline = pipeline
            if self._cancel_requested:
                pipeline.request_cancel()
            pipeline.run(self.phases)

            failed_count = int(getattr(pipeline, "_failed_assets", 0))
            self.finished.emit(
                {
                    "failed_count": failed_count,
                    "results": pipeline.results,
                    "records": [r.to_dict() for r in pipeline.records],
                }
            )
        except PipelineCancelledError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            detail = f"{exc}\n\n{traceback.format_exc()}"
            self.failed.emit(detail)
        finally:
            self._pipeline = None
            try:
                root.removeHandler(log_handler)
            except Exception:
                pass
            if staging_root:
                shutil.rmtree(staging_root, ignore_errors=True)


class MainWindow(QMainWindow):
    """Provide the main desktop UI for configuring and running the pipeline."""

    SETTINGS_ORG = "AssetBrew"
    SETTINGS_APP = "AssetBrew_ui"

    _cancel_worker = pyqtSignal()
    _cancel_preview = pyqtSignal()

    def __init__(self):
        """Initialize main window state, UI layout, and persisted settings."""
        super().__init__()
        self.setWindowTitle("AssetBrew")
        self.resize(1560, 920)

        self.config = PipelineConfig()
        self.records: List[Dict[str, Any]] = []
        self.accepted_records: List[Dict[str, Any]] = []
        self.all_input_records: List[Dict[str, Any]] = []
        self.latest_results: Dict[str, Any] = {}
        self._image_cache: "OrderedDict[str, Tuple[QImage, int]]" = OrderedDict()
        self._image_cache_bytes = 0
        self._image_cache_limit_bytes = 512 * 1024 * 1024
        self._current_map_paths: Dict[str, str] = {}
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None
        self._scan_thread: Optional[QThread] = None
        self._scan_worker: Optional[ScanWorker] = None
        self._scan_input_dir = ""
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[PreviewRenderWorker] = None
        self._image_loader_thread: Optional[QThread] = None
        self._image_loader_worker: Optional[PreviewImageLoadWorker] = None
        self._pending_image_request: Optional[Tuple[int, str, str, str, str]] = None
        self._active_image_request_token = 0
        self._latest_image_request_token = 0
        self._active_preview_labels: Tuple[str, str] = ("", "")
        self._pending_preview_request = False
        self._preview_request_reason = ""
        self._preview_fail_dialog = False
        self._progress_phase = ""
        self._progress_done = 0
        self._progress_total = 0
        self._run_phase_sequence: List[str] = []
        self._quality_field_getters: Dict[Tuple[str, ...], Callable[[], Any]] = {}
        self._quality_field_setters: Dict[Tuple[str, ...], Callable[[Any], None]] = {}
        self._log_entries: List[Tuple[str, str]] = []
        self._max_log_entries = 20000
        self._active_log_levels = set(LOG_LEVEL_FILTERS)
        self._log_filter_buttons: Dict[str, QPushButton] = {}
        self._checked_asset_relpaths: set[str] = set()
        self._auto_check_all_after_scan = False
        self._asset_table_populating = False
        self._settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.setInterval(450)
        self._auto_preview_timer.timeout.connect(self._trigger_auto_preview)

        self._config_undo_stack: List[PipelineConfig] = []
        self._config_redo_stack: List[PipelineConfig] = []
        self._CONFIG_UNDO_LIMIT = 50

        self._build_ui()
        self._load_ui_settings()
        self._load_default_config()
        self.statusBar().showMessage("Ready")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.main_splitter)

        left_tabs = QTabWidget()
        self.main_splitter.addWidget(left_tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        left_tabs.addTab(run_tab, "Pipeline")

        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        left_tabs.addTab(config_tab, "Runtime Config")

        quality_tab = QWidget()
        quality_layout = QVBoxLayout(quality_tab)
        left_tabs.addTab(quality_tab, "Quality Controls")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.main_splitter.addWidget(right_panel)

        self._build_run_controls(run_layout)
        self._build_asset_table(run_layout)
        self._build_quality_controls(quality_layout)

        self.config_editor = ConfigTreeEditor()
        config_layout.addWidget(self.config_editor)
        hint = QLabel("Tip: Use filter above to quickly find config keys.")
        hint.setStyleSheet("color: #707070;")
        config_layout.addWidget(hint)
        self.config_editor.tree.itemChanged.connect(self._on_config_tree_item_changed)

        apply_row = QHBoxLayout()
        self.apply_config_btn = QPushButton("Apply Runtime Config")
        self.apply_config_btn.clicked.connect(self._on_apply_runtime_config)
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._on_save_config)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo last config change (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_config)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setEnabled(False)
        self.redo_btn.setToolTip("Redo config change (Ctrl+Y)")
        self.redo_btn.clicked.connect(self._redo_config)
        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo_config)
        QShortcut(QKeySequence.StandardKey.Redo, self, self._redo_config)
        apply_row.addWidget(self.apply_config_btn)
        apply_row.addWidget(self.save_config_btn)
        apply_row.addWidget(self.undo_btn)
        apply_row.addWidget(self.redo_btn)
        apply_row.addStretch(1)
        config_layout.addLayout(apply_row)

        self._build_preview_panel(right_layout)
        self._build_log_panel(right_layout)

    def _build_run_controls(self, parent_layout: QVBoxLayout) -> None:
        path_box = QGroupBox("Paths")
        path_layout = QFormLayout(path_box)

        self.config_path_edit = QLineEdit("config.yaml")
        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()

        config_row = QHBoxLayout()
        config_row.addWidget(self.config_path_edit)
        config_browse = QPushButton("...")
        config_browse.setFixedWidth(34)
        config_browse.clicked.connect(self._browse_config_path)
        load_cfg = QPushButton("Load")
        load_cfg.clicked.connect(self._on_load_config)
        config_row.addWidget(config_browse)
        config_row.addWidget(load_cfg)
        path_layout.addRow("Config file", self._wrap(config_row))

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_dir_edit)
        input_browse = QPushButton("...")
        input_browse.setFixedWidth(34)
        input_browse.clicked.connect(lambda: self._browse_dir(self.input_dir_edit))
        input_row.addWidget(input_browse)
        path_layout.addRow("Input dir", self._wrap(input_row))

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_browse = QPushButton("...")
        output_browse.setFixedWidth(34)
        output_browse.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        output_row.addWidget(output_browse)
        path_layout.addRow("Output dir", self._wrap(output_row))

        parent_layout.addWidget(path_box)

        run_box = QGroupBox("Runtime")
        run_layout = QHBoxLayout(run_box)

        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["auto", "cuda", "cuda:0", "cpu"])
        device_regex = QRegularExpression(r"^(auto|cpu|cuda(?::\d+)?)$")
        device_validator = QRegularExpressionValidator(device_regex, self.device_combo)
        self.device_combo.lineEdit().setValidator(device_validator)
        self.device_combo.currentTextChanged.connect(
            lambda _text: self._schedule_auto_preview("device")
        )
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 128)
        self.workers_spin.setValue(4)
        self.workers_spin.valueChanged.connect(
            lambda _v: self._schedule_auto_preview("workers")
        )
        self.dry_run_check = QCheckBox("Dry run")
        self.dry_run_check.toggled.connect(
            lambda _on: self._schedule_auto_preview("dry_run")
        )
        self.reset_checkpoint_check = QCheckBox("Reset checkpoint")

        run_layout.addWidget(QLabel("Device"))
        run_layout.addWidget(self.device_combo)
        run_layout.addWidget(QLabel("Workers"))
        run_layout.addWidget(self.workers_spin)
        run_layout.addWidget(self.dry_run_check)
        run_layout.addWidget(self.reset_checkpoint_check)
        run_layout.addStretch(1)
        parent_layout.addWidget(run_box)

        phase_box = QGroupBox("Phases")
        phase_layout = QVBoxLayout(phase_box)
        phase_checks_row = QHBoxLayout()
        self.phase_checks: Dict[str, QCheckBox] = {}
        for phase in PHASE_OPTIONS:
            cb = QCheckBox(phase)
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda _state, name=phase: self._schedule_auto_preview(f"phase:{name}")
            )
            self.phase_checks[phase] = cb
            phase_checks_row.addWidget(cb)
        phase_checks_row.addStretch(1)
        phase_layout.addLayout(phase_checks_row)

        phase_actions = QHBoxLayout()
        phase_all_btn = QPushButton("All")
        phase_all_btn.clicked.connect(lambda: self._set_phase_selection("all"))
        phase_none_btn = QPushButton("None")
        phase_none_btn.clicked.connect(lambda: self._set_phase_selection("none"))
        phase_fast_btn = QPushButton("Preview Fast")
        phase_fast_btn.clicked.connect(lambda: self._set_phase_selection("preview_fast"))
        phase_actions.addWidget(phase_all_btn)
        phase_actions.addWidget(phase_none_btn)
        phase_actions.addWidget(phase_fast_btn)
        phase_actions.addStretch(1)
        phase_layout.addLayout(phase_actions)
        parent_layout.addWidget(phase_box)

        asset_list_box = QGroupBox("Asset List")
        asset_list_layout = QHBoxLayout(asset_list_box)
        asset_list_layout.addWidget(QLabel("Source"))
        self.asset_source_combo = QComboBox()
        self.asset_source_combo.addItem("Accepted items only", "accepted")
        self.asset_source_combo.addItem("All files from input", "all_input")
        self.asset_source_combo.currentIndexChanged.connect(self._on_asset_source_changed)
        asset_list_layout.addWidget(self.asset_source_combo)
        self.scan_btn = QPushButton("Scan Assets")
        self.scan_btn.clicked.connect(self._on_scan_assets)
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self._on_run_pipeline)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_pipeline)
        self.select_all_assets_btn = QPushButton("Select All")
        self.select_all_assets_btn.setToolTip("Check all listed assets")
        self.select_all_assets_btn.clicked.connect(
            lambda: self._set_all_listed_asset_checks(True)
        )
        self.unselect_all_assets_btn = QPushButton("Unselect All")
        self.unselect_all_assets_btn.setToolTip("Uncheck all listed assets")
        self.unselect_all_assets_btn.clicked.connect(
            lambda: self._set_all_listed_asset_checks(False)
        )
        asset_list_layout.addWidget(self.scan_btn)
        asset_list_layout.addWidget(self.run_btn)
        asset_list_layout.addWidget(self.stop_btn)
        asset_list_layout.addWidget(self.select_all_assets_btn)
        asset_list_layout.addWidget(self.unselect_all_assets_btn)
        asset_list_layout.addStretch(1)
        parent_layout.addWidget(asset_list_box)

        self.stop_hint = QLabel(
            "Stop requests cooperative cancellation. "
            "Shortcut: Ctrl+Shift+A toggles all listed checks."
        )
        self.stop_hint.setStyleSheet("color: #808080;")

        self.config_health_label = QLabel("Config status: unknown")
        self.config_health_label.setStyleSheet("color: #8A8A8A;")
        status_row = QHBoxLayout()
        status_row.addWidget(self.stop_hint)
        status_row.addStretch(1)
        status_row.addWidget(self.config_health_label)
        parent_layout.addLayout(status_row)

        self.input_dir_edit.editingFinished.connect(
            lambda: self._schedule_auto_preview("input_dir")
        )
        self.output_dir_edit.editingFinished.connect(
            lambda: self._schedule_auto_preview("output_dir")
        )
        self.output_dir_edit.editingFinished.connect(self._refresh_asset_table_statuses)

    @staticmethod
    def _get_dataclass_path_value(config: PipelineConfig, path: Sequence[str]) -> Any:
        cur: Any = config
        for key in path:
            cur = getattr(cur, key)
        return cur

    def _register_quality_binding(
        self,
        path: Sequence[str],
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
    ) -> None:
        key = tuple(path)
        self._quality_field_getters[key] = getter
        self._quality_field_setters[key] = setter

    def _on_quality_control_changed(self, reason: str) -> None:
        self._schedule_auto_preview(f"quality:{reason}")

    def _build_quality_controls(self, parent_layout: QVBoxLayout) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        content_layout = QVBoxLayout(container)

        def _set_checked(widget: QCheckBox, value: Any) -> None:
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)

        def _set_int(widget: QSpinBox, value: Any) -> None:
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

        def _set_float(widget: QDoubleSpinBox, value: Any) -> None:
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)

        def _set_text(widget: QLineEdit, value: Any) -> None:
            widget.blockSignals(True)
            widget.setText(str(value))
            widget.blockSignals(False)

        def _set_combo_data(widget: QComboBox, value: Any) -> None:
            text = str(value)
            idx = widget.findData(text)
            if idx < 0:
                idx = widget.findText(text)
            widget.blockSignals(True)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            else:
                widget.setCurrentText(text)
            widget.blockSignals(False)

        def _new_float_spin(
            min_val: float,
            max_val: float,
            step: float,
            decimals: int = 3,
        ) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setKeyboardTracking(False)
            return spin

        def _new_int_spin(min_val: int, max_val: int, step: int = 1) -> QSpinBox:
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(step)
            return spin

        def _bind_checkbox(
            form: QFormLayout,
            label: str,
            path: Tuple[str, ...],
            reason: str,
            tooltip: str = "",
        ) -> QCheckBox:
            box = QCheckBox()
            if tooltip:
                box.setToolTip(tooltip)
            box.toggled.connect(lambda _on: self._on_quality_control_changed(reason))
            form.addRow(label, box)
            self._register_quality_binding(path, box.isChecked, lambda v, w=box: _set_checked(w, v))
            return box

        def _bind_combo(
            form: QFormLayout,
            label: str,
            path: Tuple[str, ...],
            reason: str,
            options: Sequence[Tuple[str, str]],
            tooltip: str = "",
        ) -> QComboBox:
            combo = QComboBox()
            for text, value in options:
                combo.addItem(text, value)
            if tooltip:
                combo.setToolTip(tooltip)
            combo.currentIndexChanged.connect(
                lambda _idx: self._on_quality_control_changed(reason)
            )
            form.addRow(label, combo)
            self._register_quality_binding(
                path,
                lambda w=combo: w.currentData(),
                lambda v, w=combo: _set_combo_data(w, v),
            )
            return combo

        def _bind_int_spin(
            form: QFormLayout,
            label: str,
            path: Tuple[str, ...],
            reason: str,
            spin: QSpinBox,
            tooltip: str = "",
        ) -> QSpinBox:
            if tooltip:
                spin.setToolTip(tooltip)
            spin.valueChanged.connect(lambda _v: self._on_quality_control_changed(reason))
            form.addRow(label, spin)
            self._register_quality_binding(path, spin.value, lambda v, w=spin: _set_int(w, v))
            return spin

        def _bind_float_spin(
            form: QFormLayout,
            label: str,
            path: Tuple[str, ...],
            reason: str,
            spin: QDoubleSpinBox,
            tooltip: str = "",
        ) -> QDoubleSpinBox:
            if tooltip:
                spin.setToolTip(tooltip)
            spin.valueChanged.connect(lambda _v: self._on_quality_control_changed(reason))
            form.addRow(label, spin)
            self._register_quality_binding(path, spin.value, lambda v, w=spin: _set_float(w, v))
            return spin

        # Color / gamma / grading
        color_box = QGroupBox("Color / Gamma / Grading")
        color_form = QFormLayout(color_box)
        _bind_checkbox(
            color_form,
            "Enable grading",
            ("color_grading", "enabled"),
            "color_grading.enabled",
        )
        _bind_checkbox(
            color_form,
            "Process in linear",
            ("color_grading", "process_in_linear"),
            "color_grading.process_in_linear",
        )
        _bind_float_spin(
            color_form,
            "White balance shift",
            ("color_grading", "white_balance_shift"),
            "color_grading.white_balance_shift",
            _new_float_spin(-1.0, 1.0, 0.05, decimals=2),
            tooltip="-1 cool, +1 warm",
        )
        _bind_float_spin(
            color_form,
            "Exposure (EV)",
            ("color_grading", "exposure_ev"),
            "color_grading.exposure_ev",
            _new_float_spin(-8.0, 8.0, 0.1, decimals=2),
        )
        _bind_float_spin(
            color_form,
            "Midtone gamma",
            ("color_grading", "midtone_gamma"),
            "color_grading.midtone_gamma",
            _new_float_spin(0.10, 3.00, 0.05, decimals=2),
        )
        _bind_float_spin(
            color_form,
            "Saturation",
            ("color_grading", "saturation"),
            "color_grading.saturation",
            _new_float_spin(0.00, 3.00, 0.05, decimals=2),
        )
        lut_row = QHBoxLayout()
        self.cg_lut_path_edit = QLineEdit()
        self.cg_lut_path_edit.setPlaceholderText("Optional .cube LUT file")
        self.cg_lut_path_edit.editingFinished.connect(
            lambda: self._on_quality_control_changed("color_grading.lut_path")
        )
        lut_row.addWidget(self.cg_lut_path_edit)
        lut_browse = QPushButton("...")
        lut_browse.setFixedWidth(34)
        lut_browse.clicked.connect(self._browse_lut_path)
        lut_row.addWidget(lut_browse)
        color_form.addRow("LUT path", self._wrap(lut_row))
        self._register_quality_binding(
            ("color_grading", "lut_path"),
            lambda w=self.cg_lut_path_edit: w.text().strip(),
            lambda v, w=self.cg_lut_path_edit: _set_text(w, v),
        )
        _bind_float_spin(
            color_form,
            "LUT strength",
            ("color_grading", "lut_strength"),
            "color_grading.lut_strength",
            _new_float_spin(0.00, 1.00, 0.05, decimals=2),
        )
        _bind_float_spin(
            color_form,
            "Denoise",
            ("color_grading", "denoise_strength"),
            "color_grading.denoise_strength",
            _new_float_spin(0.00, 1.00, 0.05, decimals=2),
        )
        _bind_float_spin(
            color_form,
            "Sharpen",
            ("color_grading", "sharpen_strength"),
            "color_grading.sharpen_strength",
            _new_float_spin(0.00, 2.00, 0.05, decimals=2),
        )
        _bind_int_spin(
            color_form,
            "Sharpen radius",
            ("color_grading", "sharpen_radius"),
            "color_grading.sharpen_radius",
            _new_int_spin(0, 8),
        )
        _bind_checkbox(
            color_form,
            "Enforce plausible albedo",
            ("validation", "enforce_plausible_albedo"),
            "validation.enforce_plausible_albedo",
        )
        content_layout.addWidget(color_box)

        # Material intelligence
        material_box = QGroupBox("Material Intelligence")
        material_form = QFormLayout(material_box)
        _bind_checkbox(
            material_form,
            "De-light diffuse",
            ("pbr", "delight_diffuse"),
            "pbr.delight_diffuse",
        )
        _bind_combo(
            material_form,
            "De-light method",
            ("pbr", "delight_method"),
            "pbr.delight_method",
            options=(
                ("Multi-frequency", "multifrequency"),
                ("Gaussian", "gaussian"),
            ),
        )
        _bind_float_spin(
            material_form,
            "De-light strength",
            ("pbr", "delight_strength"),
            "pbr.delight_strength",
            _new_float_spin(0.00, 1.00, 0.05, decimals=2),
        )
        _bind_checkbox(
            material_form,
            "Generate zone masks",
            ("pbr", "material_zone_masks"),
            "pbr.material_zone_masks",
        )
        _bind_checkbox(
            material_form,
            "Apply zone PBR adjustments",
            ("pbr", "apply_zone_pbr_adjustments"),
            "pbr.apply_zone_pbr_adjustments",
        )
        _bind_checkbox(
            material_form,
            "Generate gloss map",
            ("pbr", "generate_gloss"),
            "pbr.generate_gloss",
        )
        content_layout.addWidget(material_box)

        # Channel packing
        packing_box = QGroupBox("Channel Packing")
        packing_form = QFormLayout(packing_box)
        _bind_checkbox(
            packing_form,
            "Enable packing",
            ("orm_packing", "enabled"),
            "orm_packing.enabled",
        )
        _bind_combo(
            packing_form,
            "Packing preset",
            ("orm_packing", "preset"),
            "orm_packing.preset",
            options=(
                ("Unreal ORM", "unreal_orm"),
                ("Unity MAS", "unity_mas"),
                ("Source phong", "source_phong"),
                ("id Tech RMA", "idtech_rma"),
                ("Custom", "custom"),
            ),
        )
        self.orm_suffix_edit = QLineEdit()
        self.orm_suffix_edit.editingFinished.connect(
            lambda: self._on_quality_control_changed("orm_packing.output_suffix")
        )
        packing_form.addRow("Output suffix", self.orm_suffix_edit)
        self._register_quality_binding(
            ("orm_packing", "output_suffix"),
            lambda w=self.orm_suffix_edit: w.text().strip(),
            lambda v, w=self.orm_suffix_edit: _set_text(w, v),
        )
        _bind_checkbox(
            packing_form,
            "Pack gloss in diffuse alpha",
            ("orm_packing", "generate_gloss_in_diffuse_alpha"),
            "orm_packing.generate_gloss_in_diffuse_alpha",
        )
        _bind_combo(
            packing_form,
            "Gloss source",
            ("orm_packing", "gloss_source"),
            "orm_packing.gloss_source",
            options=(
                ("Roughness (invert)", "roughness"),
                ("Gloss map", "gloss"),
            ),
        )
        _bind_checkbox(
            packing_form,
            "Overwrite existing alpha",
            ("orm_packing", "overwrite_existing_alpha"),
            "orm_packing.overwrite_existing_alpha",
        )
        content_layout.addWidget(packing_box)

        # Emissive / reflection masks
        em_box = QGroupBox("Emissive / Reflection")
        em_form = QFormLayout(em_box)
        _bind_checkbox(
            em_form,
            "Enable emissive detection",
            ("emissive", "enabled"),
            "emissive.enabled",
        )
        _bind_float_spin(
            em_form,
            "Emissive luminance",
            ("emissive", "luminance_threshold"),
            "emissive.luminance_threshold",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_float_spin(
            em_form,
            "Emissive saturation",
            ("emissive", "saturation_threshold"),
            "emissive.saturation_threshold",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_float_spin(
            em_form,
            "Emissive value",
            ("emissive", "value_threshold"),
            "emissive.value_threshold",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_float_spin(
            em_form,
            "Emissive boost",
            ("emissive", "boost"),
            "emissive.boost",
            _new_float_spin(0.00, 4.00, 0.05, decimals=2),
        )
        _bind_checkbox(
            em_form,
            "Enable reflection mask",
            ("reflection_mask", "enabled"),
            "reflection_mask.enabled",
        )
        _bind_float_spin(
            em_form,
            "Reflection metalness weight",
            ("reflection_mask", "metalness_weight"),
            "reflection_mask.metalness_weight",
            _new_float_spin(0.00, 2.00, 0.05, decimals=2),
        )
        _bind_float_spin(
            em_form,
            "Reflection gloss weight",
            ("reflection_mask", "gloss_weight"),
            "reflection_mask.gloss_weight",
            _new_float_spin(0.00, 2.00, 0.05, decimals=2),
        )
        _bind_float_spin(
            em_form,
            "Reflection bias",
            ("reflection_mask", "bias"),
            "reflection_mask.bias",
            _new_float_spin(-1.00, 1.00, 0.05, decimals=2),
        )
        content_layout.addWidget(em_box)

        # Seam / tiling quality
        seam_box = QGroupBox("Seam Repair / Tiling Quality")
        seam_form = QFormLayout(seam_box)
        _bind_checkbox(
            seam_form,
            "Enable seam repair",
            ("seam_repair", "enabled"),
            "seam_repair.enabled",
        )
        _bind_checkbox(
            seam_form,
            "Repair only tileable assets",
            ("seam_repair", "only_tileable"),
            "seam_repair.only_tileable",
        )
        _bind_int_spin(
            seam_form,
            "Repair border width",
            ("seam_repair", "repair_border_width"),
            "seam_repair.repair_border_width",
            _new_int_spin(1, 128),
        )
        _bind_float_spin(
            seam_form,
            "Seam detect threshold",
            ("seam_repair", "detect_threshold"),
            "seam_repair.detect_threshold",
            _new_float_spin(0.001, 1.000, 0.01, decimals=3),
        )
        _bind_float_spin(
            seam_form,
            "Seam blend strength",
            ("seam_repair", "blend_strength"),
            "seam_repair.blend_strength",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_checkbox(
            seam_form,
            "Enable tiling quality scoring",
            ("tiling_quality", "enabled"),
            "tiling_quality.enabled",
        )
        _bind_float_spin(
            seam_form,
            "Tiling warn score",
            ("tiling_quality", "warn_score"),
            "tiling_quality.warn_score",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_float_spin(
            seam_form,
            "Tiling fail score",
            ("tiling_quality", "fail_score"),
            "tiling_quality.fail_score",
            _new_float_spin(0.00, 1.00, 0.01, decimals=2),
        )
        _bind_int_spin(
            seam_form,
            "Auto-flag top N",
            ("tiling_quality", "auto_flag_top_n"),
            "tiling_quality.auto_flag_top_n",
            _new_int_spin(0, 1000),
        )
        content_layout.addWidget(seam_box)

        controls_hint = QLabel(
            "These quick controls target high-impact quality settings. "
            "The Runtime Config tree remains available for all advanced keys."
        )
        controls_hint.setWordWrap(True)
        controls_hint.setStyleSheet("color: #707070;")
        content_layout.addWidget(controls_hint)
        content_layout.addStretch(1)

        scroll.setWidget(container)
        parent_layout.addWidget(scroll)

    def _build_asset_table(self, parent_layout: QVBoxLayout) -> None:
        self.asset_table = QTableWidget(0, 8)
        self.asset_table.setHorizontalHeaderLabels(
            [
                "Sel",
                "Asset",
                "Accepted",
                "Type",
                "Resolution",
                "Channels",
                "Upscaled",
                "Flags",
            ]
        )
        self.asset_table.setAlternatingRowColors(True)
        self.asset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.asset_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.asset_table.setMinimumHeight(320)
        self.asset_table.itemSelectionChanged.connect(self._on_asset_selection_changed)
        self.asset_table.itemChanged.connect(self._on_asset_item_changed)
        self.asset_toggle_checks_shortcut = QShortcut(
            QKeySequence("Ctrl+Shift+A"),
            self.asset_table,
        )
        self.asset_toggle_checks_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.asset_toggle_checks_shortcut.activated.connect(
            self._toggle_all_listed_asset_checks
        )
        parent_layout.addWidget(self.asset_table, stretch=2)

    def _build_preview_panel(self, parent_layout: QVBoxLayout) -> None:
        panel = QGroupBox("Interactive Preview")
        layout = QVBoxLayout(panel)

        map_row = QHBoxLayout()
        map_row.addWidget(QLabel("Map mode"))
        self.preview_map_mode_combo = QComboBox()
        self.preview_map_mode_combo.addItem("Auto (Best Available)", "__auto__")
        for label, _suffix in MAP_OPTIONS:
            self.preview_map_mode_combo.addItem(label, label)
        self.preview_map_mode_combo.currentIndexChanged.connect(
            self._on_preview_map_mode_changed
        )
        map_row.addWidget(self.preview_map_mode_combo)

        map_row.addWidget(QLabel("Resolved map"))
        self.map_combo = QComboBox()
        self.map_combo.setMinimumWidth(280)
        self.map_combo.setMinimumContentsLength(24)
        self.map_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.map_combo.setMaxVisibleItems(20)
        self.map_combo.view().setMinimumWidth(320)
        self.map_combo.currentIndexChanged.connect(self._on_active_map_changed)
        map_row.addWidget(self.map_combo)
        map_row.addWidget(QLabel("Preview mode"))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItem("Split", "split")
        self.preview_mode_combo.addItem("Before", "before")
        self.preview_mode_combo.addItem("After", "after")
        self.preview_mode_combo.currentIndexChanged.connect(self._on_preview_mode_changed)
        map_row.addWidget(self.preview_mode_combo)
        map_row.addStretch(1)
        layout.addLayout(map_row)

        preview_action_row = QHBoxLayout()
        self.render_preview_btn = QPushButton("Render Preview")
        self.render_preview_btn.clicked.connect(
            lambda: self._trigger_preview_render("manual")
        )
        self.refresh_preview_btn = QPushButton("Refresh Preview")
        self.refresh_preview_btn.clicked.connect(self._on_refresh_preview_images)
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.stop_preview_btn.setEnabled(False)
        self.stop_preview_btn.clicked.connect(self._on_stop_preview)
        self.always_preview_toggle = QPushButton("Always Preview: Off")
        self.always_preview_toggle.setCheckable(True)
        self.always_preview_toggle.toggled.connect(self._on_always_preview_toggled)
        preview_action_row.addWidget(self.render_preview_btn)
        preview_action_row.addWidget(self.refresh_preview_btn)
        preview_action_row.addWidget(self.stop_preview_btn)
        preview_action_row.addWidget(self.always_preview_toggle)
        preview_action_row.addStretch(1)
        layout.addLayout(preview_action_row)

        self.map_toggle_list = QListWidget()
        self.map_toggle_list.setFlow(QListView.Flow.LeftToRight)
        self.map_toggle_list.setWrapping(True)
        self.map_toggle_list.setMaximumHeight(90)
        self.map_toggle_list.itemChanged.connect(self._refresh_map_combo)
        map_toggle_actions = QHBoxLayout()
        maps_all_btn = QPushButton("Maps: All")
        maps_all_btn.clicked.connect(lambda: self._set_map_visibility("all"))
        maps_none_btn = QPushButton("Maps: None")
        maps_none_btn.clicked.connect(lambda: self._set_map_visibility("none"))
        maps_core_btn = QPushButton("Maps: Core")
        maps_core_btn.clicked.connect(lambda: self._set_map_visibility("core"))
        map_toggle_actions.addWidget(maps_all_btn)
        map_toggle_actions.addWidget(maps_none_btn)
        map_toggle_actions.addWidget(maps_core_btn)
        map_toggle_actions.addStretch(1)
        layout.addLayout(map_toggle_actions)
        for label, _suffix in MAP_OPTIONS:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.map_toggle_list.addItem(item)
        layout.addWidget(self.map_toggle_list)

        self.before_path_label = QLabel("Input: -")
        self.after_path_label = QLabel("Output: -")
        self.before_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.after_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Keep labels for internal state/debug, but hide them from UI per UX request.
        self.before_path_label.setVisible(False)
        self.after_path_label.setVisible(False)

        self.viewer = SplitImageViewer()
        self.viewer.zoom_changed.connect(self._on_viewer_zoom_changed)
        self.viewer.split_ratio_changed.connect(self._on_viewer_split_changed)
        layout.addWidget(self.viewer, stretch=1)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Split"))
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self._on_split_changed)
        control_row.addWidget(self.split_slider, stretch=1)
        self.swap_split_btn = QPushButton("Swap L<>R: Off")
        self.swap_split_btn.setCheckable(True)
        self.swap_split_btn.toggled.connect(self._on_swap_split_toggled)
        control_row.addWidget(self.swap_split_btn)

        control_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 800)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        control_row.addWidget(self.zoom_slider, stretch=1)
        self.zoom_value_label = QLabel("100%")
        control_row.addWidget(self.zoom_value_label)
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.viewer.reset_view)
        control_row.addWidget(reset_btn)
        layout.addLayout(control_row)
        self._on_preview_mode_changed(self.preview_mode_combo.currentIndex())

        parent_layout.addWidget(panel, stretch=3)

    def _build_log_panel(self, parent_layout: QVBoxLayout) -> None:
        log_box = QGroupBox("Runtime Log")
        log_layout = QVBoxLayout(log_box)
        progress_row = QHBoxLayout()
        self.pipeline_progress_label = QLabel("Progress: idle")
        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        progress_row.addWidget(self.pipeline_progress_label)
        progress_row.addWidget(self.pipeline_progress, stretch=1)
        log_layout.addLayout(progress_row)

        overall_progress_row = QHBoxLayout()
        self.pipeline_overall_progress_label = QLabel("Overall: idle")
        self.pipeline_overall_progress = QProgressBar()
        self.pipeline_overall_progress.setRange(0, 100)
        self.pipeline_overall_progress.setValue(0)
        overall_progress_row.addWidget(self.pipeline_overall_progress_label)
        overall_progress_row.addWidget(self.pipeline_overall_progress, stretch=1)
        log_layout.addLayout(overall_progress_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Level"))
        show_all_btn = QPushButton("All")
        show_all_btn.clicked.connect(self._on_log_filter_show_all)
        filter_row.addWidget(show_all_btn)
        self._log_filter_buttons = {}
        for level in LOG_LEVEL_FILTERS:
            label = level.title() if level != "OTHER" else "Other"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(
                lambda checked, lvl=level: self._on_log_level_toggled(lvl, checked)
            )
            self._log_filter_buttons[level] = btn
            filter_row.addWidget(btn)
        filter_row.addStretch(1)
        log_layout.addLayout(filter_row)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        self.log_search_edit = QLineEdit()
        self.log_search_edit.setPlaceholderText("Filter log lines (case-insensitive)")
        self.log_search_edit.textChanged.connect(self._on_log_search_changed)
        search_row.addWidget(self.log_search_edit, stretch=1)
        clear_search = QPushButton("Clear Search")
        clear_search.clicked.connect(self._on_clear_log_search)
        clear_logs = QPushButton("Clear Logs")
        clear_logs.clicked.connect(self._clear_logs)
        search_row.addWidget(clear_search)
        search_row.addWidget(clear_logs)
        log_layout.addLayout(search_row)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(5000)
        log_layout.addWidget(self.log_text)
        parent_layout.addWidget(log_box, stretch=3)

    def _browse_lut_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose LUT file",
            self.cg_lut_path_edit.text().strip() or ".",
            "LUT Files (*.cube);;All Files (*)",
        )
        if path:
            self.cg_lut_path_edit.setText(path)
            self._on_quality_control_changed("color_grading.lut_path")

    def _apply_quality_controls_from_config(self, cfg: PipelineConfig) -> None:
        for path, setter in self._quality_field_setters.items():
            try:
                setter(self._get_dataclass_path_value(cfg, path))
            except Exception:
                continue

    def _apply_quality_controls_to_config(self, cfg: PipelineConfig) -> None:
        for path, getter in self._quality_field_getters.items():
            try:
                value = getter()
                set_dataclass_path(cfg, path, value)
            except Exception:
                continue

    def _sync_quality_control_from_tree_item(self, item: QTreeWidgetItem) -> None:
        if item is None:
            return
        path_data = item.data(0, ConfigTreeEditor.ROLE_PATH)
        if not isinstance(path_data, tuple):
            return
        path = tuple(path_data)
        setter = self._quality_field_setters.get(path)
        if setter is None:
            return

        template = item.data(0, ConfigTreeEditor.ROLE_TEMPLATE)
        value_text = item.text(1)
        try:
            parsed = parse_typed_value(value_text, template)
        except Exception:
            return
        setter(parsed)

    def _on_config_tree_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        self._sync_quality_control_from_tree_item(item)
        self._schedule_auto_preview("config_tree")

    def _refresh_config_health(self) -> None:
        try:
            self._collect_runtime_config(commit=False, emit_fixup_log=False)
        except Exception as exc:
            self.config_health_label.setText(f"Config status: invalid ({exc})")
            self.config_health_label.setStyleSheet("color: #C94F4F;")
            return
        self.config_health_label.setText("Config status: valid")
        self.config_health_label.setStyleSheet("color: #3FA55B;")

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _load_ui_settings(self) -> None:
        config_path = self._settings.value("paths/config", "config.yaml", str)
        if config_path:
            self.config_path_edit.setText(config_path)

        input_dir = self._settings.value("paths/input", "", str)
        if input_dir:
            self.input_dir_edit.setText(input_dir)

        output_dir = self._settings.value("paths/output", "", str)
        if output_dir:
            self.output_dir_edit.setText(output_dir)

        device = self._settings.value("runtime/device", "", str)
        if device:
            self.device_combo.setCurrentText(device)

        workers = self._settings.value("runtime/workers", 4, int)
        try:
            workers_int = int(workers)
        except (TypeError, ValueError):
            workers_int = 4
        workers_int = max(
            self.workers_spin.minimum(),
            min(self.workers_spin.maximum(), workers_int),
        )
        self.workers_spin.setValue(workers_int)

        self.dry_run_check.setChecked(bool(self._settings.value("runtime/dry_run", False, bool)))
        self.reset_checkpoint_check.setChecked(
            bool(self._settings.value("runtime/reset_checkpoint", False, bool))
        )

        asset_source = self._settings.value("asset_list/source", "accepted", str)
        source_idx = self.asset_source_combo.findData(asset_source)
        if source_idx >= 0:
            self.asset_source_combo.setCurrentIndex(source_idx)

        preview_mode = self._settings.value("preview/mode", "split", str)
        mode_idx = self.preview_mode_combo.findData(preview_mode)
        if mode_idx >= 0:
            self.preview_mode_combo.setCurrentIndex(mode_idx)
        self._on_preview_mode_changed(self.preview_mode_combo.currentIndex())

        preview_map_mode = self._settings.value("preview/map_mode", "__auto__", str)
        map_mode_idx = self.preview_map_mode_combo.findData(preview_map_mode)
        if map_mode_idx >= 0:
            self.preview_map_mode_combo.setCurrentIndex(map_mode_idx)
        swap_split_sides = bool(self._settings.value("preview/swap_split_sides", False, bool))
        self.swap_split_btn.setChecked(swap_split_sides)
        always_preview = bool(self._settings.value("preview/always_enabled", False, bool))
        self.always_preview_toggle.setChecked(always_preview)
        self._on_always_preview_toggled(always_preview)

        for phase in PHASE_OPTIONS:
            default = True
            checked = bool(self._settings.value(f"phases/{phase}", default, bool))
            self.phase_checks[phase].setChecked(checked)

        self.map_toggle_list.blockSignals(True)
        try:
            for idx in range(self.map_toggle_list.count()):
                item = self.map_toggle_list.item(idx)
                checked = bool(self._settings.value(f"maps/{item.text()}", True, bool))
                state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                item.setCheckState(state)
        finally:
            self.map_toggle_list.blockSignals(False)

        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        splitter = self._settings.value("window/splitter")
        if splitter is not None:
            self.main_splitter.restoreState(splitter)

    def _save_ui_settings(self) -> None:
        self._settings.setValue("paths/config", self.config_path_edit.text().strip())
        self._settings.setValue("paths/input", self.input_dir_edit.text().strip())
        self._settings.setValue("paths/output", self.output_dir_edit.text().strip())
        self._settings.setValue("runtime/device", self.device_combo.currentText())
        self._settings.setValue("runtime/workers", self.workers_spin.value())
        self._settings.setValue("runtime/dry_run", self.dry_run_check.isChecked())
        self._settings.setValue(
            "runtime/reset_checkpoint",
            self.reset_checkpoint_check.isChecked(),
        )
        self._settings.setValue(
            "asset_list/source",
            self.asset_source_combo.currentData(),
        )
        self._settings.setValue(
            "preview/mode",
            self.preview_mode_combo.currentData(),
        )
        self._settings.setValue(
            "preview/map_mode",
            self.preview_map_mode_combo.currentData(),
        )
        self._settings.setValue(
            "preview/swap_split_sides",
            self.swap_split_btn.isChecked(),
        )
        self._settings.setValue(
            "preview/always_enabled",
            self.always_preview_toggle.isChecked(),
        )

        for phase, checkbox in self.phase_checks.items():
            self._settings.setValue(f"phases/{phase}", checkbox.isChecked())

        for idx in range(self.map_toggle_list.count()):
            item = self.map_toggle_list.item(idx)
            self._settings.setValue(
                f"maps/{item.text()}",
                item.checkState() == Qt.CheckState.Checked,
            )

        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/splitter", self.main_splitter.saveState())
        self._settings.sync()

    def _load_pipeline_config(self, cfg_path: str) -> Tuple[PipelineConfig, Optional[str]]:
        """Load config from YAML."""
        return PipelineConfig.from_yaml(cfg_path), None

    def _load_default_config(self) -> None:
        input_override = self.input_dir_edit.text().strip()
        output_override = self.output_dir_edit.text().strip()
        device_override = self.device_combo.currentText()
        workers_override = self.workers_spin.value()
        dry_run_override = self.dry_run_check.isChecked()

        cfg_path = self.config_path_edit.text().strip()
        if cfg_path and os.path.exists(cfg_path):
            try:
                loaded, note = self._load_pipeline_config(cfg_path)
                self.config = loaded
                self._append_log(f"Loaded config: {cfg_path}")
                if note:
                    self._append_log(note)
            except Exception as exc:
                self.config = PipelineConfig()
                self._append_log(
                    f"Config load failed ({cfg_path}); using defaults. Error: {exc}"
                )
        else:
            self.config = PipelineConfig()
            self._append_log("Loaded default config.")

        self._apply_config_to_controls()
        if input_override:
            self.input_dir_edit.setText(input_override)
        if output_override:
            self.output_dir_edit.setText(output_override)
        if device_override:
            self.device_combo.setCurrentText(device_override)
        self.workers_spin.setValue(
            max(self.workers_spin.minimum(), min(self.workers_spin.maximum(), workers_override))
        )
        self.dry_run_check.setChecked(dry_run_override)
        self.config_editor.set_config(self.config)
        self._apply_quality_controls_from_config(self.config)
        self._refresh_map_combo()
        self._refresh_config_health()

    def _apply_config_to_controls(self) -> None:
        self.input_dir_edit.setText(self.config.input_dir)
        self.output_dir_edit.setText(self.config.output_dir)
        self.device_combo.setCurrentText(self.config.device)
        self.workers_spin.setValue(self.config.max_workers)
        self.dry_run_check.setChecked(self.config.dry_run)
        self._apply_quality_controls_from_config(self.config)
        self._refresh_config_health()
        self._refresh_asset_table_statuses()

    def _collect_runtime_config(
        self,
        commit: bool = True,
        emit_fixup_log: bool = True,
    ) -> PipelineConfig:
        cfg = self.config_editor.build_config()
        cfg.input_dir = self.input_dir_edit.text().strip()
        cfg.output_dir = self.output_dir_edit.text().strip()
        cfg.device = self.device_combo.currentText().strip()
        cfg.max_workers = self.workers_spin.value()
        cfg.dry_run = self.dry_run_check.isChecked()
        self._apply_quality_controls_to_config(cfg)

        if not cfg.input_dir:
            raise ValueError("input directory is required")
        if not cfg.output_dir:
            raise ValueError("output directory is required")

        before_half = bool(cfg.upscale.half_precision)
        cfg.validate()
        cfg.apply_runtime_fixups()
        if emit_fixup_log and before_half and not cfg.upscale.half_precision:
            self._append_log(
                "Runtime config normalized: disabled upscale.half_precision "
                "because resolved device is CPU."
            )
        if commit:
            self.config = cfg
        return cfg

    def _on_apply_runtime_config(self) -> None:
        old_config = copy.deepcopy(self.config)
        try:
            cfg = self._collect_runtime_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        self._push_config_undo(old_config)
        self.config_editor.set_config(cfg)
        self._apply_quality_controls_from_config(cfg)
        self._append_log("Runtime config updated.")
        self._refresh_config_health()
        self._schedule_auto_preview("apply_runtime_config")

    def _on_load_config(self) -> None:
        path = self.config_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Config", "Please provide a config path.")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Config", f"Config file not found:\n{path}")
            return
        old_config = copy.deepcopy(self.config)
        try:
            self.config, note = self._load_pipeline_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        self._push_config_undo(old_config)
        self._apply_config_to_controls()
        self.config_editor.set_config(self.config)
        self._apply_quality_controls_from_config(self.config)
        self._append_log(f"Loaded config: {path}")
        if note:
            self._append_log(note)
        self._refresh_config_health()
        self._schedule_auto_preview("load_config")

    def _on_save_config(self) -> None:
        try:
            cfg = self._collect_runtime_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        path = self.config_path_edit.text().strip() or "config.yaml"
        try:
            cfg.to_yaml(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._append_log(f"Saved config: {path}")

    def _push_config_undo(self, old_config: PipelineConfig) -> None:
        """Push old config onto undo stack, clear redo stack."""
        self._config_undo_stack.append(copy.deepcopy(old_config))
        if len(self._config_undo_stack) > self._CONFIG_UNDO_LIMIT:
            self._config_undo_stack.pop(0)
        self._config_redo_stack.clear()
        self._update_undo_redo_buttons()

    def _undo_config(self) -> None:
        """Restore previous config state."""
        if not self._config_undo_stack:
            return
        self._config_redo_stack.append(copy.deepcopy(self.config))
        self.config = self._config_undo_stack.pop()
        self._restore_config_to_ui()
        self._update_undo_redo_buttons()
        self._append_log("Config change undone.")

    def _redo_config(self) -> None:
        """Re-apply previously undone config change."""
        if not self._config_redo_stack:
            return
        self._config_undo_stack.append(copy.deepcopy(self.config))
        self.config = self._config_redo_stack.pop()
        self._restore_config_to_ui()
        self._update_undo_redo_buttons()
        self._append_log("Config change redone.")

    def _restore_config_to_ui(self) -> None:
        """Sync all UI controls from self.config."""
        self._apply_config_to_controls()
        self.config_editor.set_config(self.config)
        self._refresh_config_health()
        self._schedule_auto_preview("undo_redo")

    def _update_undo_redo_buttons(self) -> None:
        """Enable/disable undo/redo buttons based on stack state."""
        self.undo_btn.setEnabled(bool(self._config_undo_stack))
        self.redo_btn.setEnabled(bool(self._config_redo_stack))

    def _browse_config_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose config file",
            self.config_path_edit.text().strip() or ".",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.config_path_edit.setText(path)

    def _browse_dir(self, target: QLineEdit) -> None:
        current = target.text().strip() or "."
        path = QFileDialog.getExistingDirectory(self, "Select directory", current)
        if path:
            target.setText(path)

    def _build_all_input_records(
        self,
        input_dir: str,
        accepted_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        accepted_by_path = {str(r.get("filepath", "")): r for r in accepted_records}
        all_records: List[Dict[str, Any]] = []

        for root, _, files in os.walk(input_dir):
            for filename in sorted(files):
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, input_dir)
                accepted = accepted_by_path.get(rel_path)

                if accepted is not None:
                    rec = dict(accepted)
                    rec["accepted"] = True
                    all_records.append(rec)
                    continue

                all_records.append(
                    {
                        "filepath": rel_path,
                        "filename": filename,
                        "texture_type": "unsupported",
                        "original_width": 0,
                        "original_height": 0,
                        "channels": 0,
                        "has_alpha": False,
                        "is_tileable": False,
                        "is_hero": False,
                        "material_category": "unknown",
                        "file_size_kb": round(os.path.getsize(abs_path) / 1024.0, 1),
                        "is_gloss": False,
                        "accepted": False,
                    }
                )

        return all_records

    def _on_asset_source_changed(self, _index: int) -> None:
        source = self.asset_source_combo.currentData()
        if source == "all_input":
            self.records = list(self.all_input_records)
        else:
            self.records = list(self.accepted_records)
        valid_paths = {
            str(rec.get("filepath") or "")
            for rec in self.records
            if str(rec.get("filepath") or "")
        }
        if self._auto_check_all_after_scan:
            self._checked_asset_relpaths = set(valid_paths)
            self._auto_check_all_after_scan = False
        else:
            self._checked_asset_relpaths = {
                path for path in self._checked_asset_relpaths if path in valid_paths
            }
        self._populate_asset_table()

    def _on_scan_assets(self) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._append_log("Pipeline is running; scan request ignored.")
            return
        if self._scan_thread is not None and self._scan_thread.isRunning():
            self._append_log("Scan already in progress.")
            return

        try:
            cfg = self._collect_runtime_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        if not os.path.isdir(cfg.input_dir):
            QMessageBox.warning(self, "Input", f"Input directory not found:\n{cfg.input_dir}")
            return

        self._scan_input_dir = cfg.input_dir
        self._set_scan_running_state(True)
        self._append_log(f"Scanning assets in {cfg.input_dir} ...")

        self._scan_thread = QThread(self)
        self._scan_worker = ScanWorker(cfg)
        self._scan_worker.moveToThread(self._scan_thread)

        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.failed.connect(self._scan_thread.quit)
        self._scan_thread.finished.connect(self._cleanup_scan_worker)
        self._scan_thread.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.start()

    def _set_scan_running_state(self, running: bool) -> None:
        pipeline_running = (
            self._worker_thread is not None and self._worker_thread.isRunning()
        )
        self.scan_btn.setEnabled((not running) and (not pipeline_running))
        if running:
            self.statusBar().showMessage("Scanning assets...")

    @pyqtSlot(object)
    def _on_scan_finished(self, accepted_records: object) -> None:
        records = [dict(r) for r in (accepted_records or [])]
        for rec in records:
            rec["accepted"] = True

        self.accepted_records = records
        self.all_input_records = self._build_all_input_records(self._scan_input_dir, records)
        self._auto_check_all_after_scan = True
        self._on_asset_source_changed(self.asset_source_combo.currentIndex())

        self._append_log(
            f"Scanned accepted={len(self.accepted_records)}, all={len(self.all_input_records)}"
        )
        self.statusBar().showMessage(
            f"Scanned accepted={len(self.accepted_records)}, all={len(self.all_input_records)}"
        )

    @pyqtSlot(str)
    def _on_scan_failed(self, message: str) -> None:
        self._append_log(f"Scan failed: {message}")
        QMessageBox.critical(self, "Scan Error", message)

    def _cleanup_scan_worker(self) -> None:
        self._scan_worker = None
        self._scan_thread = None
        self._set_scan_running_state(False)

    @staticmethod
    def _format_output_timestamp(path: str) -> str:
        """Return filesystem mtime in a compact local-time format."""
        try:
            ts = os.path.getmtime(path)
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return "unknown"

    def _get_upscale_status_text(self, rec: Dict[str, Any]) -> str:
        """Return per-asset upscale status based on existing output files."""
        rel_path = str(rec.get("filepath") or "")
        if not rel_path:
            return "-"
        output_dir = self.output_dir_edit.text().strip() or self.config.output_dir
        if not output_dir:
            return "not upscaled"
        base_output = find_output_map_file(
            rel_path,
            output_dir,
            suffix="",
            ext_priority=OUTPUT_EXT_PRIORITY,
        )
        if not base_output:
            return "not upscaled"
        stamp = self._format_output_timestamp(base_output)
        return f"yes ({stamp})"

    def _refresh_asset_table_statuses(self) -> None:
        """Refresh table rows so output-derived status columns stay current."""
        if not self.records:
            return
        self._populate_asset_table()

    def _selected_asset_relpaths(self) -> List[str]:
        selected: List[str] = []
        seen: set[str] = set()
        for item in self.asset_table.selectedItems():
            if item.column() != 1:
                continue
            rel_path = str(item.data(int(Qt.ItemDataRole.UserRole)) or "")
            if rel_path and rel_path not in seen:
                selected.append(rel_path)
                seen.add(rel_path)
        return selected

    def _checked_asset_paths(self) -> List[str]:
        checked: List[str] = []
        for row in range(self.asset_table.rowCount()):
            check_item = self.asset_table.item(row, 0)
            if check_item is None:
                continue
            if check_item.checkState() != Qt.CheckState.Checked:
                continue
            rel_path = str(check_item.data(int(Qt.ItemDataRole.UserRole)) or "")
            if rel_path:
                checked.append(rel_path)
        return checked

    def _set_all_listed_asset_checks(self, checked: bool) -> None:
        listed_paths = {
            str(rec.get("filepath") or "")
            for rec in self.records
            if str(rec.get("filepath") or "")
        }
        if not listed_paths:
            self._checked_asset_relpaths.clear()
            return

        target_state = (
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        self._asset_table_populating = True
        try:
            for row in range(self.asset_table.rowCount()):
                check_item = self.asset_table.item(row, 0)
                if check_item is None:
                    continue
                check_item.setCheckState(target_state)
        finally:
            self._asset_table_populating = False

        if checked:
            self._checked_asset_relpaths = set(listed_paths)
        else:
            self._checked_asset_relpaths.clear()

    def _toggle_all_listed_asset_checks(self) -> None:
        listed_paths = {
            str(rec.get("filepath") or "")
            for rec in self.records
            if str(rec.get("filepath") or "")
        }
        if not listed_paths:
            return
        all_checked = listed_paths.issubset(self._checked_asset_relpaths)
        self._set_all_listed_asset_checks(not all_checked)

    def _populate_asset_table(self) -> None:
        selected_relpath = self._selected_asset_relpath()
        checked_relpaths = set(self._checked_asset_relpaths)
        selected_row = -1
        self._asset_table_populating = True
        self.asset_table.setRowCount(len(self.records))
        for i, rec in enumerate(self.records):
            rel_path = str(rec.get("filepath", ""))
            flags: List[str] = []
            if rec.get("is_tileable"):
                flags.append("tile")
            if rec.get("is_hero"):
                flags.append("hero")
            if rec.get("has_alpha"):
                flags.append("alpha")
            if rec.get("is_gloss"):
                flags.append("gloss")
            if not rec.get("accepted", True):
                flags.append("unsupported")

            w = int(rec.get("original_width") or 0)
            h = int(rec.get("original_height") or 0)
            resolution = f"{w}x{h}" if w > 0 and h > 0 else "-"
            channels = rec.get("channels") or "-"
            accepted_text = "yes" if rec.get("accepted", True) else "no"
            upscaled_text = self._get_upscale_status_text(rec)

            check_item = QTableWidgetItem("")
            check_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            check_item.setData(int(Qt.ItemDataRole.UserRole), rel_path)
            check_item.setCheckState(
                Qt.CheckState.Checked
                if rel_path in checked_relpaths
                else Qt.CheckState.Unchecked
            )
            self.asset_table.setItem(i, 0, check_item)

            values = [
                rel_path,
                accepted_text,
                rec.get("texture_type", ""),
                resolution,
                str(channels),
                upscaled_text,
                ", ".join(flags) if flags else "-",
            ]
            for col, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setData(int(Qt.ItemDataRole.UserRole), rel_path)
                self.asset_table.setItem(i, col, item)
            if selected_relpath and rel_path == selected_relpath:
                selected_row = i

        self._asset_table_populating = False
        self.asset_table.resizeColumnsToContents()
        if selected_row >= 0:
            self.asset_table.selectRow(selected_row)
        elif self.records:
            self.asset_table.selectRow(0)
        self._checked_asset_relpaths = set(self._checked_asset_paths())

    def _selected_asset_relpath(self) -> Optional[str]:
        row = self.asset_table.currentRow()
        if row < 0:
            return None
        item = self.asset_table.item(row, 1)
        if not item:
            return None
        return str(item.data(int(Qt.ItemDataRole.UserRole)) or "")

    def _on_asset_item_changed(self, item: QTableWidgetItem) -> None:
        if self._asset_table_populating:
            return
        if item.column() != 0:
            return
        rel_path = str(item.data(int(Qt.ItemDataRole.UserRole)) or "")
        if not rel_path:
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._checked_asset_relpaths.add(rel_path)
        else:
            self._checked_asset_relpaths.discard(rel_path)

    def _on_asset_selection_changed(self) -> None:
        self._refresh_preview_for_selected_asset()

    def _resolve_map_paths(self, rel_path: str) -> Dict[str, str]:
        result_entry = self.latest_results.get(rel_path, {})
        paths = resolve_map_paths(rel_path, result_entry, self.config.output_dir)

        input_base = os.path.join(self.config.input_dir, rel_path)
        if os.path.exists(input_base):
            paths.setdefault("Base", input_base)

        return paths

    def _iter_asset_output_candidates(self, output_dir: str, rel_path: str) -> List[str]:
        """Enumerate map output paths for one asset across known suffixes/exts."""
        rel = Path(rel_path)
        base_dir = Path(output_dir) / rel.parent
        base_stem = rel.stem
        paths: List[str] = []
        for _label, suffix in MAP_OPTIONS:
            for ext in OUTPUT_EXT_PRIORITY:
                paths.append(str(base_dir / f"{base_stem}{suffix}{ext}"))
        return paths

    def _clear_existing_output_targets(self, output_dir: str, rel_path: str) -> int:
        """Delete existing output files for one asset to avoid stale preview maps."""
        removed = 0
        for path in self._iter_asset_output_candidates(output_dir, rel_path):
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except Exception as exc:
                self._append_log(f"[preview] Could not remove stale output: {path} ({exc})")
        return removed

    def _invalidate_cached_asset_images(
        self, input_dir: str, output_dir: str, rel_path: str
    ) -> int:
        """Drop cached preview images for one asset so rerenders always refresh."""
        candidates = set(self._iter_asset_output_candidates(output_dir, rel_path))
        candidates.add(os.path.join(input_dir, rel_path))

        dropped = 0
        for path in candidates:
            cached = self._image_cache.pop(path, None)
            if cached is not None:
                _img, size = cached
                self._image_cache_bytes = max(0, self._image_cache_bytes - int(size))
                dropped += 1
        return dropped

    def _refresh_preview_for_selected_asset(self) -> None:
        rel_path = self._selected_asset_relpath()
        if not rel_path:
            self._current_map_paths = {}
            self._refresh_map_combo()
            self.viewer.set_images(None, None)
            self.before_path_label.setText("Input: -")
            self.after_path_label.setText("Output: -")
            return

        self._current_map_paths = self._resolve_map_paths(rel_path)
        self._refresh_map_combo()

    def _refresh_map_combo(self) -> None:
        previous_label = self.map_combo.currentText()
        requested_mode = str(self.preview_map_mode_combo.currentData() or "__auto__")

        self.map_combo.blockSignals(True)
        self.map_combo.clear()

        for i in range(self.map_toggle_list.count()):
            item = self.map_toggle_list.item(i)
            label = item.text()
            if item.checkState() != Qt.CheckState.Checked:
                continue
            path = self._current_map_paths.get(label)
            if path:
                self.map_combo.addItem(label, path)

        has_maps = self.map_combo.count() > 0
        if not has_maps:
            self.map_combo.addItem("(No maps available)", None)
            self.map_combo.setEnabled(False)
        else:
            self.map_combo.setEnabled(True)

        self.map_combo.blockSignals(False)

        preferred_label = requested_mode if requested_mode != "__auto__" else previous_label
        if has_maps and preferred_label:
            idx = self.map_combo.findText(preferred_label)
            if idx >= 0:
                self.map_combo.setCurrentIndex(idx)
            elif self.map_combo.count() > 0:
                self.map_combo.setCurrentIndex(0)
        elif has_maps and self.map_combo.count() > 0:
            self.map_combo.setCurrentIndex(0)

        self._update_preview_images()

    def _on_preview_map_mode_changed(self, _index: int) -> None:
        self._refresh_map_combo()
        self._schedule_auto_preview("preview_map_mode")

    def _on_active_map_changed(self, _index: int) -> None:
        self._update_preview_images()

    def _on_preview_mode_changed(self, _index: int) -> None:
        mode = str(self.preview_mode_combo.currentData() or "split")
        self.viewer.set_preview_mode(mode)
        split_mode = mode == "split"
        self.split_slider.setEnabled(split_mode)
        self.swap_split_btn.setEnabled(split_mode)

    def _update_preview_images(self) -> None:
        rel_path = self._selected_asset_relpath()
        if not rel_path:
            self.viewer.set_images(None, None)
            return

        input_path = os.path.join(self.config.input_dir, rel_path)
        output_path = self.map_combo.currentData()

        has_input = os.path.exists(input_path)
        has_output = bool(output_path and os.path.exists(output_path))

        self.before_path_label.setText(f"Input: {input_path if has_input else '-'}")
        self.after_path_label.setText(f"Output: {output_path if output_path else '-'}")

        self._request_preview_images(
            input_path if has_input else "",
            output_path if has_output else "",
            Path(input_path).name if has_input else "",
            Path(output_path).name if has_output else "",
        )

    @staticmethod
    def _estimate_qimage_bytes(image: Optional[QImage]) -> int:
        if image is None or image.isNull():
            return 0
        return int(image.bytesPerLine() * image.height())

    def _get_cached_image(self, path: str) -> Optional[QImage]:
        if not path:
            return None
        cached = self._image_cache.get(path)
        if cached is not None:
            self._image_cache.move_to_end(path)
            return cached[0]
        return None

    def _cache_image(self, path: str, image: Optional[QImage]) -> None:
        if not path or image is None or image.isNull():
            return None
        image_size = self._estimate_qimage_bytes(image)
        existing = self._image_cache.pop(path, None)
        if existing is not None:
            self._image_cache_bytes = max(0, self._image_cache_bytes - int(existing[1]))

        while (
            self._image_cache
            and (self._image_cache_bytes + image_size) > self._image_cache_limit_bytes
        ):
            _old_path, (_old_img, old_size) = self._image_cache.popitem(last=False)
            self._image_cache_bytes = max(0, self._image_cache_bytes - int(old_size))

        self._image_cache[path] = (image, image_size)
        self._image_cache_bytes += image_size

    def _request_preview_images(
        self,
        before_path: str,
        after_path: str,
        before_label: str,
        after_label: str,
    ) -> None:
        self._latest_image_request_token += 1
        token = self._latest_image_request_token

        if self._image_loader_thread is not None and self._image_loader_thread.isRunning():
            self._pending_image_request = (
                token,
                before_path,
                after_path,
                before_label,
                after_label,
            )
            return

        self._start_preview_image_loader(
            token,
            before_path,
            after_path,
            before_label,
            after_label,
        )

    def _start_preview_image_loader(
        self,
        token: int,
        before_path: str,
        after_path: str,
        before_label: str,
        after_label: str,
    ) -> None:
        before_cached = self._get_cached_image(before_path)
        after_cached = self._get_cached_image(after_path)
        self.viewer.set_images(
            before_cached,
            after_cached,
            before_label=before_label if before_cached is not None else "",
            after_label=after_label if after_cached is not None else "",
        )

        before_needs_load = bool(before_path and before_cached is None)
        after_needs_load = bool(after_path and after_cached is None)
        if not before_needs_load and not after_needs_load:
            return

        self._active_image_request_token = token
        self._active_preview_labels = (before_label, after_label)
        self._image_loader_thread = QThread(self)
        self._image_loader_worker = PreviewImageLoadWorker(token, before_path, after_path)
        self._image_loader_worker.moveToThread(self._image_loader_thread)

        self._image_loader_thread.started.connect(self._image_loader_worker.run)
        self._image_loader_worker.finished.connect(self._on_preview_images_loaded)
        self._image_loader_worker.finished.connect(self._image_loader_thread.quit)
        self._image_loader_thread.finished.connect(self._cleanup_image_loader)
        self._image_loader_thread.finished.connect(self._image_loader_worker.deleteLater)
        self._image_loader_thread.finished.connect(self._image_loader_thread.deleteLater)
        self._image_loader_thread.start()

    @pyqtSlot(object)
    def _on_preview_images_loaded(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        token = int(payload.get("token", 0))
        before_path = str(payload.get("before_path") or "")
        after_path = str(payload.get("after_path") or "")

        for path, err in payload.get("errors", []):
            self._append_log(f"[preview] Failed to load image: {path} ({err})")

        if token != self._latest_image_request_token:
            return

        before_img = payload.get("before_img")
        after_img = payload.get("after_img")

        if isinstance(before_img, QImage):
            self._cache_image(before_path, before_img)
        else:
            before_img = self._get_cached_image(before_path)

        if isinstance(after_img, QImage):
            self._cache_image(after_path, after_img)
        else:
            after_img = self._get_cached_image(after_path)

        before_label, after_label = self._active_preview_labels
        self.viewer.set_images(
            before_img,
            after_img,
            before_label=before_label if before_img is not None else "",
            after_label=after_label if after_img is not None else "",
        )

    def _cleanup_image_loader(self) -> None:
        self._image_loader_worker = None
        self._image_loader_thread = None

        pending = self._pending_image_request
        self._pending_image_request = None
        if pending is not None:
            token, before_path, after_path, before_label, after_label = pending
            self._start_preview_image_loader(
                token,
                before_path,
                after_path,
                before_label,
                after_label,
            )

    def _on_split_changed(self, value: int) -> None:
        self.viewer.set_split_ratio(value / 100.0)

    def _on_swap_split_toggled(self, checked: bool) -> None:
        self.viewer.set_split_swapped(checked)
        self.swap_split_btn.setText("Swap L<>R: On" if checked else "Swap L<>R: Off")

    def _on_zoom_slider_changed(self, value: int) -> None:
        self.zoom_value_label.setText(f"{value}%")
        self.viewer.set_zoom(value / 100.0)

    def _on_viewer_zoom_changed(self, zoom: float) -> None:
        pct = int(round(zoom * 100))
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(
            max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), pct))
        )
        self.zoom_slider.blockSignals(False)
        self.zoom_value_label.setText(f"{pct}%")

    def _on_viewer_split_changed(self, ratio: float) -> None:
        pct = int(round(ratio * 100))
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(
            max(self.split_slider.minimum(), min(self.split_slider.maximum(), pct))
        )
        self.split_slider.blockSignals(False)

    def _set_phase_selection(self, mode: str) -> None:
        mode = (mode or "").strip().lower()
        target: Dict[str, bool] = {}
        if mode == "all":
            target = {name: True for name in PHASE_OPTIONS}
        elif mode == "none":
            target = {name: False for name in PHASE_OPTIONS}
        elif mode == "preview_fast":
            target = {name: (name in PREVIEW_FAST_PHASES) for name in PHASE_OPTIONS}
        else:
            return

        for name, checkbox in self.phase_checks.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(target.get(name, True)))
            checkbox.blockSignals(False)
        self._schedule_auto_preview(f"phase_mode:{mode}")

    def _set_map_visibility(self, mode: str) -> None:
        mode = (mode or "").strip().lower()
        self.map_toggle_list.blockSignals(True)
        try:
            for idx in range(self.map_toggle_list.count()):
                item = self.map_toggle_list.item(idx)
                if mode == "all":
                    item.setCheckState(Qt.CheckState.Checked)
                elif mode == "none":
                    item.setCheckState(Qt.CheckState.Unchecked)
                elif mode == "core":
                    checked = item.text() in CORE_MAP_LABELS
                    item.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
        finally:
            self.map_toggle_list.blockSignals(False)

        self._refresh_map_combo()
        self._schedule_auto_preview(f"map_visibility:{mode}")

    def _selected_phases(self) -> Optional[List[str]]:
        selected = [name for name, cb in self.phase_checks.items() if cb.isChecked()]
        return selected if selected else None

    def _selected_map_suffixes(self) -> List[str]:
        suffix_by_label = {label: suffix for label, suffix in MAP_OPTIONS}
        selected: List[str] = []
        for idx in range(self.map_toggle_list.count()):
            item = self.map_toggle_list.item(idx)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            label = item.text()
            if label in suffix_by_label:
                selected.append(suffix_by_label[label])
        return selected

    def _on_always_preview_toggled(self, enabled: bool) -> None:
        self.always_preview_toggle.setText(
            "Always Preview: On" if enabled else "Always Preview: Off"
        )
        if enabled:
            self._schedule_auto_preview("always_preview_enabled")
        else:
            self._auto_preview_timer.stop()

    def _schedule_auto_preview(self, reason: str) -> None:
        self._refresh_config_health()
        if not self.always_preview_toggle.isChecked():
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        self._preview_request_reason = reason
        self._auto_preview_timer.start()

    def _trigger_auto_preview(self) -> None:
        if not self.always_preview_toggle.isChecked():
            return
        self._trigger_preview_render(f"auto:{self._preview_request_reason or 'unknown'}")

    def _on_refresh_preview_images(self) -> None:
        rel_path = self._selected_asset_relpath()
        if not rel_path:
            QMessageBox.information(
                self,
                "Preview",
                "Select an asset first to refresh preview.",
            )
            return

        input_dir = self.input_dir_edit.text().strip() or self.config.input_dir
        output_dir = self.output_dir_edit.text().strip() or self.config.output_dir
        dropped_cache = self._invalidate_cached_asset_images(input_dir, output_dir, rel_path)
        if dropped_cache:
            self._append_log(
                f"[preview] Refreshed cache for {rel_path}: dropped_cache={dropped_cache}"
            )
        self._refresh_preview_for_selected_asset()
        self.statusBar().showMessage("Preview images refreshed")

    def _on_stop_preview(self) -> None:
        if self._preview_thread is None or not self._preview_thread.isRunning():
            return
        if self._preview_worker is None:
            return
        self.stop_preview_btn.setEnabled(False)
        self._pending_preview_request = False
        self._preview_request_reason = ""
        self._auto_preview_timer.stop()
        self._append_log("[preview] Stop requested by user; waiting for cancellation...")
        self.statusBar().showMessage("Stopping preview render...")
        self._cancel_preview.emit()

    def _set_preview_running_state(self, running: bool) -> None:
        self.render_preview_btn.setEnabled(not running)
        self.refresh_preview_btn.setEnabled(not running)
        self.stop_preview_btn.setEnabled(running)
        if running:
            self.statusBar().showMessage("Rendering preview...")

    def _trigger_preview_render(self, reason: str) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._append_log("[preview] Main pipeline is running; preview render skipped.")
            return

        rel_path = self._selected_asset_relpath()
        if not rel_path:
            if reason.startswith("manual"):
                QMessageBox.information(
                    self,
                    "Preview",
                    "Select an asset first to render preview.",
                )
            return

        if self._preview_thread is not None and self._preview_thread.isRunning():
            self._pending_preview_request = True
            self._preview_request_reason = reason
            return

        try:
            cfg = self._collect_runtime_config(commit=False)
        except Exception as exc:
            if reason.startswith("manual"):
                QMessageBox.critical(self, "Preview Config Error", str(exc))
            else:
                self._append_log(f"[preview] Config invalid; skipped: {exc}")
            return

        source_path = os.path.join(cfg.input_dir, rel_path)
        if not os.path.isfile(source_path):
            self._append_log(f"[preview] Selected asset not found: {source_path}")
            return

        dropped_cache = self._invalidate_cached_asset_images(
            cfg.input_dir, cfg.output_dir, rel_path
        )
        self.latest_results.pop(rel_path, None)
        if dropped_cache:
            self._append_log(
                f"[preview] Cleared stale image cache for {rel_path}: "
                f"dropped_cache={dropped_cache}"
            )

        phases = self._selected_phases()
        self._preview_fail_dialog = reason.startswith("manual")
        self._set_preview_running_state(True)
        self._append_log(
            f"[preview] Rendering ({reason}) asset={rel_path} "
            f"phases={','.join(phases) if phases else 'all'}"
        )

        self._preview_thread = QThread(self)
        self._preview_worker = PreviewRenderWorker(cfg, phases, rel_path)
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.log_line.connect(lambda line: self._append_log(f"[preview] {line}"))
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._cancel_preview.connect(
            self._preview_worker.request_cancel, Qt.ConnectionType.DirectConnection
        )
        self._preview_thread.finished.connect(self._cleanup_preview_worker)
        self._preview_thread.finished.connect(self._preview_worker.deleteLater)
        self._preview_thread.finished.connect(self._preview_thread.deleteLater)
        self._preview_thread.start()

    def _on_preview_finished(self, payload: Dict[str, Any]) -> None:
        failed_count = int(payload.get("failed_count", 0))
        results = payload.get("results", {})
        if isinstance(results, dict):
            self.latest_results.update(results)

        records = payload.get("records", [])
        if records:
            by_path = {r.get("filepath"): r for r in self.accepted_records}
            for rec in records:
                rec = dict(rec)
                rec["accepted"] = True
                by_path[rec.get("filepath")] = rec
            self.accepted_records = [r for _, r in sorted(by_path.items()) if r]
            self.all_input_records = self._build_all_input_records(
                self.config.input_dir,
                self.accepted_records,
            )
            self._on_asset_source_changed(self.asset_source_combo.currentIndex())

        self._append_log(f"[preview] Finished. Failed assets: {failed_count}")
        self.statusBar().showMessage(
            f"Preview render complete (failed={failed_count})"
        )
        self._refresh_preview_for_selected_asset()

    def _on_preview_failed(self, details: str) -> None:
        if "cancelled" in details.lower():
            self._append_log("[preview] Cancelled by user.")
            self.statusBar().showMessage("Preview render cancelled")
            return
        self._append_log("[preview] Failed. See details in log.")
        self._append_log(details)
        self.statusBar().showMessage("Preview render failed")
        if getattr(self, "_preview_fail_dialog", False):
            QMessageBox.critical(self, "Preview Error", details)

    def _cleanup_preview_worker(self) -> None:
        try:
            self._cancel_preview.disconnect()
        except TypeError:
            pass
        self._preview_worker = None
        self._preview_thread = None
        self._set_preview_running_state(False)

        if self._pending_preview_request and self.always_preview_toggle.isChecked():
            self._pending_preview_request = False
            reason = self._preview_request_reason or "queued"
            self._preview_request_reason = ""
            self._trigger_preview_render(f"queued:{reason}")

    def _set_running_state(self, running: bool) -> None:
        scan_running = self._scan_thread is not None and self._scan_thread.isRunning()
        preview_running = self._preview_thread is not None and self._preview_thread.isRunning()
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.scan_btn.setEnabled((not running) and (not scan_running))
        self.apply_config_btn.setEnabled(not running)
        self.save_config_btn.setEnabled(not running)
        if running:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
        else:
            self._update_undo_redo_buttons()
        self.asset_source_combo.setEnabled(not running)
        self.select_all_assets_btn.setEnabled(not running)
        self.unselect_all_assets_btn.setEnabled(not running)
        self.render_preview_btn.setEnabled(not running)
        self.refresh_preview_btn.setEnabled(not running)
        self.stop_preview_btn.setEnabled((not running) and preview_running)
        self.always_preview_toggle.setEnabled(not running)
        if running:
            self._progress_phase = ""
            self._progress_done = 0
            self._progress_total = 0
            self.pipeline_progress.setRange(0, 0)
            self.pipeline_progress_label.setText("Progress: starting...")
            self.pipeline_overall_progress.setRange(0, 100)
            self.pipeline_overall_progress.setValue(0)
            total_phases = max(len(self._run_phase_sequence), 1)
            self.pipeline_overall_progress_label.setText(
                f"Overall: 0% (0/{total_phases} phases)"
            )
        elif self.pipeline_progress.maximum() == 0:
            self.pipeline_progress.setRange(0, 100)
            self.pipeline_progress.setValue(0)
            self.pipeline_progress_label.setText("Progress: idle")
            self.pipeline_overall_progress.setValue(0)
            self.pipeline_overall_progress_label.setText("Overall: idle")
        self.statusBar().showMessage("Running pipeline..." if running else "Ready")

    def _on_stop_pipeline(self) -> None:
        if self._worker_thread is None or not self._worker_thread.isRunning():
            return
        if self._worker is None:
            return
        self.stop_btn.setEnabled(False)
        self._append_log("Stop requested by user; waiting for current phase to cancel...")
        self.statusBar().showMessage("Stopping pipeline...")
        self._cancel_worker.emit()

    def _on_run_pipeline(self) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            QMessageBox.information(
                self,
                "Scan Running",
                "Asset scan is in progress. Wait for it to finish, then run pipeline.",
            )
            return
        if self._preview_thread is not None and self._preview_thread.isRunning():
            QMessageBox.information(
                self,
                "Preview Running",
                "Preview render is in progress. Wait for it to finish, then run pipeline.",
            )
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            QMessageBox.information(self, "Pipeline", "Pipeline is already running.")
            return

        try:
            cfg = self._collect_runtime_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config Error", str(exc))
            return

        if not os.path.isdir(cfg.input_dir):
            QMessageBox.warning(self, "Input", f"Input directory not found:\n{cfg.input_dir}")
            return

        phases = self._selected_phases()
        selected_map_suffixes = self._selected_map_suffixes()
        if not selected_map_suffixes:
            self._run_phase_sequence = []
            QMessageBox.information(
                self,
                "No Maps Selected",
                "No maps are checked. Select at least one map to run pipeline outputs.",
            )
            return
        selected_relpaths = self._checked_asset_paths() if self.records else []
        if self.records and not selected_relpaths:
            self._run_phase_sequence = []
            QMessageBox.information(
                self,
                "No Assets Selected",
                "No listed assets are checked. Select at least one asset to run.",
            )
            return
        reset_checkpoint = self.reset_checkpoint_check.isChecked()
        if reset_checkpoint:
            answer = QMessageBox.question(
                self,
                "Reset checkpoint",
                "This will clear checkpoint progress before running. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._run_phase_sequence = []
                return
        self._run_phase_sequence = [
            name
            for name in PHASE_EXECUTION_ORDER
            if phases is None or name in phases
        ]
        self._append_log(
            "Launching pipeline. "
            f"Phases: {', '.join(phases) if phases else 'all'} | dry_run={cfg.dry_run} | "
            f"assets={'all scanned' if not selected_relpaths else len(selected_relpaths)} | "
            f"maps={len(selected_map_suffixes)}"
        )

        self._set_running_state(True)
        self._worker_thread = QThread(self)
        self._worker = PipelineWorker(
            cfg,
            phases,
            reset_checkpoint,
            selected_relpaths=selected_relpaths or None,
            selected_map_suffixes=selected_map_suffixes,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.failed.connect(self._on_pipeline_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._cancel_worker.connect(
            self._worker.request_cancel, Qt.ConnectionType.DirectConnection
        )
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _cleanup_worker(self) -> None:
        try:
            self._cancel_worker.disconnect()
        except TypeError:
            pass
        self._worker = None
        self._worker_thread = None
        self._run_phase_sequence = []
        self._set_running_state(False)

    def _on_pipeline_finished(
        self,
        results: Dict[str, Any],
        records: List[Dict[str, Any]],
        failed_count: int,
    ) -> None:
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(100)
        self.pipeline_progress_label.setText("Progress: complete")
        self.pipeline_overall_progress.setRange(0, 100)
        self.pipeline_overall_progress.setValue(100)
        self.pipeline_overall_progress_label.setText("Overall: complete")
        self._image_cache.clear()
        self._image_cache_bytes = 0
        self.latest_results = results or {}
        if records:
            updated_records = [dict(r) for r in records]
            updated_by_path: Dict[str, Dict[str, Any]] = {}
            for rec in updated_records:
                rel_path = str(rec.get("filepath") or "")
                if not rel_path:
                    continue
                rec["accepted"] = True
                updated_by_path[rel_path] = rec

            if self.accepted_records:
                merged: List[Dict[str, Any]] = []
                seen: set[str] = set()
                for rec in self.accepted_records:
                    rel_path = str(rec.get("filepath") or "")
                    if rel_path and rel_path in updated_by_path:
                        merged.append(dict(updated_by_path[rel_path]))
                        seen.add(rel_path)
                    else:
                        kept = dict(rec)
                        kept["accepted"] = True
                        merged.append(kept)
                        if rel_path:
                            seen.add(rel_path)
                for rel_path, rec in updated_by_path.items():
                    if rel_path not in seen:
                        merged.append(dict(rec))
                self.accepted_records = merged
            else:
                self.accepted_records = list(updated_by_path.values())

            self.all_input_records = self._build_all_input_records(
                self.config.input_dir,
                self.accepted_records,
            )
            self._on_asset_source_changed(self.asset_source_combo.currentIndex())
        self._append_log(
            f"Pipeline finished. Failed assets: {failed_count}. "
            f"Results entries: {len(self.latest_results)}"
        )
        self.statusBar().showMessage(
            f"Finished. Failed assets: {failed_count} / {max(1, len(self.records))}"
        )
        self._refresh_preview_for_selected_asset()

    def _on_pipeline_failed(self, details: str) -> None:
        if "cancelled" in details.lower():
            self._append_log("Pipeline cancelled by user.")
            self.statusBar().showMessage("Pipeline cancelled")
            self.pipeline_progress.setRange(0, 100)
            self.pipeline_progress.setValue(0)
            self.pipeline_progress_label.setText("Progress: cancelled")
            self.pipeline_overall_progress_label.setText("Overall: cancelled")
            return
        self._append_log("Pipeline failed. See error dialog for details.")
        self.statusBar().showMessage("Pipeline failed")
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(0)
        self.pipeline_progress_label.setText("Progress: failed")
        self.pipeline_overall_progress_label.setText("Overall: failed")
        QMessageBox.critical(self, "Pipeline Error", details)

    @pyqtSlot(str, int, int)
    def _on_worker_progress(self, phase: str, done: int, total: int) -> None:
        total_i = max(int(total), 1)
        done_i = max(0, min(int(done), total_i))
        self._progress_phase = phase
        self._progress_done = done_i
        self._progress_total = total_i
        self.pipeline_progress.setRange(0, total_i)
        self.pipeline_progress.setValue(done_i)
        pct = int(round((done_i / total_i) * 100))
        self.pipeline_progress_label.setText(
            f"Progress: {phase} {done_i}/{total_i} ({pct}%)"
        )

        phase_seq = self._run_phase_sequence or list(PHASE_EXECUTION_ORDER)
        if phase in phase_seq:
            phase_idx = phase_seq.index(phase)
            phase_fraction = done_i / total_i
            overall = ((phase_idx + phase_fraction) / max(len(phase_seq), 1)) * 100.0
            overall_pct = int(round(overall))
            self.pipeline_overall_progress.setRange(0, 100)
            self.pipeline_overall_progress.setValue(max(0, min(overall_pct, 100)))
            completed_phases = phase_idx + (1 if done_i >= total_i else 0)
            self.pipeline_overall_progress_label.setText(
                f"Overall: {overall_pct}% ({completed_phases}/{len(phase_seq)} phases)"
            )

    @staticmethod
    def _extract_log_level(line: str) -> str:
        match = re.search(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]", line)
        if match:
            return match.group(1)
        return "OTHER"

    def _line_matches_log_filters(self, level: str, line: str) -> bool:
        if level not in self._active_log_levels:
            return False
        needle = (
            self.log_search_edit.text().strip().lower()
            if hasattr(self, "log_search_edit") else ""
        )
        if needle and needle not in line.lower():
            return False
        return True

    def _rebuild_log_view(self) -> None:
        visible_lines = [
            msg for level, msg in self._log_entries
            if self._line_matches_log_filters(level, msg)
        ]
        self.log_text.setPlainText("\n".join(visible_lines))
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_log_level_toggled(self, level: str, checked: bool) -> None:
        if checked:
            self._active_log_levels.add(level)
        else:
            self._active_log_levels.discard(level)
        self._rebuild_log_view()

    def _on_log_filter_show_all(self) -> None:
        self._active_log_levels = set(LOG_LEVEL_FILTERS)
        for level, button in self._log_filter_buttons.items():
            button.blockSignals(True)
            button.setChecked(level in self._active_log_levels)
            button.blockSignals(False)
        self._rebuild_log_view()

    def _on_log_search_changed(self, _text: str) -> None:
        self._rebuild_log_view()

    def _on_clear_log_search(self) -> None:
        self.log_search_edit.clear()

    def _append_log(self, line: str) -> None:
        normalized = line.rstrip()
        level = self._extract_log_level(normalized)
        self._log_entries.append((level, normalized))
        if len(self._log_entries) > self._max_log_entries:
            # Trim in larger chunks to avoid O(n) rebuild per line after overflow.
            overflow = len(self._log_entries) - self._max_log_entries
            trim_count = max(overflow, self._max_log_entries // 10)
            trim_count = min(trim_count, len(self._log_entries) - 1)
            del self._log_entries[:trim_count]
            self._rebuild_log_view()
        elif self._line_matches_log_filters(level, normalized):
            self.log_text.appendPlainText(normalized)
            bar = self.log_text.verticalScrollBar()
            bar.setValue(bar.maximum())

        if not normalized.startswith("[preview]"):
            match = re.search(
                r"\[progress\]\s+phase=(\w+)\s+done=(\d+)\s+total=(\d+)",
                normalized,
            )
            if match:
                phase = match.group(1)
                done = int(match.group(2))
                total = max(int(match.group(3)), 1)
                self._progress_phase = phase
                self._progress_done = max(done, 0)
                self._progress_total = total
                self.pipeline_progress.setRange(0, total)
                self.pipeline_progress.setValue(min(done, total))
                self.pipeline_progress_label.setText(
                    f"Progress: {phase} {done}/{total}"
                )

    def _clear_logs(self) -> None:
        self._log_entries.clear()
        self.log_text.clear()

    def closeEvent(self, event) -> None:  # noqa: ANN001
        """Handle window close with confirmation for active background tasks."""
        logger = logging.getLogger(__name__)
        # Collect all running threads/workers for a single prompt.
        running_threads = []
        cancel_signals = {
            "Pipeline": self._cancel_worker,
            "Preview render": self._cancel_preview,
        }
        worker_pairs = [
            (self._worker_thread, getattr(self, "_worker", None), "Pipeline"),
            (self._preview_thread, getattr(self, "_preview_worker", None), "Preview render"),
            (self._scan_thread, getattr(self, "_scan_worker", None), "Asset scan"),
            (self._image_loader_thread, None, "Image loader"),
        ]
        for thread, _worker, label in worker_pairs:
            if thread is not None and thread.isRunning():
                running_threads.append((thread, _worker, label))

        if running_threads:
            labels = ", ".join(label for _, _, label in running_threads)
            answer = QMessageBox.question(
                self,
                "Exit",
                f"Background tasks still running: {labels}. Exit anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            # Request cancellation via signals for workers that support it.
            for _thread, _worker, label in running_threads:
                sig = cancel_signals.get(label)
                if sig is not None:
                    sig.emit()

            # Disconnect worker signals to prevent post-close delivery.
            for _thread, worker, _label in running_threads:
                if worker is not None:
                    try:
                        worker.disconnect()
                    except TypeError:
                        pass

            # Graceful shutdown with forced termination fallback.
            for thread, _worker, label in running_threads:
                thread.quit()
                if not thread.wait(3000):
                    logger.warning("Thread '%s' did not stop in 3s, terminating.", label)
                    thread.terminate()
                    thread.wait(1000)

        self._auto_preview_timer.stop()
        self._save_ui_settings()
        super().closeEvent(event)


def main() -> None:
    """Launch the PyQt desktop application entrypoint."""
    def _qt_excepthook(exc_type, exc, tb):  # noqa: ANN001
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            logging.getLogger("asset_pipeline.ui").error(details)
        except Exception:
            pass
        print(details, file=sys.stderr)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Unhandled Error", str(exc))

    sys.excepthook = _qt_excepthook
    app = QApplication(sys.argv)
    app.setApplicationName("AssetBrew")
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
