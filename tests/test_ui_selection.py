"""UI selection behavior tests for asset table and worker forwarding."""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_PYQT6 = importlib.util.find_spec("PyQt6") is not None

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

    from AssetBrew.config import PipelineConfig
    from AssetBrew.ui.app import MAP_OPTIONS, MainWindow, PipelineWorker


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not installed")
class TestUISelectionBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _accepted_record(rel_path: str) -> dict:
        return {
            "filepath": rel_path,
            "filename": os.path.basename(rel_path),
            "texture_type": "diffuse",
            "original_width": 64,
            "original_height": 64,
            "channels": 3,
            "has_alpha": False,
            "is_tileable": False,
            "is_hero": False,
            "material_category": "default",
            "file_size_kb": 1.0,
            "is_gloss": False,
        }

    @staticmethod
    def _mk_workspace_tmpdir() -> str:
        root = os.path.join(os.getcwd(), "temp_test")
        os.makedirs(root, exist_ok=True)
        return tempfile.mkdtemp(dir=root)

    def test_scan_defaults_to_all_checked_and_toggle_works(self):
        with mock.patch.object(PipelineConfig, "resolve_device", return_value="cpu"):
            window = MainWindow()
        try:
            window._scan_input_dir = ""
            with mock.patch.object(window, "_build_all_input_records", return_value=[]):
                window._on_scan_finished(
                    [
                        self._accepted_record("a.png"),
                        self._accepted_record("b.dds"),
                    ]
                )

            self.assertEqual(set(window._checked_asset_paths()), {"a.png", "b.dds"})

            window._set_all_listed_asset_checks(False)
            self.assertEqual(window._checked_asset_paths(), [])

            window._toggle_all_listed_asset_checks()
            self.assertEqual(set(window._checked_asset_paths()), {"a.png", "b.dds"})

            window._toggle_all_listed_asset_checks()
            self.assertEqual(window._checked_asset_paths(), [])
        finally:
            window.close()

    def test_pipeline_worker_forwards_selected_assets(self):
        cfg = PipelineConfig()
        out_dir = self._mk_workspace_tmpdir()
        try:
            cfg.output_dir = out_dir
            cfg.input_dir = out_dir
            worker = PipelineWorker(
                cfg,
                phases=["upscale"],
                reset_checkpoint=False,
                selected_relpaths=["a.png"],
                selected_map_suffixes=["", "_normal"],
            )
            with mock.patch("AssetBrew.ui.app.setup_logging"):
                with mock.patch("AssetBrew.ui.app.AssetPipeline") as pipeline_cls:
                    pipeline = pipeline_cls.return_value
                    pipeline.run.return_value = None
                    pipeline.records = []
                    pipeline.results = {}
                    pipeline._failed_assets = 0

                    worker.run()

            pipeline.run.assert_called_once_with(
                ["upscale"],
                selected_assets=["a.png"],
                selected_maps=["", "_normal"],
            )
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_selected_map_suffixes_follow_checkbox_state(self):
        with mock.patch.object(PipelineConfig, "resolve_device", return_value="cpu"):
            window = MainWindow()
        try:
            window._set_map_visibility("all")
            all_suffixes = [suffix for _label, suffix in MAP_OPTIONS]
            self.assertEqual(window._selected_map_suffixes(), all_suffixes)

            window._set_map_visibility("core")
            core_suffixes = {
                "",
                "_albedo",
                "_normal",
                "_roughness",
                "_orm",
            }
            self.assertEqual(set(window._selected_map_suffixes()), core_suffixes)
        finally:
            window.close()

    def test_overall_progress_tracks_across_phases(self):
        with mock.patch.object(PipelineConfig, "resolve_device", return_value="cpu"):
            window = MainWindow()
        try:
            window._run_phase_sequence = ["upscale", "pbr", "normal"]
            window._set_running_state(True)

            window._on_worker_progress("upscale", 1, 2)
            self.assertEqual(window.pipeline_overall_progress.value(), 17)
            self.assertIn("Overall: 17%", window.pipeline_overall_progress_label.text())

            window._on_worker_progress("pbr", 3, 3)
            self.assertEqual(window.pipeline_overall_progress.value(), 67)
            self.assertIn("Overall: 67%", window.pipeline_overall_progress_label.text())

            window._on_pipeline_finished({}, [], 0)
            self.assertEqual(window.pipeline_overall_progress.value(), 100)
            self.assertEqual(window.pipeline_overall_progress_label.text(), "Overall: complete")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
