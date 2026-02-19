"""Tests for pipeline orchestrator flow."""

import contextlib
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock
import importlib.util
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, save_image, get_output_path

HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_requires_cv2_scipy = unittest.skipUnless(HAS_CV2 and HAS_SCIPY, "cv2/scipy not installed")


class TestPhaseOrdering(unittest.TestCase):
    def test_phases_run_in_canonical_order(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.dry_run = True
        try:
            pipeline = AssetPipeline(config)
            execution_order = []
            all_phase_names = [
                "upscale", "pbr", "normal", "pom",
                "mipmap", "postprocess", "validate"]
            for name in all_phase_names:
                method_name = {
                    "upscale": "phase1_upscale", "pbr": "phase2_pbr",
                    "normal": "phase3_normals", "pom": "phase4_pom",
                    "mipmap": "phase6_mipmaps", "postprocess": "phase5_postprocess",
                    "validate": "phase7_validate",
                }[name]
                def make_recorder(n, orig):
                    def recorder():
                        execution_order.append(n)
                        return orig()
                    return recorder
                orig_fn = getattr(pipeline, method_name)
                setattr(pipeline, method_name, make_recorder(name, orig_fn))
            pipeline.run(["validate", "normal", "pbr"])
            self.assertEqual(execution_order, ["pbr", "normal", "validate"])
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestZeroAssetsWarning(unittest.TestCase):
    def test_empty_input_warns(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.dry_run = True
        try:
            pipeline = AssetPipeline(config)
            with self.assertLogs("asset_pipeline", level="WARNING") as cm:
                pipeline.run()
            found = any("No assets found" in msg for msg in cm.output)
            self.assertTrue(found)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestSelectedAssetFiltering(unittest.TestCase):
    @staticmethod
    def _make_record(rel_path: str) -> AssetRecord:
        name = os.path.basename(rel_path)
        return AssetRecord(
            filepath=rel_path,
            filename=name,
            texture_type="diffuse",
            original_width=64,
            original_height=64,
            channels=3,
            has_alpha=False,
            is_tileable=False,
            is_hero=False,
            material_category="default",
            file_size_kb=1.0,
            file_hash=f"hash-{name}",
            is_gloss=False,
        )

    def test_run_scan_filters_to_selected_assets(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.dry_run = True
        try:
            records = [
                self._make_record("a.png"),
                self._make_record("folder\\b.png"),
                self._make_record("c.png"),
            ]
            pipeline = AssetPipeline(config)

            with mock.patch("AssetBrew.pipeline.scan_assets", return_value=records):
                with mock.patch.object(pipeline.checkpoint, "prune") as prune_mock:
                    pipeline.run(["scan"], selected_assets=["a.png", "folder/b.png"])

            self.assertEqual(
                [os.path.normpath(r.filepath) for r in pipeline.records],
                [os.path.normpath("a.png"), os.path.normpath("folder/b.png")],
            )
            self.assertEqual(
                {os.path.normpath(path) for path in pipeline.results.keys()},
                {os.path.normpath("a.png"), os.path.normpath("folder/b.png")},
            )
            prune_mock.assert_called_once()
            self.assertEqual(
                {os.path.normpath(path) for path in prune_mock.call_args[0][0]},
                {
                    os.path.normpath("a.png"),
                    os.path.normpath("folder/b.png"),
                    os.path.normpath("c.png"),
                },
            )
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestSelectedMapFiltering(unittest.TestCase):
    def test_copy_to_output_respects_selected_maps(self):
        from AssetBrew.pipeline import AssetPipeline

        temp_root = os.path.join(os.getcwd(), ".tmp_local", f"map_filter_{uuid4().hex}")
        os.makedirs(temp_root, exist_ok=True)
        config = PipelineConfig()
        config.input_dir = os.path.join(temp_root, "input")
        config.output_dir = os.path.join(temp_root, "output")
        config.intermediate_dir = os.path.join(temp_root, "intermediate")
        config.comparison_dir = os.path.join(temp_root, "comparison")
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.intermediate_dir, exist_ok=True)
        os.makedirs(config.comparison_dir, exist_ok=True)
        config.compression.generate_dds = False
        config.compression.generate_ktx2 = False
        config.compression.generate_tga = False
        try:
            base_path = os.path.join(config.input_dir, "brick_diff.png")
            albedo_path = os.path.join(config.intermediate_dir, "brick_diff_albedo.png")
            normal_path = os.path.join(config.intermediate_dir, "brick_diff_normal.png")
            for path in (base_path, albedo_path, normal_path):
                with open(path, "wb") as f:
                    f.write(b"PNG")

            record = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc123",
                is_gloss=False,
            )
            pipeline = AssetPipeline(config)
            pipeline.records = [record]
            pipeline.results = {
                record.filepath: {
                    "pbr": {"albedo": albedo_path},
                    "normal": {"normal": normal_path},
                }
            }
            pipeline._selected_map_suffixes = {"", "_albedo"}

            pipeline.copy_to_output()

            out_base = get_output_path(record.filepath, config.output_dir, suffix="", ext=".png")
            out_albedo = get_output_path(
                record.filepath, config.output_dir, suffix="_albedo", ext=".png"
            )
            out_normal = get_output_path(
                record.filepath, config.output_dir, suffix="_normal", ext=".png"
            )
            self.assertTrue(os.path.exists(out_base))
            self.assertTrue(os.path.exists(out_albedo))
            self.assertFalse(os.path.exists(out_normal))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


class TestRunFinalizationOnFailure(unittest.TestCase):
    def test_run_still_persists_state_on_phase_exception(self):
        """After a fatal phase error, copy_to_output is skipped (to avoid
        copying partial/corrupt data), but _save_results and checkpoint.save
        still run to preserve pipeline state for debugging."""
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.upscale.enabled = False
        try:
            save_image(
                np.random.rand(32, 32, 3).astype(np.float32),
                os.path.join(config.input_dir, "brick_diff.png"),
            )
            pipeline = AssetPipeline(config)
            with mock.patch.object(
                pipeline, "phase2_pbr", side_effect=RuntimeError("phase boom")
            ):
                with mock.patch.object(pipeline, "copy_to_output") as copy_mock:
                    with mock.patch.object(pipeline, "_validate_outputs") as val_mock:
                        with mock.patch.object(pipeline, "_save_results") as save_mock:
                            with mock.patch.object(pipeline.checkpoint, "save") as ckpt_save:
                                with self.assertRaises(RuntimeError):
                                    pipeline.run()
            # Fatal errors skip copy_to_output to avoid copying corrupt data
            copy_mock.assert_not_called()
            val_mock.assert_not_called()
            # But results and checkpoint are always saved
            save_mock.assert_called_once()
            self.assertGreaterEqual(ckpt_save.call_count, 1)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_scan_only_does_not_run_copy_or_output_validation(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            save_image(
                np.random.rand(32, 32, 3).astype(np.float32),
                os.path.join(config.input_dir, "brick_diff.png"),
            )
            pipeline = AssetPipeline(config)
            with mock.patch.object(pipeline, "copy_to_output") as copy_mock:
                with mock.patch.object(pipeline, "_validate_outputs") as val_mock:
                    with mock.patch.object(pipeline, "_save_results") as save_mock:
                        with mock.patch.object(pipeline.checkpoint, "save"):
                            pipeline.run(["scan"])
            copy_mock.assert_not_called()
            val_mock.assert_not_called()
            save_mock.assert_not_called()
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_empty_phase_selection_is_not_scan_only(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            with contextlib.ExitStack() as stack:
                for phase_name in (
                    "phase0_scan",
                    "phase1_upscale",
                    "phase2_pbr",
                    "phase3_normals",
                    "phase4_pom",
                    "phase5_postprocess",
                    "phase6_mipmaps",
                    "phase7_validate",
                ):
                    stack.enter_context(mock.patch.object(pipeline, phase_name))
                copy_mock = stack.enter_context(
                    mock.patch.object(pipeline, "copy_to_output")
                )
                val_mock = stack.enter_context(
                    mock.patch.object(pipeline, "_validate_outputs")
                )
                save_mock = stack.enter_context(
                    mock.patch.object(pipeline, "_save_results")
                )
                stack.enter_context(mock.patch.object(pipeline.checkpoint, "save"))
                pipeline.run([])
            copy_mock.assert_called_once()
            val_mock.assert_called_once()
            save_mock.assert_called_once()
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_run_skips_copy_and_validation_after_timeout(self):
        from AssetBrew.pipeline import AssetPipeline, PhaseTimeoutError

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            with mock.patch.object(
                pipeline,
                "phase1_upscale",
                side_effect=PhaseTimeoutError("upscale timed out"),
            ):
                with mock.patch.object(pipeline, "copy_to_output") as copy_mock:
                    with mock.patch.object(pipeline, "_validate_outputs") as val_mock:
                        with mock.patch.object(pipeline, "_save_results") as save_mock:
                            with mock.patch.object(pipeline.checkpoint, "save"):
                                with self.assertRaises(PhaseTimeoutError):
                                    pipeline.run(["upscale"])
            copy_mock.assert_not_called()
            val_mock.assert_not_called()
            save_mock.assert_called_once()
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestCopyFailureTracking(unittest.TestCase):
    def test_copy_error_counted_as_failed_asset(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            img_path = os.path.join(config.input_dir, "brick_diff.png")
            img = np.random.rand(64, 64, 3).astype(np.float32)
            save_image(img, img_path)
            pipeline = AssetPipeline(config)
            record = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=64,
                original_height=64,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=10.0,
                file_hash="abc",
                is_gloss=False,
            )
            pipeline.records = [record]
            pipeline.results = {record.filepath: {"upscale": {"upscaled": img_path}}}

            with mock.patch(
                "AssetBrew.pipeline.shutil.copyfile",
                side_effect=PermissionError("copy denied"),
            ):
                pipeline.copy_to_output()

            copy_err = pipeline.results[record.filepath].get("copy", {}).get("error", "")
            self.assertIn("copy denied", copy_err)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_copy_to_output_import_resolved_at_method_level(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            pipeline.copy_to_output()
            self.assertTrue(hasattr(pipeline, '_copy_load_image'))
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestSkippedPhasesExitCode(unittest.TestCase):
    def test_skipped_phases_no_exit_failure(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            save_image(
                np.random.rand(32, 32, 3).astype(np.float32),
                os.path.join(config.input_dir, "brick_diff.png"),
            )
            config.compression.generate_dds = False
            config.compression.generate_ktx2 = False
            config.compression.generate_tga = False
            pipeline = AssetPipeline(config)
            with mock.patch.object(pipeline, "_check_deps", return_value=["torch"]):
                pipeline.run(["upscale"])
            self.assertEqual(getattr(pipeline, "_failed_assets", 0), 0)
            upscale_result = pipeline.results["brick_diff.png"]["upscale"]
            self.assertTrue(upscale_result.get("skipped"))
            self.assertEqual(upscale_result.get("fallback"), "original_input")
            self.assertIn("torch", upscale_result.get("reason", ""))
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestFailureAccounting(unittest.TestCase):
    def test_validate_phase_passed_false_counts_as_failed_asset(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.validation.enabled = True
        try:
            pipeline = AssetPipeline(config)
            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=64,
                original_height=64,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            def _fake_scan():
                pipeline.records = [rec]
                pipeline.results = {rec.filepath: {}}

            def _fake_validate():
                pipeline.results[rec.filepath]["validate"] = {
                    "passed": False,
                    "errors": ["normal map invalid"],
                }

            with mock.patch.object(pipeline, "phase0_scan", side_effect=_fake_scan):
                with mock.patch.object(pipeline, "phase7_validate", side_effect=_fake_validate):
                    with mock.patch.object(pipeline, "copy_to_output"):
                        with mock.patch.object(pipeline, "_validate_outputs"):
                            with mock.patch.object(pipeline, "_save_results"):
                                with mock.patch.object(pipeline.checkpoint, "save"):
                                    pipeline.run(["validate"])

            self.assertEqual(getattr(pipeline, "_failed_assets", 0), 1)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestParallelTimeoutHandling(unittest.TestCase):
    def test_parallel_phase_timeout_aborts_phase_without_hanging(self):
        from AssetBrew.pipeline import AssetPipeline, PhaseTimeoutError

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.phase_timeout_seconds = 1
        try:
            pipeline = AssetPipeline(config)
            rec = AssetRecord(
                filepath="slow_diff.png",
                filename="slow_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )
            pipeline.results = {rec.filepath: {}}

            def _slow_process(_record):
                time.sleep(3.0)
                return {"ok": True}

            start = time.time()
            with self.assertRaises(PhaseTimeoutError):
                pipeline._run_parallel(
                    "pbr",
                    [rec],
                    _slow_process,
                    desc="Timeout test",
                    max_workers=2,
                )
            elapsed = time.time() - start

            err = pipeline.results[rec.filepath]["pbr"].get("error", "")
            self.assertIn("Timed out", err)
            # Use a generous limit to avoid flaky CI failures from
            # scheduling jitter; the point is that we don't wait the full
            # 3-second sleep, not that we finish in exactly N seconds.
            self.assertLess(elapsed, 10.0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_parallel_timeout_ignores_queue_time(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.phase_timeout_seconds = 1
        try:
            pipeline = AssetPipeline(config)
            records = []
            for idx in range(6):
                records.append(AssetRecord(
                    filepath=f"queue{idx}.png",
                    filename=f"queue{idx}.png",
                    texture_type="diffuse",
                    original_width=32,
                    original_height=32,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=1.0,
                    file_hash=f"q{idx}",
                    is_gloss=False,
                ))
            pipeline.results = {r.filepath: {} for r in records}

            def _short_task(_record):
                time.sleep(0.2)
                return {"ok": True}

            pipeline._run_parallel(
                "pbr",
                records,
                _short_task,
                desc="Queue timeout test",
                max_workers=1,
            )

            for rec in records:
                result = pipeline.results.get(rec.filepath, {}).get("pbr", {})
                self.assertFalse(result.get("error"))
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_parallel_timeout_cancels_pending_tasks(self):
        from AssetBrew.pipeline import AssetPipeline, PhaseTimeoutError

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.phase_timeout_seconds = 1
        try:
            pipeline = AssetPipeline(config)
            records = []
            for idx in range(3):
                records.append(AssetRecord(
                    filepath=f"slow{idx}.png",
                    filename=f"slow{idx}.png",
                    texture_type="diffuse",
                    original_width=32,
                    original_height=32,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=1.0,
                    file_hash=f"s{idx}",
                    is_gloss=False,
                ))
            pipeline.results = {r.filepath: {} for r in records}

            def _very_slow(_record):
                time.sleep(3.0)
                return {"ok": True}

            with self.assertRaises(PhaseTimeoutError):
                pipeline._run_parallel(
                    "pbr",
                    records,
                    _very_slow,
                    desc="Timeout cancel test",
                    max_workers=2,
                )

            errors = [
                pipeline.results.get(r.filepath, {}).get("pbr", {}).get("error", "")
                for r in records
            ]
            self.assertTrue(any("Timed out after" in err for err in errors))
            self.assertTrue(any("Cancelled after timeout in pbr" in err for err in errors))
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestCachedResultPathValidation(unittest.TestCase):
    def test_stale_cached_output_path_reprocesses_record(self):
        from AssetBrew.pipeline import AssetPipeline

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            record = AssetRecord(
                filepath="stale_diff.png",
                filename="stale_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="stale-hash",
                is_gloss=False,
            )
            pipeline.results = {record.filepath: {}}
            stale_cached_path = os.path.join(
                config.intermediate_dir,
                "missing",
                "stale_albedo.png",
            )
            process_calls = []

            def _process(rec):
                process_calls.append(rec.filepath)
                return {"ok": True}

            with mock.patch.object(pipeline, "_should_skip", return_value=True):
                with mock.patch.object(
                    pipeline.checkpoint,
                    "get_result",
                    return_value={"albedo": stale_cached_path},
                ):
                    with mock.patch.object(pipeline.checkpoint, "mark_completed") as mark_completed:
                        pipeline._run_parallel(
                            "pbr",
                            [record],
                            _process,
                            desc="stale cache test",
                            max_workers=1,
                        )

            self.assertEqual(process_calls, [record.filepath])
            self.assertEqual(pipeline.results[record.filepath]["pbr"], {"ok": True})
            mark_completed.assert_called_once()
        finally:
            for d in [
                config.input_dir,
                config.output_dir,
                config.intermediate_dir,
                config.comparison_dir,
            ]:
                shutil.rmtree(d, ignore_errors=True)


class TestPhaseFailureThreshold(unittest.TestCase):
    def test_sequential_phase_failure_threshold_aborts(self):
        from AssetBrew.pipeline import AssetPipeline, PhaseFailureThresholdError

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.phase_failure_abort_ratio = 0.5
        config.phase_failure_abort_min_processed = 2
        try:
            pipeline = AssetPipeline(config)
            records = []
            for idx in range(4):
                records.append(AssetRecord(
                    filepath=f"file{idx}.png",
                    filename=f"file{idx}.png",
                    texture_type="diffuse",
                    original_width=32,
                    original_height=32,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=1.0,
                    file_hash=f"h{idx}",
                    is_gloss=False,
                ))
            pipeline.results = {r.filepath: {} for r in records}

            def _fail(_record):
                return {"error": "boom"}

            with self.assertRaises(PhaseFailureThresholdError):
                pipeline._run_parallel(
                    "pbr",
                    records,
                    _fail,
                    desc="Failure threshold",
                    max_workers=1,
                )
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_parallel_phase_failure_threshold_cancels_pending(self):
        from AssetBrew.pipeline import AssetPipeline, PhaseFailureThresholdError

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.phase_failure_abort_ratio = 0.5
        config.phase_failure_abort_min_processed = 2
        try:
            pipeline = AssetPipeline(config)
            records = []
            for idx in range(6):
                records.append(AssetRecord(
                    filepath=f"file{idx}.png",
                    filename=f"file{idx}.png",
                    texture_type="diffuse",
                    original_width=32,
                    original_height=32,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=1.0,
                    file_hash=f"h{idx}",
                    is_gloss=False,
                ))
            pipeline.results = {r.filepath: {} for r in records}

            def _fail(record):
                if record.filename in {"file2.png", "file3.png", "file4.png", "file5.png"}:
                    time.sleep(0.4)
                return {"error": "boom"}

            with self.assertRaises(PhaseFailureThresholdError) as ctx:
                pipeline._run_parallel(
                    "pbr",
                    records,
                    _fail,
                    desc="Failure threshold",
                    max_workers=2,
                )
            self.assertIn("failure ratio", str(ctx.exception).lower())

            errors = []
            for r in records:
                err = pipeline.results.get(r.filepath, {}).get("pbr", {}).get("error", "")
                if err:
                    errors.append(err)
            self.assertTrue(errors)
            self.assertTrue(
                any("cancelled after failure threshold in pbr" in e.lower() for e in errors)
            )
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestPhaseOrderPostprocessBeforeMipmap(unittest.TestCase):
    def test_postprocess_before_mipmap_in_all_phases(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.dry_run = True
        try:
            pipeline = AssetPipeline(config)
            order = []

            def _wrap(name, fn):
                def _inner():
                    order.append(name)
                    return fn()
                return _inner

            pipeline.phase5_postprocess = _wrap("postprocess", pipeline.phase5_postprocess)
            pipeline.phase6_mipmaps = _wrap("mipmap", pipeline.phase6_mipmaps)
            pipeline.run(["mipmap", "postprocess"])
            self.assertEqual(order, ["postprocess", "mipmap"])
        finally:
            shutil.rmtree(config.input_dir, ignore_errors=True)
            shutil.rmtree(config.output_dir, ignore_errors=True)
            shutil.rmtree(config.intermediate_dir, ignore_errors=True)
            shutil.rmtree(config.comparison_dir, ignore_errors=True)

    def test_mipmap_uses_postprocessed_roughness(self):
        from AssetBrew.pipeline import AssetPipeline
        from AssetBrew.config import TextureType
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            fp = "test.png"
            main_path = os.path.join(config.input_dir, "main.png")
            aa_path = os.path.join(config.output_dir, "aa_rough.png")
            pbr_path = os.path.join(config.output_dir, "pbr_rough.png")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), main_path)
            save_image(np.random.rand(32, 32).astype(np.float32), aa_path)
            save_image(np.random.rand(32, 32).astype(np.float32), pbr_path)

            rec = AssetRecord(
                filepath=fp, filename=fp, texture_type="diffuse",
                original_width=32, original_height=32, channels=3,
                has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=1.0,
                file_hash="abc", is_gloss=False,
            )
            pipeline.records = [rec]
            pipeline.results = {
                fp: {
                    "upscale": {"upscaled": main_path},
                    "specular_aa": {"roughness_aa": aa_path},
                    "pbr": {"roughness": pbr_path},
                }
            }

            class _FakeMipmapGenerator:
                def __init__(self, cfg):
                    self.calls = []

                def process(self, record, source_path, tex_type_override=None, name_tag=None):
                    self.calls.append((record.filepath, source_path, tex_type_override, name_tag))
                    return {"mips": []}

            with mock.patch.object(pipeline, "_check_deps", return_value=[]):
                with mock.patch(
                    "AssetBrew.phases.mipmap.MipmapGenerator",
                    _FakeMipmapGenerator,
                ):
                    pipeline.phase6_mipmaps()

            calls = pipeline._mipmap_gen.calls
            rough_calls = [c for c in calls if c[2] == TextureType.ROUGHNESS]
            self.assertTrue(rough_calls, f"Expected roughness mip call, got: {calls}")
            self.assertEqual(rough_calls[0][1], aa_path)
        finally:
            shutil.rmtree(config.input_dir, ignore_errors=True)
            shutil.rmtree(config.output_dir, ignore_errors=True)
            shutil.rmtree(config.intermediate_dir, ignore_errors=True)
            shutil.rmtree(config.comparison_dir, ignore_errors=True)

    def test_mipmap_generates_all_core_data_maps(self):
        from AssetBrew.pipeline import AssetPipeline
        from AssetBrew.config import TextureType

        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        try:
            pipeline = AssetPipeline(config)
            fp = "test.png"
            paths = {}
            for key in (
                "main", "albedo", "roughness", "metalness",
                "ao", "normal", "height", "orm",
            ):
                p = os.path.join(config.output_dir, f"{key}.png")
                save_image(np.random.rand(32, 32, 3).astype(np.float32), p)
                paths[key] = p

            rec = AssetRecord(
                filepath=fp, filename=fp, texture_type="diffuse",
                original_width=32, original_height=32, channels=3,
                has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=1.0,
                file_hash="abc", is_gloss=False,
            )
            pipeline.records = [rec]
            pipeline.results = {
                fp: {
                    "upscale": {"upscaled": paths["main"]},
                    "color_consistency": {"corrected": paths["albedo"]},
                    "specular_aa": {"roughness_aa": paths["roughness"]},
                    "pbr": {
                        "roughness": paths["roughness"],
                        "metalness": paths["metalness"],
                        "ao": paths["ao"],
                        "albedo": paths["albedo"],
                    },
                    "normal": {"normal": paths["normal"], "height": paths["height"]},
                    "orm": {"orm": paths["orm"]},
                }
            }

            class _FakeMipmapGenerator:
                def __init__(self, cfg):
                    self.calls = []

                def process(self, record, source_path, tex_type_override=None, name_tag=None):
                    self.calls.append((record.filepath, source_path, tex_type_override, name_tag))
                    return {"mips": []}

            with mock.patch.object(pipeline, "_check_deps", return_value=[]):
                with mock.patch(
                    "AssetBrew.phases.mipmap.MipmapGenerator",
                    _FakeMipmapGenerator,
                ):
                    pipeline.phase6_mipmaps()

            calls = pipeline._mipmap_gen.calls
            seen = {c[3] for c in calls}
            expected = {
                "main", "albedo", "roughness", "metalness",
                "ao", "normal", "height", "orm",
            }
            self.assertTrue(expected.issubset(seen), f"Missing mip calls. Seen={seen}")

            by_tag = {c[3]: c[2] for c in calls if c[3] in expected}
            self.assertEqual(by_tag["albedo"], TextureType.ALBEDO)
            self.assertEqual(by_tag["metalness"], TextureType.METALNESS)
            self.assertEqual(by_tag["ao"], TextureType.AO)
            self.assertEqual(by_tag["orm"], TextureType.ORM)
        finally:
            shutil.rmtree(config.input_dir, ignore_errors=True)
            shutil.rmtree(config.output_dir, ignore_errors=True)
            shutil.rmtree(config.intermediate_dir, ignore_errors=True)
            shutil.rmtree(config.comparison_dir, ignore_errors=True)


class TestOutputValidation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_pipeline(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = os.path.join(self.tmpdir, "input")
        config.output_dir = self.output_dir
        config.intermediate_dir = os.path.join(self.tmpdir, "inter")
        os.makedirs(config.input_dir, exist_ok=True)
        os.makedirs(config.intermediate_dir, exist_ok=True)
        pipe = AssetPipeline(config)
        return pipe

    def _make_record(self, filename="brick_diff.png"):
        return AssetRecord(
            filepath=filename, filename=filename,
            texture_type="diffuse", original_width=128, original_height=128,
            channels=3, has_alpha=False, is_tileable=True, is_hero=False,
            material_category="brick", file_size_kb=50.0,
            file_hash="abc123", is_gloss=False,
        )

    def test_valid_png_passes(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        path = os.path.join(self.output_dir, "brick_diff.png")
        Image.fromarray(arr).save(path)
        asset_dims = {}
        issues = pipe._validate_single_output(path, "", record, asset_dims)
        self.assertEqual(issues, [])

    def test_zero_byte_detected(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.png")
        open(path, "wb").close()
        issues = pipe._validate_single_output(path, "", record, {})
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["check"], "non_zero_size")

    def test_bad_magic_detected(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.png")
        with open(path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        issues = pipe._validate_single_output(path, "", record, {})
        checks = [i["check"] for i in issues]
        self.assertIn("magic_bytes", checks)

    def test_non_power_of_two(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        path = os.path.join(self.output_dir, "brick_diff.png")
        Image.fromarray(arr).save(path)
        issues = pipe._validate_single_output(path, "", record, {})
        checks = [i["check"] for i in issues]
        self.assertIn("power_of_two", checks)

    def test_solid_color_detected(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        arr = np.full((128, 128, 3), 128, dtype=np.uint8)
        path = os.path.join(self.output_dir, "brick_diff.png")
        Image.fromarray(arr).save(path)
        issues = pipe._validate_single_output(path, "", record, {})
        checks = [i["check"] for i in issues]
        self.assertIn("solid_color", checks)

    def test_16bit_near_solid_detected(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        arr = np.full((128, 128), 32000, dtype=np.uint16)
        arr[::2, ::2] = 32100
        path = os.path.join(self.output_dir, "brick_diff.png")
        Image.fromarray(arr).save(path)
        issues = pipe._validate_single_output(path, "", record, {})
        checks = [i["check"] for i in issues]
        self.assertIn("solid_color", checks)

    def test_normal_good(self):
        pipe = self._make_pipeline()
        record = self._make_record("brick_diff_normal.png")
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[:, :, 0] = 128
        arr[:, :, 1] = 128
        arr[:, :, 2] = 255
        arr[0, 0, 0] = 130
        arr[1, 1, 1] = 126
        path = os.path.join(self.output_dir, "brick_diff_normal.png")
        Image.fromarray(arr).save(path)
        issues = pipe._validate_single_output(path, "_normal", record, {})
        normal_issues = [i for i in issues if i["check"] == "normal_integrity"]
        self.assertEqual(normal_issues, [])

    def test_normal_bad_z(self):
        pipe = self._make_pipeline()
        record = self._make_record("brick_diff_normal.png")
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[:, :, 0] = 128
        arr[:, :, 1] = 128
        arr[:, :, 2] = 50
        path = os.path.join(self.output_dir, "brick_diff_normal.png")
        Image.fromarray(arr).save(path)
        issues = pipe._validate_single_output(path, "_normal", record, {})
        checks = [i["check"] for i in issues]
        self.assertIn("normal_integrity", checks)

    def test_dimension_consistency(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        asset_dims = {"": (256, 256), "_normal": (128, 128)}
        issues = pipe._check_dimension_consistency(record, asset_dims)
        checks = [i["check"] for i in issues]
        self.assertIn("dimension_consistency", checks)

    def test_dds_magic_valid(self):
        import struct

        pipe = self._make_pipeline()
        pipe.config.validation.require_full_mipchain = False
        pipe.config.compression.format_map["diffuse"] = "bc1"
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)          # dwSize
        struct.pack_into("<I", header, 4, 0x1007)       # dwFlags
        struct.pack_into("<I", header, 8, 128)          # dwHeight
        struct.pack_into("<I", header, 12, 128)         # dwWidth
        struct.pack_into("<I", header, 24, 1)           # dwMipMapCount
        struct.pack_into("<I", header, 72, 32)          # ddspf.dwSize
        header[80:84] = b"DXT1"                         # ddspf.dwFourCC
        struct.pack_into("<I", header, 104, 0x1000)     # dwCaps
        with open(path, "wb") as f:
            f.write(b"DDS " + bytes(header))
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        self.assertEqual(issues, [])

    def test_dds_missing_required_flags_and_caps_detected(self):
        import struct

        pipe = self._make_pipeline()
        pipe.config.validation.require_full_mipchain = False
        pipe.config.compression.format_map["diffuse"] = "bc1"
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)          # dwSize
        struct.pack_into("<I", header, 4, 0x00000006)   # dwFlags (HEIGHT|WIDTH only)
        struct.pack_into("<I", header, 8, 128)          # dwHeight
        struct.pack_into("<I", header, 12, 128)         # dwWidth
        struct.pack_into("<I", header, 24, 1)           # dwMipMapCount
        struct.pack_into("<I", header, 72, 32)          # ddspf.dwSize
        header[80:84] = b"DXT1"                         # ddspf.dwFourCC
        struct.pack_into("<I", header, 104, 0x00000000) # dwCaps
        with open(path, "wb") as f:
            f.write(b"DDS " + bytes(header))
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        checks = {issue["check"] for issue in issues}
        self.assertIn("dds_flags", checks)
        self.assertIn("dds_caps_texture", checks)

    def test_dds_codec_mismatch_detected(self):
        import struct

        pipe = self._make_pipeline()
        pipe.config.validation.require_full_mipchain = False
        # Base diffuse expects BC7 by default, but file is authored as DXT1/BC1.
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)
        struct.pack_into("<I", header, 4, 0x1007)
        struct.pack_into("<I", header, 8, 128)
        struct.pack_into("<I", header, 12, 128)
        struct.pack_into("<I", header, 24, 1)
        struct.pack_into("<I", header, 72, 32)
        header[80:84] = b"DXT1"
        struct.pack_into("<I", header, 104, 0x1000)
        with open(path, "wb") as f:
            f.write(b"DDS " + bytes(header))
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        checks = [issue["check"] for issue in issues]
        self.assertIn("dds_codec", checks)

    def test_dds_mipchain_required_flags_single_mip(self):
        import struct

        pipe = self._make_pipeline()
        pipe.config.validation.require_full_mipchain = True
        pipe.config.mipmap.enabled = True
        pipe._executed_phases.add("mipmap")
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)          # dwSize
        struct.pack_into("<I", header, 4, 0x1007)       # dwFlags
        struct.pack_into("<I", header, 8, 128)          # dwHeight
        struct.pack_into("<I", header, 12, 128)         # dwWidth
        struct.pack_into("<I", header, 24, 1)           # dwMipMapCount
        struct.pack_into("<I", header, 72, 32)          # ddspf.dwSize
        header[80:84] = b"DXT1"                         # ddspf.dwFourCC
        struct.pack_into("<I", header, 104, 0x1000)     # dwCaps
        with open(path, "wb") as f:
            f.write(b"DDS " + bytes(header))
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        checks = [issue["check"] for issue in issues]
        self.assertIn("dds_mipchain", checks)

    def test_dds_mipchain_not_required_when_mipmap_disabled(self):
        import struct

        pipe = self._make_pipeline()
        pipe.config.validation.require_full_mipchain = True
        pipe.config.mipmap.enabled = False
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)          # dwSize
        struct.pack_into("<I", header, 4, 0x1007)       # dwFlags
        struct.pack_into("<I", header, 8, 128)          # dwHeight
        struct.pack_into("<I", header, 12, 128)         # dwWidth
        struct.pack_into("<I", header, 24, 1)           # dwMipMapCount
        struct.pack_into("<I", header, 72, 32)          # ddspf.dwSize
        header[80:84] = b"DXT1"                         # ddspf.dwFourCC
        struct.pack_into("<I", header, 104, 0x1000)     # dwCaps
        with open(path, "wb") as f:
            f.write(b"DDS " + bytes(header))
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        checks = [issue["check"] for issue in issues]
        self.assertNotIn("dds_mipchain", checks)

    def test_dds_magic_invalid(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        path = os.path.join(self.output_dir, "brick_diff.dds")
        with open(path, "wb") as f:
            f.write(b"BAAD" + b"\x00" * 124)
        issues = pipe._validate_compressed_output(path, "dds", "", record)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["check"], "magic_bytes")

    def test_report_written(self):
        pipe = self._make_pipeline()
        record = self._make_record()
        pipe.records = [record]
        pipe.results = {record.filepath: {}}
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        path = os.path.join(self.output_dir, "brick_diff.png")
        Image.fromarray(arr).save(path)
        pipe._validate_outputs()
        report_path = os.path.join(self.output_dir, "output_validation.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path) as f:
            report = json.load(f)
        self.assertIn("total_files_checked", report)
        self.assertIn("passed", report)


class TestSolidColorValidation(unittest.TestCase):
    def _make_pipeline_with_flat_output(self, suffix):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        pipeline = AssetPipeline(config)
        flat = np.full((128, 128, 3), 0.5, dtype=np.float32)
        out_path = get_output_path("brick_diff.png", config.output_dir, suffix=suffix, ext=".png")
        save_image(flat, out_path)
        return pipeline, config, out_path

    def test_flat_roughness_not_flagged(self):
        pipeline, config, path = self._make_pipeline_with_flat_output("_roughness")
        try:
            issues = pipeline._validate_single_output(path, "_roughness", None, {})
            solid_issues = [i for i in issues if i["check"] == "solid_color"]
            self.assertEqual(len(solid_issues), 0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_flat_diffuse_still_flagged(self):
        pipeline, config, path = self._make_pipeline_with_flat_output("")
        try:
            issues = pipeline._validate_single_output(path, "", None, {})
            solid_issues = [i for i in issues if i["check"] == "solid_color"]
            self.assertGreater(len(solid_issues), 0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_flat_metalness_not_flagged(self):
        pipeline, config, path = self._make_pipeline_with_flat_output("_metalness")
        try:
            issues = pipeline._validate_single_output(path, "_metalness", None, {})
            solid_issues = [i for i in issues if i["check"] == "solid_color"]
            self.assertEqual(len(solid_issues), 0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestTGAOutput(unittest.TestCase):
    def test_tga_generated_when_enabled(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.compression.generate_tga = True
        config.compression.generate_dds = False
        config.compression.generate_ktx2 = False
        try:
            img = np.random.rand(64, 64, 3).astype(np.float32)
            save_image(img, os.path.join(config.input_dir, "brick_diff.png"))
            pipeline = AssetPipeline(config)
            pipeline.run()
            tga_files = [f for f in os.listdir(config.output_dir) if f.endswith(".tga")]
            self.assertGreater(len(tga_files), 0)
            from PIL import Image as PILImage
            tga_path = os.path.join(config.output_dir, tga_files[0])
            tga_img = PILImage.open(tga_path)
            tga_img.load()
            self.assertEqual(tga_img.size, (64, 64))
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class _FakeMipmapGenerator:
    def __init__(self):
        self.ktx2_calls = 0
        self.ktx2_mipchain_calls = 0

    def generate_ktx2(self, source_path, output_path, srgb=False, **kw):
        self.ktx2_calls += 1
        with open(output_path, "wb") as f:
            f.write(b"\xabKTX 20\xbb\r\n\x1a\n")
        return True

    def generate_ktx2_mipchain(self, mip_paths, output_path, uastc=True, srgb=False):
        self.ktx2_mipchain_calls += 1
        with open(output_path, "wb") as f:
            f.write(b"\xabKTX 20\xbb\r\n\x1a\n")
        return True


@pytest.mark.slow
class TestKTX2MipchainPreference(unittest.TestCase):
    def test_copy_prefers_mipchain_for_ktx2(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.compression.generate_dds = False
        config.compression.generate_ktx2 = True
        config.validation.enabled = False
        try:
            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), src)

            record = AssetRecord(
                filepath="brick_diff.png", filename="brick_diff.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="brick", file_size_kb=10.0, file_hash="abc123",
            )

            mip0 = os.path.join(config.intermediate_dir, "mip0.png")
            mip1 = os.path.join(config.intermediate_dir, "mip1.png")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            pipeline = AssetPipeline(config)
            pipeline.records = [record]
            pipeline.results = {
                record.filepath: {
                    "upscale": {"upscaled": src},
                    "mipmap": {
                        "main": {
                            "mips": [
                                {"level": 0, "path": mip0},
                                {"level": 1, "path": mip1},
                            ]
                        }
                    },
                }
            }
            pipeline._mipmap_gen = _FakeMipmapGenerator()

            pipeline.copy_to_output()

            self.assertEqual(pipeline._mipmap_gen.ktx2_mipchain_calls, 1)
            self.assertEqual(pipeline._mipmap_gen.ktx2_calls, 0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)

    def test_tga_not_generated_when_disabled(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.input_dir = tempfile.mkdtemp()
        config.output_dir = tempfile.mkdtemp()
        config.intermediate_dir = tempfile.mkdtemp()
        config.comparison_dir = tempfile.mkdtemp()
        config.compression.generate_dds = False
        try:
            save_image(np.random.rand(64, 64, 3).astype(np.float32),
                       os.path.join(config.input_dir, "brick_diff.png"))
            pipeline = AssetPipeline(config)
            pipeline.run()
            tga_files = [f for f in os.listdir(config.output_dir) if f.endswith(".tga")]
            self.assertEqual(len(tga_files), 0)
        finally:
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                shutil.rmtree(d, ignore_errors=True)


class TestPostprocessGating(unittest.TestCase):
    def test_postprocess_skips_when_all_subpasses_disabled(self):
        from AssetBrew.pipeline import AssetPipeline
        config = PipelineConfig()
        config.color_consistency.enabled = False
        config.color_grading.enabled = False
        config.specular_aa.enabled = False
        config.detail_map.enabled = False
        config.orm_packing.enabled = False
        config.seam_repair.enabled = False
        config.emissive.enabled = False
        config.reflection_mask.enabled = False
        pipeline = AssetPipeline(config)

        with mock.patch.object(pipeline, "_check_deps") as deps_check:
            pipeline.phase5_postprocess()
            deps_check.assert_not_called()


@pytest.mark.slow
class TestCompressionFormatSelection(unittest.TestCase):
    def test_base_rgba_prefers_alpha_format_override(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.compression.generate_tga = False
            config.compression.format_map["diffuse_alpha"] = "bc3"

            src = os.path.join(config.input_dir, "alpha_diff.png")
            save_image(np.random.rand(32, 32, 4).astype(np.float32), src)

            rec = AssetRecord(
                filepath="alpha_diff.png",
                filename="alpha_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=4,
                has_alpha=True,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FakeMipmapGenerator:
                def __init__(self):
                    self.formats = []

                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    self.formats.append(compression)
                    # Simulate generated DDS for downstream validation.
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 124)
                    return True

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    self.formats.append(compression)
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 124)
                    return (True, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FakeMipmapGenerator()

            pipeline.copy_to_output()
            self.assertIn("bc3", pipeline._mipmap_gen.formats)


@pytest.mark.slow
class TestCompressionLogging(unittest.TestCase):
    def test_copy_to_output_logs_compression_status_and_size_delta(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.compression.generate_tga = False
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=64,
                original_height=64,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FakeMipmapGenerator:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 128)
                    return True

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 128)
                    return (True, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FakeMipmapGenerator()

            with self.assertLogs("asset_pipeline", level="INFO") as cm:
                pipeline.copy_to_output()

            log_text = "\n".join(cm.output)
            self.assertIn("[compress][DDS][extra] start", log_text)
            self.assertIn("[compress][DDS][extra] OK", log_text)
            self.assertIn("DDS compression delta:", log_text)
            self.assertIn("Compression total delta:", log_text)


@pytest.mark.slow
class TestCompressionFailureEscalation(unittest.TestCase):
    def test_primary_dds_failure_is_recorded_as_copy_error(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.dds",
                filename="brick_diff.dds",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FailingMipmapGenerator:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    return False

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    return (False, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FailingMipmapGenerator()

            pipeline.copy_to_output()
            msg = pipeline.results[rec.filepath]["copy"]["error"]
            self.assertIn("DDS primary compression failed", msg)

    def test_extra_dds_failure_is_recorded_as_copy_error(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FailingMipmapGenerator:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    return False

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    return (False, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FailingMipmapGenerator()

            pipeline.copy_to_output()
            msg = pipeline.results[rec.filepath]["copy"]["error"]
            self.assertIn("DDS extra generation failed", msg)

    def test_degraded_dds_mipchain_is_recorded_as_copy_error(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _DegradingMipmapGenerator:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 128)
                    return True

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    with open(output_path, "wb") as f:
                        f.write(b"DDS " + b"\x00" * 128)
                    return (True, True)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {
                rec.filepath: {
                    "upscale": {"upscaled": src},
                    "mipmap": {"main": {"mips": [{"level": 0, "path": src}]}},
                }
            }
            pipeline._mipmap_gen = _DegradingMipmapGenerator()

            pipeline.copy_to_output()
            msg = pipeline.results[rec.filepath]["copy"]["error"]
            self.assertIn("mipchain degraded", msg)


@pytest.mark.slow
class TestCompressedOutputValidation(unittest.TestCase):
    def test_dds_header_fields_are_validated(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            pipeline = AssetPipeline(config)
            dds_path = os.path.join(tmpdir, "bad.dds")
            with open(dds_path, "wb") as f:
                f.write(b"DDS ")
                f.write(b"\x00" * 124)

            issues = pipeline._validate_compressed_output(
                dds_path, "dds", "", AssetRecord(
                    filepath="bad.dds",
                    filename="bad.dds",
                    texture_type="unknown",
                    original_width=1,
                    original_height=1,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=0.1,
                ),
            )

            checks = {i["check"] for i in issues}
            self.assertIn("dds_header", checks)
            self.assertIn("dds_pixel_format", checks)

    def test_ktx2_level_index_is_validated(self):
        from AssetBrew.pipeline import AssetPipeline
        import struct

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            pipeline = AssetPipeline(config)
            ktx_path = os.path.join(tmpdir, "bad.ktx2")

            header = bytearray(80)
            header[:12] = b"\xabKTX 20\xbb\r\n\x1a\n"
            struct.pack_into("<I", header, 20, 64)   # pixelWidth
            struct.pack_into("<I", header, 24, 64)   # pixelHeight
            struct.pack_into("<I", header, 36, 1)    # faceCount
            struct.pack_into("<I", header, 40, 2)    # levelCount
            with open(ktx_path, "wb") as f:
                f.write(header)

            issues = pipeline._validate_compressed_output(
                ktx_path, "ktx2", "", AssetRecord(
                    filepath="bad.ktx2",
                    filename="bad.ktx2",
                    texture_type="unknown",
                    original_width=1,
                    original_height=1,
                    channels=3,
                    has_alpha=False,
                    is_tileable=False,
                    is_hero=False,
                    material_category="default",
                    file_size_kb=0.1,
                ),
            )

            checks = {i["check"] for i in issues}
            self.assertIn("ktx2_level_index", checks)


class TestUpscaleFallbackWarnings(unittest.TestCase):
    def test_phase1_missing_ai_deps_marks_fallback(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)

            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=64,
                original_height=64,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {}}

            with mock.patch.object(
                pipeline,
                "_check_deps",
                return_value=["torch", "realesrgan", "basicsr"],
            ):
                with self.assertLogs("asset_pipeline", level="WARNING") as cm:
                    pipeline.phase1_upscale()

            self.assertTrue(any("Phase 1 fallback active" in m for m in cm.output))
            upscale = pipeline.results[rec.filepath]["upscale"]
            self.assertEqual(upscale.get("fallback"), "original_input")
            self.assertIn("missing AI dependencies", upscale.get("reason", ""))

    def test_copy_to_output_warns_on_base_texture_fallback(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.intermediate_dir, exist_ok=True)
            os.makedirs(config.comparison_dir, exist_ok=True)
            config.compression.enabled = False
            config.compression.generate_dds = False
            config.compression.generate_ktx2 = False
            config.compression.generate_tga = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.png",
                filename="brick_diff.png",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {
                rec.filepath: {
                    "upscale": {
                        "upscaled": None,
                        "skipped": True,
                        "fallback": "original_input",
                        "reason": "missing AI dependencies: torch",
                    }
                }
            }

            with self.assertLogs("asset_pipeline", level="WARNING") as cm:
                pipeline.copy_to_output()

            self.assertTrue(
                any("Base texture fallback used original inputs" in m for m in cm.output)
            )
            self.assertTrue(
                os.path.exists(os.path.join(config.output_dir, "brick_diff.png"))
            )


@_requires_cv2_scipy
class TestPipelineIntegration(unittest.TestCase):
    def test_end_to_end_run_without_upscale(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            config.upscale.enabled = False
            config.compression.generate_dds = False
            config.compression.generate_ktx2 = False
            config.compression.generate_tga = False
            config.checkpoint.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.png")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), src)

            pipeline = AssetPipeline(config)
            pipeline.run()

            self.assertTrue(
                os.path.exists(os.path.join(config.output_dir, "brick_diff.png"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(config.output_dir, "pipeline_results.json"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(config.output_dir, "output_validation.json"))
            )


@pytest.mark.slow
class TestCompressionFallbackTracking(unittest.TestCase):
    """Verify compression_fallbacks stat is populated on DDS failure."""

    def test_compression_fallback_tracked(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                os.makedirs(d, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.compression.fail_on_fallback = False
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.dds")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.dds",
                filename="brick_diff.dds",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FailingMipmapGen:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    return False

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    return (False, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FailingMipmapGen()

            pipeline.copy_to_output()

            # The PNG fallback should have been shipped and counted
            png_out = os.path.join(config.output_dir, "brick_diff.png")
            self.assertTrue(os.path.exists(png_out))

    def test_fail_on_fallback_prevents_png_output(self):
        from AssetBrew.pipeline import AssetPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            for d in [config.input_dir, config.output_dir,
                      config.intermediate_dir, config.comparison_dir]:
                os.makedirs(d, exist_ok=True)
            config.compression.enabled = True
            config.compression.generate_dds = True
            config.compression.generate_ktx2 = False
            config.compression.fail_on_fallback = True
            config.validation.enabled = False

            src = os.path.join(config.input_dir, "brick_diff.dds")
            save_image(np.random.rand(32, 32, 3).astype(np.float32), src)

            rec = AssetRecord(
                filepath="brick_diff.dds",
                filename="brick_diff.dds",
                texture_type="diffuse",
                original_width=32,
                original_height=32,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            class _FailingMipmapGen:
                def generate_dds(self, source_path, output_path, compression, srgb=False):
                    return False

                def generate_dds_mipchain(self, mip_paths, output_path, compression, srgb=False):
                    return (False, False)

            pipeline = AssetPipeline(config)
            pipeline.records = [rec]
            pipeline.results = {rec.filepath: {"upscale": {"upscaled": src}}}
            pipeline._mipmap_gen = _FailingMipmapGen()

            pipeline.copy_to_output()

            # With fail_on_fallback=True, no PNG should be shipped
            png_out = os.path.join(config.output_dir, "brick_diff.png")
            self.assertFalse(os.path.exists(png_out))
            # Error should be recorded
            msg = pipeline.results[rec.filepath]["copy"]["error"]
            self.assertIn("fail_on_fallback", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
