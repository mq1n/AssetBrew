"""Tests for validation and sphere render."""

import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np

from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, load_image, save_image

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestValidator(unittest.TestCase):
    def test_tiling_check_good(self):
        from AssetBrew.phases.validate import Validator
        config = PipelineConfig()
        v = Validator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a deterministic, periodic signal so all wrap seams are smooth.
            # This avoids flaky random-border construction in CI environments.
            x = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=True)
            y = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=True)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            arr = np.stack(
                [
                    0.5 + 0.25 * np.sin(xx) + 0.1 * np.cos(yy),
                    0.5 + 0.2 * np.cos(2.0 * xx),
                    0.5 + 0.2 * np.sin(2.0 * yy),
                ],
                axis=-1,
            ).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            path = os.path.join(tmpdir, "tiling.png")
            save_image(arr, path)
            ok, score = v._check_tiling(path)
            self.assertLess(score, 0.05)

    def test_normal_validation_good(self):
        from AssetBrew.phases.validate import Validator
        config = PipelineConfig()
        v = Validator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            normals = np.zeros((32, 32, 3), dtype=np.float32)
            normals[:, :, 0] = 0.5
            normals[:, :, 1] = 0.5
            normals[:, :, 2] = 1.0
            path = os.path.join(tmpdir, "normal.png")
            save_image(normals, path)
            ok, issues = v._check_normal_map(path)
            self.assertTrue(ok, f"Issues found: {issues}")

    def test_material_semantics_warns_for_nonmetal_high_metalness(self):
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        config.validation.check_tiling = False
        config.validation.check_normals = False
        config.validation.render_test_sphere = False
        config.validation.regression_diff = False
        config.validation.output_comparison = False
        config.validation.strict_material_semantics = False
        config.validation.fail_on_heuristic_maps = False
        validator = Validator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            metalness = np.clip(
                0.9 + (np.random.rand(128, 128).astype(np.float32) * 0.08),
                0.0,
                1.0,
            )
            metalness_path = os.path.join(tmpdir, "metalness.png")
            save_image(metalness, metalness_path, bits=16)

            record = AssetRecord(
                filepath="wood_diff.png",
                filename="wood_diff.png",
                texture_type="diffuse",
                original_width=128,
                original_height=128,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="wood",
                file_size_kb=10.0,
            )

            report = validator.validate_asset(record, {"metalness": metalness_path})
            self.assertTrue(report["passed"])
            self.assertIn("material_semantics", report["checks"])
            self.assertTrue(
                any("Non-metal category" in warning for warning in report["warnings"])
            )

    def test_material_semantics_strict_mode_fails(self):
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        config.validation.check_tiling = False
        config.validation.check_normals = False
        config.validation.render_test_sphere = False
        config.validation.regression_diff = False
        config.validation.output_comparison = False
        config.validation.strict_material_semantics = True
        config.validation.fail_on_heuristic_maps = False
        validator = Validator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            metalness = np.clip(
                0.9 + (np.random.rand(128, 128).astype(np.float32) * 0.08),
                0.0,
                1.0,
            )
            metalness_path = os.path.join(tmpdir, "metalness.png")
            save_image(metalness, metalness_path, bits=16)

            record = AssetRecord(
                filepath="wood_diff.png",
                filename="wood_diff.png",
                texture_type="diffuse",
                original_width=128,
                original_height=128,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="wood",
                file_size_kb=10.0,
            )

            report = validator.validate_asset(record, {"metalness": metalness_path})
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("Non-metal category" in error for error in report["errors"])
            )

    def test_heuristic_generation_can_fail_when_configured(self):
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        config.validation.check_tiling = False
        config.validation.check_normals = False
        config.validation.render_test_sphere = False
        config.validation.regression_diff = False
        config.validation.output_comparison = False
        config.validation.enforce_material_semantics = False
        config.validation.fail_on_heuristic_maps = True
        validator = Validator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            roughness = np.random.rand(128, 128).astype(np.float32)
            normal = np.zeros((128, 128, 3), dtype=np.float32)
            normal[:, :, 0] = 0.5
            normal[:, :, 1] = 0.5
            normal[:, :, 2] = 1.0

            roughness_path = os.path.join(tmpdir, "roughness.png")
            normal_path = os.path.join(tmpdir, "normal.png")
            save_image(roughness, roughness_path, bits=16)
            save_image(normal, normal_path, bits=16)

            record = AssetRecord(
                filepath="stone_diff.png",
                filename="stone_diff.png",
                texture_type="diffuse",
                original_width=128,
                original_height=128,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="stone",
                file_size_kb=10.0,
            )

            report = validator.validate_asset(
                record,
                {"roughness": roughness_path, "normal": normal_path},
            )
            self.assertFalse(report["passed"])
            self.assertIn("heuristic_limitations", report["checks"])
            self.assertTrue(
                any("synthesized" in error.lower() for error in report["errors"])
            )

    def test_albedo_plausibility_warns_on_overbright_texture(self):
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        config.validation.check_tiling = False
        config.validation.check_normals = False
        config.validation.render_test_sphere = False
        config.validation.regression_diff = False
        config.validation.output_comparison = False
        config.validation.strict_material_semantics = False
        config.validation.fail_on_heuristic_maps = False
        validator = Validator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            albedo = np.full((128, 128, 3), 1.0, dtype=np.float32)
            albedo_path = os.path.join(tmpdir, "albedo.png")
            save_image(albedo, albedo_path)
            record = AssetRecord(
                filepath="bright_diff.png",
                filename="bright_diff.png",
                texture_type="diffuse",
                original_width=128,
                original_height=128,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=10.0,
            )
            report = validator.validate_asset(record, {"albedo": albedo_path})
            self.assertTrue(
                any("near-white" in warning.lower() for warning in report["warnings"])
            )

    def test_generate_tiling_quality_report_writes_json(self):
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        validator = Validator(config)
        reports = [
            {
                "filename": "good.png",
                "passed": True,
                "checks": {"tiling": {"passed": True, "seam_score": 0.01, "quality_score": 0.95}},
            },
            {
                "filename": "bad.png",
                "passed": False,
                "checks": {"tiling": {"passed": False, "seam_score": 0.20, "quality_score": 0.20}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "tiling_quality_report.json")
            validator.generate_tiling_quality_report(reports, out)
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total_scored"], 2)
            self.assertEqual(data["entries"][0]["filename"], "bad.png")
            self.assertTrue(data["entries"][0]["flagged_for_review"])


@_requires_cv2
class TestPreClipRangeWarning(unittest.TestCase):
    def test_to_scalar_map_warns_on_out_of_range(self):
        """Verify _to_scalar_map logs a warning when >1% of pixels are out of [0,1]."""
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        v = Validator(config)

        # Create an array where ~50% of pixels are out of range
        arr = np.full((100, 100), 1.5, dtype=np.float32)
        arr[:50, :] = 0.5  # half in range, half out

        with self.assertLogs("asset_pipeline.validator", level="WARNING") as cm:
            result = v._to_scalar_map(arr)

        self.assertTrue(
            any("outside [0, 1] range" in msg for msg in cm.output),
            f"Expected out-of-range warning, got: {cm.output}",
        )
        # Verify result is clipped
        self.assertLessEqual(float(np.max(result)), 1.0)
        self.assertGreaterEqual(float(np.min(result)), 0.0)

    def test_to_scalar_map_no_warning_for_valid_range(self):
        """Verify _to_scalar_map does NOT warn when pixels are in [0,1]."""
        from AssetBrew.phases.validate import Validator

        config = PipelineConfig()
        v = Validator(config)

        arr = np.random.rand(100, 100).astype(np.float32)

        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError if no logs are captured
            with self.assertLogs("asset_pipeline.validator", level="WARNING"):
                v._to_scalar_map(arr)


@_requires_cv2
class TestSphereRenderPBR(unittest.TestCase):
    def test_metal_sphere_reflects_albedo(self):
        from AssetBrew.phases.validate import Validator
        config = PipelineConfig()
        v = Validator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.comparison_dir = tmpdir
            albedo = np.zeros((64, 64, 3), dtype=np.float32)
            albedo[:, :, 0] = 0.9
            albedo_path = os.path.join(tmpdir, "albedo.png")
            save_image(albedo, albedo_path)
            metal = np.ones((64, 64), dtype=np.float32)
            metal_path = os.path.join(tmpdir, "metal.png")
            save_image(metal, metal_path)
            rough = np.full((64, 64), 0.1, dtype=np.float32)
            rough_path = os.path.join(tmpdir, "rough.png")
            save_image(rough, rough_path)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="metal", file_size_kb=10.0,
            )
            paths = {"albedo": albedo_path, "roughness": rough_path, "metalness": metal_path}
            result = v._render_test_sphere(record, paths)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))
            rendered = load_image(result)
            sres = config.validation.sphere_resolution
            cy, cx = sres // 2, sres // 2
            r = sres // 6
            patch = rendered[cy - r:cy + r, cx - r:cx + r, :3]
            avg_r = float(np.mean(patch[:, :, 0]))
            avg_g = float(np.mean(patch[:, :, 1]))
            self.assertGreater(avg_r, avg_g + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
