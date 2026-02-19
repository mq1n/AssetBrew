"""Tests for POM processor."""

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np

from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, load_image, save_image

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestPOMProcessorFixes(unittest.TestCase):
    def test_bilateral_filter_preserves_float_precision(self):
        import cv2 as _cv2
        config = PipelineConfig()
        height = np.linspace(0.0, 0.01, 256 * 256).reshape(256, 256).astype(np.float32)
        filtered = _cv2.bilateralFilter(
            height, d=config.pom.bilateral_filter_d,
            sigmaColor=config.pom.bilateral_sigma_color,
            sigmaSpace=config.pom.bilateral_sigma_space
        )
        self.assertGreater(np.std(filtered), 0)

    def test_normalization_preserves_relative_scale(self):
        from AssetBrew.phases.pom import POMProcessor
        config = PipelineConfig()
        config.intermediate_dir = tempfile.mkdtemp()
        config.input_dir = tempfile.mkdtemp()
        try:
            proc = POMProcessor(config)
            height = np.random.uniform(0.4, 0.6, (64, 64)).astype(np.float32)
            h_path = os.path.join(config.input_dir, "test_height.png")
            save_image(height, h_path, bits=16)
            record = AssetRecord(
                filepath="test_height.png", filename="test_height.png",
                texture_type="height", original_width=64, original_height=64,
                channels=1, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="fabric", file_size_kb=1.0
            )
            result = proc.process(record, h_path)
            self.assertIsNotNone(result["height_refined"])
            refined = load_image(result["height_refined"])
            if refined.ndim == 3:
                refined = np.mean(refined[:, :, :3], axis=-1)
            self.assertGreater(np.min(refined), 0.15)
            self.assertLess(np.max(refined), 0.85)
        finally:
            shutil.rmtree(config.intermediate_dir, ignore_errors=True)
            shutil.rmtree(config.input_dir, ignore_errors=True)

    def test_shader_div_by_zero_guard_glsl(self):
        from AssetBrew.phases.pom import POMProcessor
        glsl = POMProcessor.generate_pom_shader_glsl()
        self.assertIn("abs(denom)", glsl)
        self.assertIn("1e-6", glsl)

    def test_shader_div_by_zero_guard_hlsl(self):
        from AssetBrew.phases.pom import POMProcessor
        hlsl = POMProcessor.generate_pom_shader_hlsl()
        self.assertIn("abs(denom)", hlsl)
        self.assertIn("1e-6", hlsl)


@_requires_cv2
class TestBilateralSigmaColorFloatRange(unittest.TestCase):
    """Regression: bilateral_sigma_color must be passed in [0,1] float range.

    A previous bug divided the config value (already in [0,1]) by 255.0 again,
    making the bilateral filter effectively a no-op.  This test mocks
    cv2.bilateralFilter and asserts the sigmaColor argument stays < 1.0
    (the default is 0.3).
    """

    def test_sigma_color_not_multiplied_by_255(self):
        from unittest import mock
        from AssetBrew.phases.pom import POMProcessor

        config = PipelineConfig()
        config.intermediate_dir = tempfile.mkdtemp()
        config.input_dir = tempfile.mkdtemp()

        try:
            proc = POMProcessor(config)

            # Create a height map PNG using cv2 directly (bypasses save_image
            # temp-file extension issue on some cv2 versions).
            import cv2 as _cv2
            height = np.random.uniform(0.3, 0.7, (64, 64)).astype(np.float32)
            h_path = os.path.join(config.input_dir, "test_sigma.png")
            arr_16 = np.round(height * 65535.0).astype(np.uint16)
            _cv2.imwrite(h_path, arr_16)

            record = AssetRecord(
                filepath="test_sigma.png", filename="test_sigma.png",
                texture_type="height", original_width=64, original_height=64,
                channels=1, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=1.0,
            )

            with mock.patch("AssetBrew.phases.pom.cv2.bilateralFilter",
                            wraps=_cv2.bilateralFilter) as mock_bf:
                # Also mock save_image to avoid the temp-file extension issue
                with mock.patch("AssetBrew.phases.pom.save_image"):
                    proc.process(record, h_path)
                mock_bf.assert_called_once()
                call_kwargs = mock_bf.call_args
                # bilateralFilter(src, d, sigmaColor, sigmaSpace)
                # Can be passed positionally or as keyword args
                if call_kwargs.kwargs.get("sigmaColor") is not None:
                    sigma_color = call_kwargs.kwargs["sigmaColor"]
                else:
                    # Positional: bilateralFilter(src, d, sigmaColor, sigmaSpace)
                    sigma_color = call_kwargs.args[2]

                self.assertLess(sigma_color, 1.0,
                                f"sigmaColor={sigma_color} should be in [0,1] float range, "
                                f"not scaled to uint8 range")
                self.assertAlmostEqual(sigma_color, config.pom.bilateral_sigma_color,
                                       places=5,
                                       msg="sigmaColor should match the config value directly")
        finally:
            shutil.rmtree(config.intermediate_dir, ignore_errors=True)
            shutil.rmtree(config.input_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
