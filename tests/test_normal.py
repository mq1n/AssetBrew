"""Tests for normal map generation."""

import importlib.util
import unittest

import numpy as np

from AssetBrew.config import PipelineConfig

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestNormalMapGeneration(unittest.TestCase):
    def test_flat_heightmap_gives_up_normals(self):
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        normal = gen._sobel_normal(flat)
        center = normal[16:48, 16:48, :]
        np.testing.assert_allclose(center[:, :, 0], 0.5, atol=0.01)
        np.testing.assert_allclose(center[:, :, 1], 0.5, atol=0.01)
        np.testing.assert_allclose(center[:, :, 2], 1.0, atol=0.02)

    def test_normals_are_unit_length(self):
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        height = np.random.rand(64, 64).astype(np.float32)
        normal = gen._sobel_normal(height)
        decoded = normal * 2.0 - 1.0
        lengths = np.sqrt(np.sum(decoded ** 2, axis=-1))
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)

    def test_validate_and_fix_nan(self):
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        normal = np.random.rand(32, 32, 3).astype(np.float32)
        normal[5, 5, :] = float('nan')
        fixed = gen._validate_and_fix(normal)
        self.assertTrue(np.all(np.isfinite(fixed)))

    def test_strength_adjustment(self):
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        normal = np.full((16, 16, 3), 0.5, dtype=np.float32)
        normal[:, :, 0] = 0.6
        normal[:, :, 2] = 0.9
        weak = gen._adjust_normal_strength(normal, 0.5)
        strong = gen._adjust_normal_strength(normal, 2.0)
        self.assertGreater(
            np.mean(np.abs(strong[:, :, 0] - 0.5)),
            np.mean(np.abs(weak[:, :, 0] - 0.5))
        )


@_requires_cv2
class TestNormalGeneratorProcess(unittest.TestCase):
    """Integration test: full NormalMapGenerator.process() with Sobel method."""

    def test_normal_generator_process_sobel(self):
        import os
        import tempfile
        from AssetBrew.phases.normal import NormalMapGenerator
        from AssetBrew.core import AssetRecord, save_image

        config = PipelineConfig()
        config.normal.method = "sobel"
        gen = NormalMapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            diffuse = np.random.rand(64, 64, 3).astype(np.float32)
            diffuse_path = os.path.join(tmpdir, "diffuse.png")
            save_image(diffuse, diffuse_path)

            record = AssetRecord(
                filepath="diffuse.png", filename="diffuse.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=5.0,
            )
            result = gen.process(record, upscaled_path=diffuse_path)
            self.assertIsNone(result.get("error"), f"Error: {result.get('error')}")
            self.assertFalse(result.get("skipped"))
            self.assertIsNotNone(result.get("normal"))
            self.assertTrue(os.path.exists(result["normal"]))

            # Verify normals are unit-length
            from AssetBrew.core import load_image, ensure_rgb
            normal_img = load_image(result["normal"])
            decoded = ensure_rgb(normal_img) * 2.0 - 1.0
            lengths = np.sqrt(np.sum(decoded ** 2, axis=-1))
            np.testing.assert_allclose(lengths, 1.0, atol=0.02)


@_requires_cv2
class TestNormalInvertY(unittest.TestCase):
    """Verify invert_y produces different Y channels."""

    def test_normal_invert_y(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        height = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25

        config_gl = PipelineConfig()
        config_gl.normal.invert_y = False
        gen_gl = NormalMapGenerator(config_gl)
        normal_gl = gen_gl._sobel_normal(height)

        config_dx = PipelineConfig()
        config_dx.normal.invert_y = True
        gen_dx = NormalMapGenerator(config_dx)
        normal_dx = gen_dx._sobel_normal(height)

        # Y channels should differ (one inverted)
        diff = np.abs(normal_gl[:, :, 1] - normal_dx[:, :, 1]).mean()
        self.assertGreater(diff, 0.001,
                           "invert_y should produce different Y channels")


@_requires_cv2
class TestHeightFromNormalsFallbackFlat(unittest.TestCase):
    """Flat normals should produce flat height map."""

    def test_height_from_normals_fallback_flat(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)

        # Flat normal: (0.5, 0.5, 1.0) encoded -> pointing straight up
        flat_normal = np.zeros((64, 64, 3), dtype=np.float32)
        flat_normal[:, :, 0] = 0.5
        flat_normal[:, :, 1] = 0.5
        flat_normal[:, :, 2] = 1.0

        height = gen._height_from_normals_fallback(flat_normal)
        self.assertEqual(height.shape[:2], (64, 64))
        # Flat normals = no gradient -> height should be nearly constant
        std = float(np.std(height))
        self.assertLess(std, 0.05,
                        f"Flat normals should give near-constant height, got std={std}")


@_requires_cv2
class TestFrankotChellappaFFTRegression(unittest.TestCase):
    """Regression: height-from-normals must use FFT integration, not cumulative sum.

    The old cumulative-sum approach produced a linearly increasing ramp along
    one axis when given a constant-gradient normal map.  Frankot-Chellappa FFT
    integration produces a smooth surface instead.  This test creates a normal
    map encoding a uniform X-gradient and verifies the resulting height map
    does NOT exhibit a monotonic left-to-right ramp (which would indicate
    cumulative sum is still in use).
    """

    def test_height_from_normals_no_ramp_artifact(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)

        # Build a 64x64 normal map with a constant X-gradient:
        # normal = normalize(-0.3, 0, 1) encoded to [0,1]
        size = 64
        nx_val, ny_val, nz_val = -0.3, 0.0, 1.0
        length = np.sqrt(nx_val**2 + ny_val**2 + nz_val**2)
        nx_val /= length
        ny_val /= length
        nz_val /= length

        normal_map = np.zeros((size, size, 3), dtype=np.float32)
        normal_map[:, :, 0] = nx_val * 0.5 + 0.5
        normal_map[:, :, 1] = ny_val * 0.5 + 0.5
        normal_map[:, :, 2] = nz_val * 0.5 + 0.5

        height = gen._height_from_normals_fallback(normal_map)

        # Basic sanity: output is finite and in [0,1]
        self.assertTrue(np.all(np.isfinite(height)),
                        "Height map contains non-finite values")
        self.assertGreaterEqual(float(np.min(height)), 0.0)
        self.assertLessEqual(float(np.max(height)), 1.0)

        # Regression check: cumulative sum would make each row monotonically
        # increasing (or decreasing).  With FFT integration the row means
        # should NOT form a strict linear ramp.  Check that the per-column
        # mean is NOT monotonically sorted (strict ramp).
        col_means = np.mean(height, axis=0)
        diffs = np.diff(col_means)
        all_positive = np.all(diffs > 0)
        all_negative = np.all(diffs < 0)
        self.assertFalse(
            all_positive or all_negative,
            "Height column means form a strict monotonic ramp -- "
            "this indicates cumulative-sum integration instead of FFT"
        )


@_requires_cv2
class TestNormalStrengthZeroZRegression(unittest.TestCase):
    """Regression: _adjust_normal_strength must not produce zero-Z normals.

    With strength > ~2.0 the scaled XY can exceed the unit sphere, and a
    naive sqrt(1 - x^2 - y^2) would yield zero (or NaN for negative
    arguments).  The fix clamps z_sq to a small positive floor.  This test
    verifies Z > 0 for all pixels after applying an extreme strength of 5.0.
    """

    def test_high_strength_no_zero_z(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)

        # Normal map with non-trivial XY content (tilted normals)
        normal = np.zeros((32, 32, 3), dtype=np.float32)
        normal[:, :, 0] = 0.65  # X offset from 0.5
        normal[:, :, 1] = 0.55  # Y offset from 0.5
        normal[:, :, 2] = 0.85  # Z

        result = gen._adjust_normal_strength(normal, 5.0)

        # Decode back to [-1,1]
        decoded = result * 2.0 - 1.0
        z_values = decoded[:, :, 2]

        self.assertTrue(np.all(np.isfinite(decoded)),
                        "Adjusted normal map has non-finite values")
        self.assertTrue(np.all(z_values > 0),
                        f"Z channel has zero or negative values: min={float(z_values.min())}")

        # Also verify unit length is maintained
        lengths = np.sqrt(np.sum(decoded ** 2, axis=-1))
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)


@_requires_cv2
class TestDetailNormalUsesConfigNzScale(unittest.TestCase):
    """Verify _detail_normal respects config's nz_scale (not hardcoded 2.0)."""

    def test_detail_normal_nz_scale_affects_output(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        rgb = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25

        config_flat = PipelineConfig()
        config_flat.normal.nz_scale = 4.0  # flatter normals
        gen_flat = NormalMapGenerator(config_flat)
        normal_flat = gen_flat._detail_normal(rgb)

        config_steep = PipelineConfig()
        config_steep.normal.nz_scale = 0.5  # steeper normals
        gen_steep = NormalMapGenerator(config_steep)
        normal_steep = gen_steep._detail_normal(rgb)

        # Steeper nz_scale should produce more XY deviation from 0.5
        xy_dev_flat = np.mean(np.abs(normal_flat[:, :, :2] - 0.5))
        xy_dev_steep = np.mean(np.abs(normal_steep[:, :, :2] - 0.5))
        self.assertGreater(xy_dev_steep, xy_dev_flat,
                           "Lower nz_scale should produce steeper normals")

    def test_detail_normal_output_is_unit_length(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        normal = gen._detail_normal(rgb)
        decoded = normal * 2.0 - 1.0
        lengths = np.sqrt(np.sum(decoded ** 2, axis=-1))
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)


@_requires_cv2
class TestDeepBumpNormalMocked(unittest.TestCase):
    """Test _deepbump_normal with mocked ONNX and DeepBump modules."""

    def test_deepbump_normal_returns_valid_map(self):
        from unittest import mock
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)

        h, w = 32, 32
        # Fake output: C,H,W normal map in [-1,1]
        fake_pred = np.zeros((3, h, w), dtype=np.float32)
        fake_pred[0] = 0.0   # nx
        fake_pred[1] = 0.0   # ny
        fake_pred[2] = 1.0   # nz (pointing up)

        fake_utils = mock.MagicMock()
        fake_utils.tiles_split.return_value = (np.zeros((1, 1, 256, 256)), {})
        fake_utils.tiles_infer.return_value = np.zeros((1, 3, 256, 256))
        fake_utils.tiles_merge.return_value = fake_pred
        fake_utils.normalize.return_value = fake_pred

        fake_ort_session = mock.MagicMock()

        gen._deepbump_modules = (fake_utils, mock.MagicMock(), "/fake/model.onnx")
        gen._ort_session = fake_ort_session

        rgb = np.random.rand(h, w, 3).astype(np.float32)

        with mock.patch.dict("sys.modules", {"onnxruntime": mock.MagicMock()}):
            import sys
            sys.modules["onnxruntime"].InferenceSession = mock.MagicMock(
                return_value=fake_ort_session
            )
            result = gen._deepbump_normal(rgb)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (h, w, 3))
        self.assertEqual(result.dtype, np.float32)
        # Values should be in [0,1] (remapped from [-1,1])
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_deepbump_normal_returns_none_without_modules(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)
        gen._deepbump_modules = False  # mark as failed

        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        result = gen._deepbump_normal(rgb)
        self.assertIsNone(result)


@_requires_cv2
class TestFFTMemoryGuard(unittest.TestCase):
    """Verify FFT integration handles normal-sized images and has a memory guard."""

    def test_fft_normal_size_produces_valid_height(self):
        from AssetBrew.phases.normal import NormalMapGenerator

        config = PipelineConfig()
        gen = NormalMapGenerator(config)

        normal = np.zeros((64, 64, 3), dtype=np.float32)
        normal[:, :, 0] = 0.5
        normal[:, :, 1] = 0.5
        normal[:, :, 2] = 1.0

        height = gen._height_from_normals_fallback(normal)
        self.assertEqual(height.shape, (64, 64))
        self.assertEqual(height.dtype, np.float32)
        self.assertGreaterEqual(float(height.min()), 0.0)
        self.assertLessEqual(float(height.max()), 1.0)


@_requires_cv2
class TestFrankotChellappaMinPadding(unittest.TestCase):
    """Tests for FFT mirror-padding in non-seamless mode (Issue 8)."""

    def test_non_seamless_no_bowl_artifact(self):
        """Non-seamless height should not force borders to zero."""
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        config.normal.deepbump_seamless_height = False
        gen = NormalMapGenerator(config)

        # Create a normal map with a uniform bump (non-flat normals)
        normal = np.zeros((64, 64, 3), dtype=np.float32)
        normal[:, :, 0] = 0.6   # tilted normal
        normal[:, :, 1] = 0.5
        normal[:, :, 2] = 0.9

        height = gen._height_from_normals_fallback(normal)
        self.assertEqual(height.shape, (64, 64))
        # Border mean should not be dramatically lower than interior
        border = np.concatenate([
            height[0, :], height[-1, :], height[:, 0], height[:, -1]
        ])
        # With the old Hann window, border would be forced near 0
        # With mirror-padding, border should be similar to interior
        self.assertGreater(float(border.mean()), 0.1,
                           "Border should not be forced to zero by windowing")

    def test_non_seamless_shape_preserved(self):
        """Output height map should have same shape as input normal map."""
        from AssetBrew.phases.normal import NormalMapGenerator
        config = PipelineConfig()
        config.normal.deepbump_seamless_height = False
        gen = NormalMapGenerator(config)

        normal = np.zeros((48, 80, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0
        normal[:, :, 0] = 0.55

        height = gen._height_from_normals_fallback(normal)
        self.assertEqual(height.shape, (48, 80))


if __name__ == "__main__":
    unittest.main(verbosity=2)
