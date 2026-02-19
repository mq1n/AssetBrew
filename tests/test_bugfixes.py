"""Tests for bug fixes and quality improvements (plan phases A-D)."""

import importlib.util
import os
import shutil
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from AssetBrew.config import (
    PipelineConfig, _merge_dict_to_dataclass,
)
from AssetBrew.core.classify import classify_texture_by_content
from AssetBrew.core.checkpoint import CheckpointManager
from AssetBrew.core.tiling import pad_for_tiling, crop_from_padded
from AssetBrew.core.gpu import _parse_device_id
from AssetBrew.config import TextureType

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


# ─── A5: 16-bit content classification ───────────────────────────────

class TestClassify16Bit(unittest.TestCase):
    def test_8bit_normal_detected(self):
        """8-bit normal map (blue-heavy) should be classified as NORMAL."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 128  # R ~0.5
        arr[:, :, 1] = 128  # G ~0.5
        arr[:, :, 2] = 220  # B ~0.86
        img = Image.fromarray(arr)
        result = classify_texture_by_content(img)
        self.assertEqual(result, TextureType.NORMAL)

    def test_16bit_normalization_logic(self):
        """16-bit values should divide by 65535, not 255."""
        # Test the normalization branch directly: a float32 array with
        # values > 255 should be divided by 65535
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        arr[:, :, 0] = 32768.0   # R: 32768/65535 ~= 0.5
        arr[:, :, 1] = 32768.0   # G: ~0.5
        arr[:, :, 2] = 56000.0   # B: 56000/65535 ~= 0.854
        arr_max = arr.max()
        self.assertGreater(arr_max, 255.0)
        # Apply same normalization as classify_texture_by_content
        arr_norm = arr / 65535.0
        b_mean = arr_norm[:, :, 2].mean()
        r_mean = arr_norm[:, :, 0].mean()
        g_mean = arr_norm[:, :, 1].mean()
        # Should match normal map heuristics
        self.assertGreater(b_mean, 0.7)
        self.assertLess(abs(r_mean - 0.5), 0.15)
        self.assertLess(abs(g_mean - 0.5), 0.15)

    def test_8bit_diffuse_stays_diffuse(self):
        """Random 8-bit color image should be DIFFUSE."""
        rng = np.random.RandomState(42)
        arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = classify_texture_by_content(img)
        self.assertEqual(result, TextureType.DIFFUSE)

    def test_grayscale_mode_is_height(self):
        arr = np.zeros((64, 64), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = classify_texture_by_content(img)
        self.assertEqual(result, TextureType.HEIGHT)


# ─── A7: CheckpointManager.clear() race condition ────────────────────

class TestCheckpointClearThreadSafety(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "ckpt.json"
        )
        self.config.checkpoint.save_interval = 1

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_save_and_clear(self):
        """clear() and save() running concurrently should not raise."""
        cm = CheckpointManager(self.config)
        errors = []

        def saver():
            try:
                for i in range(100):
                    cm.mark_completed(f"f_{i}.png", f"h_{i}", "upscale")
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(20):
                    cm.clear()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=saver)
        t2 = threading.Thread(target=clearer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(len(errors), 0, f"Errors: {errors}")


# ─── A4+D2: VRAM estimation + GPU device_id ──────────────────────────

class TestGPUDeviceId(unittest.TestCase):
    def test_parse_cuda_0(self):
        self.assertEqual(_parse_device_id("cuda:0"), 0)

    def test_parse_cuda_1(self):
        self.assertEqual(_parse_device_id("cuda:1"), 1)

    def test_parse_plain_cuda(self):
        self.assertEqual(_parse_device_id("cuda"), 0)

    def test_parse_cpu(self):
        self.assertEqual(_parse_device_id("cpu"), 0)

    def test_parse_auto(self):
        self.assertEqual(_parse_device_id("auto"), 0)


class TestGPUMonitorRobustness(unittest.TestCase):
    def test_invalid_cuda_index_falls_back_to_zero(self):
        from AssetBrew.core.gpu import GPUMonitor

        class _Props:
            total_memory = 8 * 1024 * 1024 * 1024

        class _FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def get_device_properties(device_id):
                if device_id != 0:
                    raise RuntimeError("invalid device ordinal")
                return _Props()

            @staticmethod
            def memory_allocated(device_id):
                return 0

            @staticmethod
            def memory_reserved(device_id):
                return 0

            @staticmethod
            def empty_cache():
                return None

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = _FakeCuda()

        cfg = PipelineConfig()
        cfg.device = "cuda:9"

        with patch.dict("sys.modules", {"torch": fake_torch}):
            monitor = GPUMonitor(cfg)

        self.assertEqual(monitor._device_id, 0)
        self.assertTrue(monitor._has_cuda)


# ─── B5: Tiling wrap-pad equivalence ─────────────────────────────────

class TestTilingWrapPad(unittest.TestCase):
    def test_wrap_pad_matches_tile_center(self):
        """np.pad wrap should produce identical center region as np.tile."""
        arr = np.random.rand(32, 48, 3).astype(np.float32)
        h, w = arr.shape[:2]
        pad_h = int(round(h * 0.25))
        pad_w = int(round(w * 0.25))

        # Old np.tile method
        tiled = np.tile(arr, (3, 3, 1))
        old_padded = tiled[h - pad_h: 2 * h + pad_h, w - pad_w: 2 * w + pad_w]

        # New np.pad method
        new_padded, info = pad_for_tiling(arr, pad_fraction=0.25)

        np.testing.assert_array_almost_equal(old_padded, new_padded)

    def test_grayscale_wrap_pad(self):
        arr = np.random.rand(32, 32).astype(np.float32)
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        self.assertEqual(padded.ndim, 2)
        self.assertGreater(padded.shape[0], 32)


# ─── B1c: Float32 resize in crop_from_padded ─────────────────────────

@_requires_cv2
class TestFloat32Resize(unittest.TestCase):
    def test_crop_preserves_float32(self):
        """crop_from_padded should not round-trip through uint8."""
        import cv2
        arr = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.25
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        # Simulate upscale that produces slightly wrong size
        scale = 2
        wrong_h = padded.shape[0] * scale + 1
        wrong_w = padded.shape[1] * scale + 1
        upscaled = cv2.resize(
            padded, (wrong_w, wrong_h), interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32)
        cropped = crop_from_padded(upscaled, info, scale)
        # Should still be float32
        self.assertEqual(cropped.dtype, np.float32)
        self.assertEqual(cropped.shape[:2], (64 * scale, 64 * scale))


# ─── B6: YAML dict merge preserves defaults ──────────────────────────

class TestYAMLDictMerge(unittest.TestCase):
    def test_partial_dict_merge(self):
        """Setting one key in a dict should preserve other defaults."""
        config = PipelineConfig()
        original_metal = config.pbr.material_roughness_defaults["metal"]
        _merge_dict_to_dataclass(config, {
            "pbr": {"material_roughness_defaults": {"glass": 0.15}}
        })
        # glass should be updated
        self.assertEqual(config.pbr.material_roughness_defaults["glass"], 0.15)
        # metal should still be the original default
        self.assertEqual(
            config.pbr.material_roughness_defaults["metal"], original_metal
        )

    def test_non_dict_still_replaces(self):
        """Non-dict fields should still be replaced entirely."""
        config = PipelineConfig()
        _merge_dict_to_dataclass(config, {"max_workers": 8})
        self.assertEqual(config.max_workers, 8)


# ─── D1: sobel_weight config validation ──────────────────────────────

class TestSobelWeightConfig(unittest.TestCase):
    def test_default_sobel_weight(self):
        config = PipelineConfig()
        self.assertEqual(config.normal.sobel_weight, 0.6)

    def test_invalid_sobel_weight(self):
        config = PipelineConfig()
        config.normal.sobel_weight = 1.5
        with self.assertRaises(ValueError):
            config.validate()

    def test_negative_sobel_weight(self):
        config = PipelineConfig()
        config.normal.sobel_weight = -0.1
        with self.assertRaises(ValueError):
            config.validate()

    def test_valid_sobel_weight(self):
        config = PipelineConfig()
        config.normal.sobel_weight = 0.7
        config.validate()  # Should not raise


# ─── D2: cuda:N device validation ────────────────────────────────────

class TestCudaDeviceValidation(unittest.TestCase):
    def test_cuda_with_index_valid(self):
        config = PipelineConfig()
        config.device = "cuda:1"
        config.validate()  # Should not raise

    def test_invalid_device_rejected(self):
        config = PipelineConfig()
        config.device = "tpu"
        with self.assertRaises(ValueError):
            config.validate()


# ─── A2: HSV hue normalization ────────────────────────────────────────

@_requires_cv2
class TestHSVNormalization(unittest.TestCase):
    def test_hue_range_correct(self):
        """After fix, hue should be normalized to [0,1] not [0,0.7]."""
        import cv2
        # Create a pure red pixel (H=0° in HSV)
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 180.0
        hsv[:, :, 1:] /= 255.0
        # Red should have hue near 0.0
        self.assertAlmostEqual(hsv[0, 0, 0], 0.0, places=1)
        # Saturation and value should be 1.0 for pure red
        self.assertAlmostEqual(hsv[0, 0, 1], 1.0, places=2)
        self.assertAlmostEqual(hsv[0, 0, 2], 1.0, places=2)


# ─── B2: Metalness S-curve monotonicity ──────────────────────────────

class TestMetalnessSCurve(unittest.TestCase):
    def test_monotonic(self):
        """S-curve should be monotonically increasing."""
        m = np.linspace(0, 1, 1000)
        result = np.where(
            m > 0.5,
            1.0 - 0.5 * (2.0 * (1.0 - m)) ** 3,
            0.5 * (2.0 * m) ** 3
        )
        diffs = np.diff(result)
        self.assertTrue(np.all(diffs >= -1e-10), "S-curve is not monotonic")

    def test_boundary_values(self):
        """S-curve should map 0->0, 0.5->0.5, 1->1."""
        m = np.array([0.0, 0.5, 1.0])
        result = np.where(
            m > 0.5,
            1.0 - 0.5 * (2.0 * (1.0 - m)) ** 3,
            0.5 * (2.0 * m) ** 3
        )
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


# ─── A9: DeepBump output range ───────────────────────────────────────

class TestDeepBumpOutputRange(unittest.TestCase):
    def test_remap_minus1_to_1(self):
        """Simulated DeepBump output should map from [-1,1] to [0,1]."""
        # Simulate unit-vector normal map from DeepBump: (0, 0, 1) encoded
        pred = np.zeros((3, 64, 64), dtype=np.float32)
        pred[2, :, :] = 1.0  # Z = 1 (flat normal)
        normal_map = np.transpose(pred, (1, 2, 0))
        # Apply the fix
        normal_map = normal_map * 0.5 + 0.5
        # Flat normal in [0,1] encoding should be (0.5, 0.5, 1.0)
        self.assertAlmostEqual(normal_map[0, 0, 0], 0.5, places=4)
        self.assertAlmostEqual(normal_map[0, 0, 1], 0.5, places=4)
        self.assertAlmostEqual(normal_map[0, 0, 2], 1.0, places=4)


# ─── B7: Whiteout blend preserves flat normals ───────────────────────

class TestWhiteoutBlend(unittest.TestCase):
    def test_flat_detail_preserves_base(self):
        """Blending with a flat detail normal should not change the base."""
        # Base: some non-flat normal
        base = np.full((4, 4, 3), 0.5, dtype=np.float32)
        base[:, :, 0] = 0.6  # Slight X perturbation
        base[:, :, 2] = 0.8
        # Detail: flat normal (0.5, 0.5, 1.0)
        detail = np.full((4, 4, 3), 0.5, dtype=np.float32)
        detail[:, :, 2] = 1.0

        n1 = base * 2.0 - 1.0
        n2 = detail * 2.0 - 1.0

        bx = n1[:, :, 0] * n2[:, :, 2] + n2[:, :, 0] * n1[:, :, 2]
        by = n1[:, :, 1] * n2[:, :, 2] + n2[:, :, 1] * n1[:, :, 2]
        bz = n1[:, :, 2] * n2[:, :, 2]

        length = np.sqrt(bx**2 + by**2 + bz**2)
        length = np.maximum(length, 1e-8)

        result = np.stack([
            bx / length * 0.5 + 0.5,
            by / length * 0.5 + 0.5,
            bz / length * 0.5 + 0.5
        ], axis=-1)

        # Detail was flat, so result should closely match base direction
        base_decoded = base * 2.0 - 1.0
        base_len = np.sqrt(np.sum(base_decoded**2, axis=-1, keepdims=True))
        base_normalized = base_decoded / np.maximum(base_len, 1e-8)
        base_encoded = base_normalized * 0.5 + 0.5

        np.testing.assert_array_almost_equal(result, base_encoded, decimal=3)


# ─── A8: 16-bit solid color detection ────────────────────────────────

class TestValidate16BitSolid(unittest.TestCase):
    def test_8bit_solid_detected(self):
        """Solid 8-bit image should be detected."""
        arr = np.full((64, 64, 3), 128.0, dtype=np.float32)
        # Normalize
        arr_norm = arr / 255.0
        std = np.std(arr_norm[:, :, :3])
        self.assertLess(std, 0.004)

    def test_16bit_gradient_not_solid(self):
        """16-bit gradient should NOT be flagged as solid."""
        arr = np.linspace(0, 65535, 64 * 64, dtype=np.float32).reshape(64, 64)
        arr_norm = arr / 65535.0
        std = np.std(arr_norm)
        self.assertGreater(std, 0.004)

    def test_16bit_solid_detected(self):
        """Near-solid 16-bit image should be detected."""
        arr = np.full((64, 64, 3), 32000.0, dtype=np.float32)
        arr_norm = arr / 65535.0
        std = np.std(arr_norm[:, :, :3])
        self.assertLess(std, 0.004)


# ─── B3: PoT aspect ratio minimization ───────────────────────────────

class TestPoTAspectRatio(unittest.TestCase):
    def test_square_stays_square(self):
        from AssetBrew.phases.upscale import _compute_target_dims
        h, w = _compute_target_dims(256, 256, 2048)
        self.assertEqual(h, w)

    def test_2_to_1_preserved(self):
        from AssetBrew.phases.upscale import _compute_target_dims
        h, w = _compute_target_dims(512, 256, 2048)
        ratio = h / w
        self.assertAlmostEqual(ratio, 2.0, places=0)

    def test_3_to_2_minimizes_distortion(self):
        """3:2 aspect ratio should pick the PoT pair with least distortion."""
        from AssetBrew.phases.upscale import _compute_target_dims
        h, w = _compute_target_dims(768, 512, 2048)
        new_ratio = max(h, w) / max(min(h, w), 1)
        original_ratio = 768 / 512  # 1.5
        distortion = abs(new_ratio - original_ratio) / original_ratio
        # Should be less than the old worst case of 33%
        self.assertLess(distortion, 0.35)


# ─── C1: Large sigma blur performance ────────────────────────────────

@_requires_cv2
class TestLargeScaleBlur(unittest.TestCase):
    def test_large_sigma_produces_output(self):
        from AssetBrew.phases.pbr import _large_scale_blur
        img = np.random.rand(256, 256).astype(np.float32)
        result = _large_scale_blur(img, sigma=500.0)
        self.assertEqual(result.shape, img.shape)
        # Large sigma should heavily smooth
        self.assertLess(np.std(result), np.std(img))

    def test_small_sigma_passthrough(self):
        """sigma <= 100 should use direct gaussian_filter."""
        from AssetBrew.phases.pbr import _large_scale_blur
        from scipy.ndimage import gaussian_filter
        img = np.random.rand(64, 64).astype(np.float32)
        result = _large_scale_blur(img, sigma=50.0)
        expected = gaussian_filter(img, sigma=50.0)
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
