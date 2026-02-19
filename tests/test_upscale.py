"""Tests for upscaler PoT snapping and aspect ratio."""

import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from AssetBrew.phases.upscale import _compute_target_dims, _snap_to_pot
from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, save_image


class TestUpscalerPoTSnapping(unittest.TestCase):
    def test_square_scales_without_distortion(self):
        h, w = _compute_target_dims(256, 256, 2048)
        self.assertEqual(h, 2048)
        self.assertEqual(w, 2048)

    def test_non_square_preserves_aspect_by_default(self):
        h, w = _compute_target_dims(768, 1024, 2048)
        self.assertEqual(w, 2048)
        self.assertEqual(h, 1536)

    def test_aspect_ratio_preserved_2to1(self):
        h, w = _compute_target_dims(512, 1024, 2048)
        self.assertEqual(w, 2048)
        self.assertEqual(h, 1024)

    def test_aspect_ratio_large_dim_anchored(self):
        h, w = _compute_target_dims(256, 512, 1024)
        self.assertEqual(w, 1024)
        self.assertEqual(h, 512)

    def test_small_image_not_forced_to_pot_by_default(self):
        h, w = _compute_target_dims(100, 60, 256)
        self.assertEqual(h, 256)
        self.assertEqual(w, 154)

    def test_power_of_two_mode_snaps_dimensions(self):
        h, w = _compute_target_dims(768, 1024, 2048, enforce_power_of_two=True)
        self.assertEqual(w, 2048)
        self.assertTrue(h & (h - 1) == 0)

    def test_snap_to_pot_helper(self):
        self.assertEqual(_snap_to_pot(1), 1)
        self.assertEqual(_snap_to_pot(2), 2)
        self.assertEqual(_snap_to_pot(3), 4)
        self.assertEqual(_snap_to_pot(5), 4)
        self.assertEqual(_snap_to_pot(6), 8)
        self.assertEqual(_snap_to_pot(1536), 2048)
        self.assertEqual(_snap_to_pot(1024), 1024)
        self.assertEqual(_snap_to_pot(2048), 2048)


class TestPoTAspectRatioWarning(unittest.TestCase):
    def test_square_no_warning(self):
        with self.assertLogs("asset_pipeline.upscaler", level="WARNING") as cm:
            logging.getLogger("asset_pipeline.upscaler").warning("sentinel")
            _compute_target_dims(512, 512, 2048, enforce_power_of_two=True)
        distortion_msgs = [m for m in cm.output if "distort" in m.lower()]
        self.assertEqual(len(distortion_msgs), 0)

    def test_3to2_ratio_warns(self):
        with self.assertLogs("asset_pipeline.upscaler", level="WARNING") as cm:
            _compute_target_dims(384, 256, 2048, enforce_power_of_two=True)
        distortion_msgs = [m for m in cm.output if "distort" in m.lower()]
        self.assertGreater(len(distortion_msgs), 0)


class TestPreserveTilingConfig(unittest.TestCase):
    def test_preserve_tiling_respects_config_flag(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        tmpdir = tempfile.mkdtemp()
        try:
            config = PipelineConfig()
            config.input_dir = os.path.join(tmpdir, "in")
            config.output_dir = os.path.join(tmpdir, "out")
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            config.comparison_dir = os.path.join(tmpdir, "cmp")
            os.makedirs(config.input_dir, exist_ok=True)
            config.upscale.preserve_tiling = False

            input_path = os.path.join(config.input_dir, "tile_diff.png")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), input_path)

            rec = AssetRecord(
                filepath="tile_diff.png",
                filename="tile_diff.png",
                texture_type="diffuse",
                original_width=16,
                original_height=16,
                channels=3,
                has_alpha=False,
                is_tileable=True,
                is_hero=False,
                material_category="stone",
                file_size_kb=1.0,
                file_hash="abc",
                is_gloss=False,
            )

            up = TextureUpscaler(config)
            seen = {"preserve": None}

            def _fake_ai(
                img,
                orig_h,
                orig_w,
                target_res,
                preserve_tiling=False,
                gamma_correct_model_input=False,
            ):
                seen["preserve"] = preserve_tiling
                return img

            up._upscale_ai_with_recovery = _fake_ai
            up.process(rec)
            self.assertIs(seen["preserve"], False)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestModelDownloadIntegrity(unittest.TestCase):
    def test_auto_download_requires_checksum_by_default(self):
        from AssetBrew.phases import upscale as upscale_module
        from AssetBrew.phases.upscale import TextureUpscaler

        tmpdir = tempfile.mkdtemp()
        try:
            config = PipelineConfig()
            config.upscale.model_dir = os.path.join(tmpdir, "models")
            config.upscale.allow_unverified_model_download = False
            config.upscale.model_sha256 = {}

            up = TextureUpscaler(config)
            with mock.patch.dict(
                upscale_module.MODEL_URLS,
                {"UnitTestModel": "https://example.com/UnitTestModel.pth"},
                clear=False,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    up._resolve_model_path("UnitTestModel")
            self.assertIn("without SHA-256", str(ctx.exception))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_auto_download_can_be_opted_in_without_checksum(self):
        from AssetBrew.phases import upscale as upscale_module
        from AssetBrew.phases.upscale import TextureUpscaler

        tmpdir = tempfile.mkdtemp()
        try:
            config = PipelineConfig()
            config.upscale.model_dir = os.path.join(tmpdir, "models")
            config.upscale.allow_unverified_model_download = True
            config.upscale.model_sha256 = {}

            up = TextureUpscaler(config)

            def _fake_load_file_from_url(url, model_dir, progress=True, file_name=None):
                del url, progress, file_name
                os.makedirs(model_dir, exist_ok=True)
                out = os.path.join(model_dir, "UnitTestModel.pth")
                with open(out, "wb") as f:
                    f.write(b"unit-test-model")
                return out

            fake_basicsr = types.ModuleType("basicsr")
            fake_basicsr_utils = types.ModuleType("basicsr.utils")
            fake_download = types.ModuleType("basicsr.utils.download_util")
            fake_basicsr.__path__ = []
            fake_basicsr_utils.__path__ = []
            fake_basicsr.utils = fake_basicsr_utils
            fake_basicsr_utils.download_util = fake_download
            fake_download.load_file_from_url = _fake_load_file_from_url

            patched_modules = {
                "basicsr": fake_basicsr,
                "basicsr.utils": fake_basicsr_utils,
                "basicsr.utils.download_util": fake_download,
            }
            with mock.patch.dict(sys.modules, patched_modules):
                with mock.patch.dict(
                    upscale_module.MODEL_URLS,
                    {"UnitTestModel": "https://example.com/UnitTestModel.pth"},
                    clear=False,
                ):
                    out_path = up._resolve_model_path("UnitTestModel")
            self.assertTrue(os.path.exists(out_path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestUpscaleRecoveryAndGuards(unittest.TestCase):
    def test_process_rejects_too_small_input(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        tmpdir = tempfile.mkdtemp()
        try:
            cfg = PipelineConfig()
            cfg.input_dir = os.path.join(tmpdir, "in")
            cfg.intermediate_dir = os.path.join(tmpdir, "inter")
            cfg.min_texture_dim = 4
            os.makedirs(cfg.input_dir, exist_ok=True)

            input_name = "tiny_diff.png"
            input_path = os.path.join(cfg.input_dir, input_name)
            save_image(np.random.rand(2, 2, 3).astype(np.float32), input_path)

            rec = AssetRecord(
                filepath=input_name,
                filename=input_name,
                texture_type="diffuse",
                original_width=2,
                original_height=2,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=0.1,
                file_hash="tiny",
                is_gloss=False,
            )

            up = TextureUpscaler(cfg)
            result = up.process(rec)
            self.assertIsNone(result.get("upscaled"))
            self.assertIn("too small", str(result.get("error", "")).lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ai_recovery_reduces_tiles_then_falls_back_cpu(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        cfg.gpu.min_tile_size = 32
        cfg.gpu.fallback_to_cpu = True
        up = TextureUpscaler(cfg)
        up.device = "cuda"
        up._current_tile_size = 128

        clear_calls = {"n": 0}

        def _fake_clear():
            clear_calls["n"] += 1

        up.gpu_monitor.clear_cache = _fake_clear

        def _fake_upscale_ai(*args, **kwargs):
            del args, kwargs
            if up.device == "cpu":
                return np.zeros((16, 16, 3), dtype=np.float32)
            raise RuntimeError("CUDA out of memory")

        up._upscale_ai = _fake_upscale_ai

        out = up._upscale_ai_with_recovery(
            np.zeros((8, 8, 3), dtype=np.float32),
            8,
            8,
            256,
            False,
            False,
        )
        self.assertEqual(out.shape, (16, 16, 3))
        self.assertEqual(up.device, "cuda")
        self.assertGreaterEqual(clear_calls["n"], 1)

    def test_ai_recovery_non_oom_runtime_error_is_propagated(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        up = TextureUpscaler(cfg)

        def _raise_non_oom(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("invalid tensor shape")

        up._upscale_ai = _raise_non_oom
        with self.assertRaises(RuntimeError):
            up._upscale_ai_with_recovery(
                np.zeros((8, 8, 3), dtype=np.float32),
                8,
                8,
                256,
                False,
                False,
            )


class TestUpscaleNormalFloat32Precision(unittest.TestCase):
    """Verify normal map upscale preserves float32 precision beyond 1/255."""

    def test_upscale_normal_float32_precision(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        up = TextureUpscaler(cfg)

        # Create a normal map with values that are NOT multiples of 1/255
        # e.g. 0.50196 (= 128/255) vs 0.50000 — these differ by ~0.002
        normal = np.full((8, 8, 3), 0.5, dtype=np.float32)
        normal[:, :, 2] = 0.75  # Z component

        result = up._upscale_normal(normal, 8, 8, 16)

        # After float32 resize + renormalize, values should NOT be quantized
        # to 1/255 steps. The Z channel should be close to the original
        # renormalized value, not to 191/255 = 0.74902.
        z_mean = float(np.mean(result[:, :, 2]))
        # With uint8 path, z would snap to 0.74902; with float32, it stays
        # closer to the true renormalized value
        quantized_z = 191.0 / 255.0  # 0.74902
        self.assertNotAlmostEqual(z_mean, quantized_z, places=3,
                                  msg="Z channel appears uint8-quantized")


class TestUpscaleNearestFloat32Precision(unittest.TestCase):
    """Verify _upscale_nearest preserves float32 precision (no uint8 quantization)."""

    def test_upscale_nearest_float32_no_quantization(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        up = TextureUpscaler(cfg)

        # Create a soft gradient mask with values not on 1/255 grid
        mask = np.linspace(0.0, 1.0, 8 * 8, dtype=np.float32).reshape(8, 8)
        result = up._upscale_nearest(mask, 8, 8, 16)

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (16, 16))
        # If uint8-quantized, unique values would be ≤256; with float32 we keep
        # the original precision (nearest-neighbor duplicates but doesn't round).
        unique_vals = np.unique(result)
        # Original has 64 unique values; nearest 2x should replicate them
        self.assertGreater(len(unique_vals), 30)
        # Verify values NOT snapped to 1/255 grid
        residuals = np.abs(result.ravel() * 255 - np.round(result.ravel() * 255))
        has_non_grid = np.any(residuals > 0.001)
        self.assertTrue(has_non_grid,
                        "All values snapped to uint8 grid — float32 precision lost")


class TestUpscaleAIMockedModel(unittest.TestCase):
    """Test _upscale_ai with a mocked Real-ESRGAN model."""

    def test_upscale_ai_srgb_uint8_input_and_output_shape(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        up = TextureUpscaler(cfg)

        captured = {}

        class FakeModel:
            def enhance(self, bgr_img, outscale=4):
                captured["input_dtype"] = bgr_img.dtype
                captured["input_shape"] = bgr_img.shape
                captured["outscale"] = outscale
                # Simulate 4x upscale
                h, w = bgr_img.shape[:2]
                return np.zeros((h * outscale, w * outscale, 3), dtype=np.uint8), None

        up._model = FakeModel()
        up._model_netscale = 4

        def _fake_load():
            return up._model
        up._load_model = _fake_load

        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        result = up._upscale_ai(rgb, 16, 16, 64)

        # Model should receive uint8 (sRGB)
        self.assertEqual(captured["input_dtype"], np.uint8,
                         "Real-ESRGAN should receive uint8 sRGB input")
        # Output should be float32 in [0,1]
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        # Output shape should match target
        self.assertEqual(result.shape[:2], (64, 64))

    def test_upscale_ai_with_alpha(self):
        from AssetBrew.phases.upscale import TextureUpscaler

        cfg = PipelineConfig()
        up = TextureUpscaler(cfg)

        class FakeModel:
            def enhance(self, bgr_img, outscale=4):
                h, w = bgr_img.shape[:2]
                return np.full((h * outscale, w * outscale, 3), 128, dtype=np.uint8), None

        up._model = FakeModel()
        up._model_netscale = 4
        up._load_model = lambda: up._model

        # RGBA input
        rgba = np.random.rand(16, 16, 4).astype(np.float32)
        rgba[:, :, 3] = 0.7  # soft alpha
        result = up._upscale_ai(rgba, 16, 16, 64)

        # Should have 4 channels (alpha preserved)
        self.assertEqual(result.shape, (64, 64, 4))
        self.assertEqual(result.dtype, np.float32)


class TestNormalUpscaleFixes(unittest.TestCase):
    """Tests for normal map upscale improvements (Issues 7, 11, 12)."""

    def test_binary_alpha_tolerance_catches_antialiased(self):
        """Tolerance=0.02 catches AI-upscaled near-binary values."""
        from AssetBrew.phases.upscale import TextureUpscaler
        # Values within 0.02 of 0 or 1 should be considered binary
        alpha = np.array([0.0, 0.015, 0.985, 1.0], dtype=np.float32)
        self.assertTrue(TextureUpscaler._is_binary_alpha(alpha))

    def test_binary_alpha_rejects_gradient(self):
        """Gradient alpha with values like 0.3, 0.7 should not be binary."""
        from AssetBrew.phases.upscale import TextureUpscaler
        alpha = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float32)
        self.assertFalse(TextureUpscaler._is_binary_alpha(alpha))

    def test_normal_upscale_preserves_alpha(self):
        """Normal maps with alpha should retain 4 channels after upscale."""
        import importlib.util
        if not importlib.util.find_spec("cv2"):
            self.skipTest("cv2 not installed")
        from AssetBrew.phases.upscale import TextureUpscaler
        config = PipelineConfig()
        up = TextureUpscaler(config)
        # Create a 16x16 RGBA normal map
        normal_rgba = np.zeros((16, 16, 4), dtype=np.float32)
        normal_rgba[:, :, 0] = 0.5  # flat normal x
        normal_rgba[:, :, 1] = 0.5  # flat normal y
        normal_rgba[:, :, 2] = 1.0  # flat normal z
        normal_rgba[:, :, 3] = 0.8  # alpha (e.g. AO data)
        result = up._upscale_normal(normal_rgba, 16, 16, 64)
        self.assertEqual(result.shape[2], 4, "Alpha channel should be preserved")

    def test_normal_upscale_uses_non_lanczos(self):
        """Upscale should NOT use Lanczos to avoid ringing on normals."""
        import importlib.util
        if not importlib.util.find_spec("cv2"):
            self.skipTest("cv2 not installed")
        import cv2
        from AssetBrew.phases.upscale import TextureUpscaler
        config = PipelineConfig()
        up = TextureUpscaler(config)
        normal = np.zeros((16, 16, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0  # flat normal
        with mock.patch("cv2.resize", wraps=cv2.resize) as mock_resize:
            up._upscale_normal(normal, 16, 16, 64)
            # Should have been called with INTER_LINEAR_EXACT (upscale case)
            call_args = mock_resize.call_args
            kw = call_args[1]
            positional = call_args[0]
            interp = kw.get(
                "interpolation",
                positional[-1] if len(positional) > 2 else None,
            )
            self.assertNotEqual(interp, cv2.INTER_LANCZOS4,
                                "Should not use Lanczos for normals")


if __name__ == "__main__":
    unittest.main(verbosity=2)
