"""Tests for image I/O utilities."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from AssetBrew.core import (
    load_image, save_image, ensure_rgb, extract_alpha, merge_alpha,
)
from AssetBrew.core.io import srgb_to_linear, linear_to_srgb, luminance_bt709


class TestImageIO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_image(self, w, h, channels=3, mode="RGB"):
        arr = np.random.rand(h, w, channels).astype(np.float32)
        return arr

    def test_save_load_rgb(self):
        arr = self._create_test_image(64, 64, 3)
        path = os.path.join(self.tmpdir, "test_rgb.png")
        save_image(arr, path)
        loaded = load_image(path)
        self.assertEqual(loaded.shape[:2], (64, 64))
        self.assertEqual(loaded.dtype, np.float32)
        self.assertTrue(0 <= loaded.min() and loaded.max() <= 1.0)

    def test_save_load_grayscale(self):
        arr = np.random.rand(32, 32).astype(np.float32)
        path = os.path.join(self.tmpdir, "test_gray.png")
        save_image(arr, path)
        loaded = load_image(path)
        self.assertEqual(loaded.shape, (32, 32, 3))

    def test_save_16bit_grayscale_png_preserves_uint16_storage(self):
        arr = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = os.path.join(self.tmpdir, "test_gray_16.png")
        save_image(arr, path, bits=16)
        with Image.open(path) as img:
            stored = np.asarray(img)
        self.assertEqual(stored.dtype, np.uint16)
        self.assertGreater(int(stored.max()), 255)

    def test_save_load_rgba(self):
        arr = np.random.rand(32, 32, 4).astype(np.float32)
        path = os.path.join(self.tmpdir, "test_rgba.png")
        save_image(arr, path)
        loaded = load_image(path)
        self.assertEqual(loaded.shape, (32, 32, 4))

    def test_clipping(self):
        arr = np.array([[-0.5, 0.5], [1.5, 0.8]], dtype=np.float32)
        path = os.path.join(self.tmpdir, "clip.png")
        save_image(arr, path)
        loaded = load_image(path)
        self.assertGreaterEqual(loaded.min(), 0.0)
        self.assertLessEqual(loaded.max(), 1.0)

    def test_jpeg_strips_alpha(self):
        arr = np.random.rand(32, 32, 4).astype(np.float32)
        path = os.path.join(self.tmpdir, "test.jpg")
        save_image(arr, path)
        loaded = load_image(path)
        self.assertEqual(loaded.shape[-1], 3)

    def test_load_float_image_preserves_hdr_range(self):
        arr = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
        path = os.path.join(self.tmpdir, "hdr_float.tif")
        Image.fromarray(arr).save(path)
        loaded = load_image(path)
        self.assertGreater(float(loaded.max()), 1.0)
        ratio = float(loaded[1, 1, 0] / loaded[1, 0, 0])
        self.assertAlmostEqual(ratio, 2.0, places=2)

    def test_load_mode_conversion_does_not_leak_file_handle(self):
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        path = os.path.join(self.tmpdir, "palette.png")
        renamed = os.path.join(self.tmpdir, "palette_renamed.png")
        Image.fromarray(arr).convert("P").save(path)
        _ = load_image(path)
        os.replace(path, renamed)
        self.assertTrue(os.path.exists(renamed))

    def test_ensure_rgb(self):
        gray = np.random.rand(16, 16).astype(np.float32)
        rgb = ensure_rgb(gray)
        self.assertEqual(rgb.shape, (16, 16, 3))

        rgba = np.random.rand(16, 16, 4).astype(np.float32)
        rgb = ensure_rgb(rgba)
        self.assertEqual(rgb.shape, (16, 16, 3))

    def test_extract_merge_alpha(self):
        rgba = np.random.rand(16, 16, 4).astype(np.float32)
        alpha = extract_alpha(rgba)
        self.assertIsNotNone(alpha)
        self.assertEqual(alpha.shape, (16, 16))

        rgb = rgba[:, :, :3]
        merged = merge_alpha(rgb, alpha)
        self.assertEqual(merged.shape, (16, 16, 4))
        np.testing.assert_allclose(merged[:, :, 3], alpha, atol=1e-6)

    def test_extract_alpha_none_for_rgb(self):
        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        self.assertIsNone(extract_alpha(rgb))

    def test_merge_alpha_rejects_invalid_shape(self):
        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        bad_alpha = np.random.rand(1, 16, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            merge_alpha(rgb, bad_alpha)


class TestDDSReadIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_uncompressed_dds(self, width, height):
        import struct
        DDS_MAGIC = b"DDS "
        HEADER_SIZE = 124
        DDSD_CAPS = 0x1
        DDSD_HEIGHT = 0x2
        DDSD_WIDTH = 0x4
        DDSD_PIXELFORMAT = 0x1000
        FLAGS = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
        DDSCAPS_TEXTURE = 0x1000
        DDPF_RGB = 0x40
        DDPF_ALPHAPIXELS = 0x1
        PF_FLAGS = DDPF_RGB | DDPF_ALPHAPIXELS
        PF_SIZE = 32
        BPP = 32
        R_MASK = 0x00FF0000
        G_MASK = 0x0000FF00
        B_MASK = 0x000000FF
        A_MASK = 0xFF000000
        pitch = width * 4
        header = struct.pack(
            "<4s I I I I I I", DDS_MAGIC, HEADER_SIZE,
            FLAGS, height, width, pitch, 0)
        header += struct.pack("<I", 0)
        header += b"\x00" * 44
        pf = struct.pack(
            "<I I I I I I I I", PF_SIZE, PF_FLAGS, 0,
            BPP, R_MASK, G_MASK, B_MASK, A_MASK)
        header += pf
        header += struct.pack("<I I I I", DDSCAPS_TEXTURE, 0, 0, 0)
        header += struct.pack("<I", 0)
        pixel = struct.pack("BBBB", 0, 128, 255, 255)
        data = pixel * (width * height)
        return header + data

    def test_load_dds_uncompressed(self):
        dds_bytes = self._create_uncompressed_dds(16, 16)
        dds_path = os.path.join(self.tmpdir, "test_color.dds")
        with open(dds_path, "wb") as f:
            f.write(dds_bytes)

        img = load_image(dds_path)
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(img.shape[0], 16)
        self.assertEqual(img.shape[1], 16)
        self.assertTrue(img.shape[2] >= 3)
        self.assertGreaterEqual(img.min(), 0.0)
        self.assertLessEqual(img.max(), 1.0)

        rgb = img[:, :, :3]
        np.testing.assert_allclose(rgb[:, :, 0].mean(), 1.0, atol=0.02)
        np.testing.assert_allclose(rgb[:, :, 1].mean(), 128 / 255.0, atol=0.02)
        np.testing.assert_allclose(rgb[:, :, 2].mean(), 0.0, atol=0.02)


class TestIModeNormalization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_i_mode_uses_declared_bit_depth_not_pixel_max(self):
        # Very dark 32-bit integer TIFF: old max-based logic could leave values unnormalized.
        arr = np.ones((8, 8), dtype=np.int32)
        img = Image.fromarray(arr)
        path = os.path.join(self.tmpdir, "dark_i32.tif")
        img.save(path)

        loaded = load_image(path)
        self.assertEqual(loaded.shape, (8, 8, 3))
        self.assertLess(float(loaded.max()), 1e-6)


class TestCopyToOutputNonPNG(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmpdir, "input")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_jpeg_input_produces_valid_png(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        jpg_path = os.path.join(self.input_dir, "brick_diff.jpg")
        Image.fromarray(arr).save(jpg_path, format="JPEG")

        dst = os.path.join(self.output_dir, "brick_diff.png")
        img = load_image(jpg_path)
        save_image(img, dst)

        with open(dst, "rb") as f:
            magic = f.read(8)
        self.assertEqual(magic[:4], b"\x89PNG")

        reloaded = load_image(dst)
        self.assertEqual(reloaded.shape[0], 32)
        self.assertEqual(reloaded.shape[1], 32)

    def test_bmp_input_produces_valid_png(self):
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        bmp_path = os.path.join(self.input_dir, "stone_diff.bmp")
        Image.fromarray(arr).save(bmp_path, format="BMP")

        dst = os.path.join(self.output_dir, "stone_diff.png")
        img = load_image(bmp_path)
        save_image(img, dst)

        with open(dst, "rb") as f:
            magic = f.read(4)
        self.assertEqual(magic, b"\x89PNG")


class TestSave8BitRounding(unittest.TestCase):
    """Verify 8-bit save uses rounding, not truncation."""

    def test_save_8bit_rounding(self):
        """Values like 0.999 should round to 255, not truncate to 254."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 0.999 * 255 = 254.745 — truncation gives 254, rounding gives 255
            arr = np.full((4, 4, 3), 0.999, dtype=np.float32)
            path = os.path.join(tmpdir, "round_test.png")
            save_image(arr, path, bits=8)

            with Image.open(path) as img:
                pixels = np.asarray(img)
            self.assertEqual(int(pixels[0, 0, 0]), 255)

            # Also verify a value that should round down
            arr2 = np.full((4, 4, 3), 0.501, dtype=np.float32)
            # 0.501 * 255 = 127.755 → rounds to 128
            path2 = os.path.join(tmpdir, "round_test2.png")
            save_image(arr2, path2, bits=8)

            with Image.open(path2) as img2:
                pixels2 = np.asarray(img2)
            self.assertEqual(int(pixels2[0, 0, 0]), 128)


class TestColorFunctions(unittest.TestCase):
    """Tests for sRGB/linear conversion and BT.709 luminance functions."""

    def test_srgb_to_linear_known_values(self):
        vals = np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)
        result = srgb_to_linear(vals)
        # 0.0 stays 0.0
        self.assertAlmostEqual(float(result[0]), 0.0, places=6)
        # 0.04045 is the exact threshold: 0.04045 / 12.92 ≈ 0.003131
        self.assertAlmostEqual(float(result[1]), 0.04045 / 12.92, places=5)
        # 0.5 → ~0.2140 (sRGB to linear)
        self.assertAlmostEqual(float(result[2]), 0.214, places=2)
        # 1.0 stays 1.0
        self.assertAlmostEqual(float(result[3]), 1.0, places=6)

    def test_linear_to_srgb_known_values(self):
        vals = np.array([0.0, 0.0031308, 0.5, 1.0], dtype=np.float32)
        result = linear_to_srgb(vals)
        self.assertAlmostEqual(float(result[0]), 0.0, places=6)
        # 0.0031308 is the threshold: 0.0031308 * 12.92 ≈ 0.04045
        self.assertAlmostEqual(float(result[1]), 0.0031308 * 12.92, places=4)
        # 0.5 → ~0.735 (linear to sRGB)
        self.assertAlmostEqual(float(result[2]), 0.735, places=2)
        self.assertAlmostEqual(float(result[3]), 1.0, places=6)

    def test_srgb_linear_roundtrip(self):
        original = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        roundtripped = linear_to_srgb(srgb_to_linear(original))
        np.testing.assert_allclose(roundtripped, original, atol=1e-5)

    def test_luminance_bt709_pure_channels(self):
        r = np.zeros((1, 1, 3), dtype=np.float32)
        r[0, 0, 0] = 1.0
        g = np.zeros((1, 1, 3), dtype=np.float32)
        g[0, 0, 1] = 1.0
        b = np.zeros((1, 1, 3), dtype=np.float32)
        b[0, 0, 2] = 1.0
        self.assertAlmostEqual(float(luminance_bt709(r)[0, 0]), 0.2126, places=3)
        self.assertAlmostEqual(float(luminance_bt709(g)[0, 0]), 0.7152, places=3)
        self.assertAlmostEqual(float(luminance_bt709(b)[0, 0]), 0.0722, places=3)

    def test_luminance_bt709_assume_srgb(self):
        # With assume_srgb=True, the function should linearize first.
        # sRGB 0.5 → linear ~0.214 → luminance should be ~0.214 for a gray
        gray = np.full((1, 1, 3), 0.5, dtype=np.float32)
        lum_linear = float(luminance_bt709(gray, assume_srgb=False)[0, 0])
        lum_srgb = float(luminance_bt709(gray, assume_srgb=True)[0, 0])
        # Without sRGB assumption, lum = 0.5
        self.assertAlmostEqual(lum_linear, 0.5, places=3)
        # With sRGB assumption, lum ≈ 0.214 (linearized)
        self.assertAlmostEqual(lum_srgb, 0.214, places=2)


class TestNaNGuard(unittest.TestCase):
    """Tests for NaN guard in sRGB/linear conversion (Issue 15)."""

    def test_srgb_to_linear_nan_replaced(self):
        arr = np.array([0.5, float("nan"), 0.8], dtype=np.float32)
        result = srgb_to_linear(arr)
        self.assertFalse(np.isnan(result).any(), "NaN should be replaced")
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_linear_to_srgb_nan_replaced(self):
        arr = np.array([0.2, float("nan"), 0.6], dtype=np.float32)
        result = linear_to_srgb(arr)
        self.assertFalse(np.isnan(result).any(), "NaN should be replaced")
        self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_srgb_to_linear_no_nan_unchanged(self):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = srgb_to_linear(arr)
        self.assertFalse(np.isnan(result).any())
        self.assertAlmostEqual(float(result[2]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
