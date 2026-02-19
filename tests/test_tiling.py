"""Tests for tiling math."""

import importlib.util
import unittest
from unittest import mock

import numpy as np

from AssetBrew.core import pad_for_tiling, crop_from_padded

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestTilingMath(unittest.TestCase):
    def test_roundtrip_square(self):
        arr = np.random.rand(64, 64, 3).astype(np.float32)
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        self.assertGreater(padded.shape[0], 64)
        self.assertGreater(padded.shape[1], 64)
        scale = 4
        import cv2
        padded_up = cv2.resize(
            (padded * 255).astype(np.uint8),
            (padded.shape[1] * scale, padded.shape[0] * scale),
            interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32) / 255.0
        cropped = crop_from_padded(padded_up, info, scale)
        self.assertEqual(cropped.shape, (64 * scale, 64 * scale, 3))

    def test_roundtrip_rectangular(self):
        arr = np.random.rand(128, 64, 3).astype(np.float32)
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        scale = 4
        import cv2
        padded_up = cv2.resize(
            (padded * 255).astype(np.uint8),
            (padded.shape[1] * scale, padded.shape[0] * scale),
            interpolation=cv2.INTER_LANCZOS4
        ).astype(np.float32) / 255.0
        cropped = crop_from_padded(padded_up, info, scale)
        self.assertEqual(cropped.shape, (128 * scale, 64 * scale, 3))

    def test_pad_info_fields(self):
        arr = np.random.rand(100, 80, 3).astype(np.float32)
        _, info = pad_for_tiling(arr, pad_fraction=0.25)
        self.assertEqual(info["original_h"], 100)
        self.assertEqual(info["original_w"], 80)
        self.assertGreater(info["pad_h"], 0)
        self.assertGreater(info["pad_w"], 0)

    def test_grayscale_tiling(self):
        arr = np.random.rand(64, 64).astype(np.float32)
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        self.assertEqual(padded.ndim, 2)

    def test_crop_from_padded_falls_back_when_cv2_missing(self):
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        padded, info = pad_for_tiling(arr, pad_fraction=0.25)
        scale = 2

        wrong_h = padded.shape[0] * scale - 1
        wrong_w = padded.shape[1] * scale - 1
        upscaled = np.random.rand(wrong_h, wrong_w, 3).astype(np.float32)

        real_import = __import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2":
                raise ImportError("mocked missing cv2")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=_guarded_import):
            cropped = crop_from_padded(upscaled, info, scale)

        self.assertEqual(cropped.shape, (arr.shape[0] * scale, arr.shape[1] * scale, 3))
        self.assertTrue(np.isfinite(cropped).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
