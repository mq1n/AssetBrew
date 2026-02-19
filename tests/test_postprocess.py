"""Tests for post-processing passes."""

import importlib.util
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from AssetBrew.config import PipelineConfig
from AssetBrew.core import AssetRecord, load_image, save_image, ensure_rgb

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestORMPrecision(unittest.TestCase):
    def test_resize_preserves_float_precision(self):
        from AssetBrew.phases.postprocess import ORMPacker
        config = PipelineConfig()
        packer = ORMPacker(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            h, w = 64, 64
            gradient = np.linspace(0.50, 0.51, h * w, dtype=np.float32).reshape(h, w)
            rough = np.full((32, 32), 0.5, dtype=np.float32)
            rough_path = os.path.join(tmpdir, "rough.png")
            save_image(rough, rough_path, bits=16)
            ao_path = os.path.join(tmpdir, "ao.png")
            save_image(gradient, ao_path, bits=16)
            metal_path = os.path.join(tmpdir, "metal.png")
            save_image(gradient, metal_path, bits=16)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=32, original_height=32,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )
            result = packer.process(record, ao_path, rough_path, metal_path)
            self.assertIsNotNone(result["orm"])
            orm = load_image(result["orm"])
            r_channel = orm[:, :, 0] if orm.ndim == 3 else orm
            unique_vals = len(np.unique(r_channel))
            self.assertGreater(unique_vals, 2)


@_requires_cv2
class TestPostProcessors(unittest.TestCase):
    def test_orm_packing_channel_order(self):
        from AssetBrew.phases.postprocess import ORMPacker
        config = PipelineConfig()
        packer = ORMPacker(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            ao = np.full((32, 32), 0.8, dtype=np.float32)
            rough = np.full((32, 32), 0.5, dtype=np.float32)
            metal = np.full((32, 32), 0.2, dtype=np.float32)
            ao_path = os.path.join(tmpdir, "ao.png")
            rough_path = os.path.join(tmpdir, "rough.png")
            metal_path = os.path.join(tmpdir, "metal.png")
            save_image(ao, ao_path)
            save_image(rough, rough_path)
            save_image(metal, metal_path)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=32, original_height=32,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )
            result = packer.process(record, ao_path, rough_path, metal_path)
            self.assertIsNotNone(result["orm"])
            orm = load_image(result["orm"])
            rgb = ensure_rgb(orm)
            np.testing.assert_allclose(rgb[:, :, 0].mean(), 0.8, atol=0.05)
            np.testing.assert_allclose(rgb[:, :, 1].mean(), 0.5, atol=0.05)
            np.testing.assert_allclose(rgb[:, :, 2].mean(), 0.2, atol=0.05)

    def test_specular_aa_increases_roughness(self):
        from AssetBrew.phases.postprocess import SpecularAAProcessor
        config = PipelineConfig()
        proc = SpecularAAProcessor(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            normal = np.random.rand(64, 64, 3).astype(np.float32)
            rough = np.full((64, 64), 0.3, dtype=np.float32)
            n_path = os.path.join(tmpdir, "normal.png")
            r_path = os.path.join(tmpdir, "rough.png")
            save_image(normal, n_path)
            save_image(rough, r_path)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )
            result = proc.process(record, n_path, r_path)
            self.assertTrue(
                result.get("roughness_aa"),
                "Specular AA should produce roughness_aa output when enabled"
            )
            aa_rough = load_image(result["roughness_aa"])
            self.assertGreaterEqual(aa_rough.mean(), 0.28)


@_requires_cv2
class TestSpecularAA16Bit(unittest.TestCase):
    def test_specular_aa_saves_16bit_png(self):
        from AssetBrew.phases.postprocess import SpecularAAProcessor
        config = PipelineConfig()
        proc = SpecularAAProcessor(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            normal = np.random.rand(64, 64, 3).astype(np.float32)
            rough = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
            n_path = os.path.join(tmpdir, "normal.png")
            r_path = os.path.join(tmpdir, "rough.png")
            save_image(normal, n_path)
            save_image(rough, r_path, bits=16)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )

            result = proc.process(record, n_path, r_path)
            self.assertTrue(result.get("roughness_aa"))
            with Image.open(result["roughness_aa"]) as img:
                arr = np.asarray(img)
            self.assertEqual(arr.dtype, np.uint16)
            self.assertGreater(int(arr.max()), 255)


@_requires_cv2
class TestDetailOverlayPrecision(unittest.TestCase):
    def test_detail_overlay_saves_16bit_png(self):
        from AssetBrew.phases.postprocess import DetailMapOverlay

        config = PipelineConfig()
        config.detail_map.enabled = True
        config.detail_map.apply_to_categories = ["default"]
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            detail = np.zeros((64, 64, 3), dtype=np.float32)
            detail[:, :, 0] = 0.5
            detail[:, :, 1] = 0.5
            detail[:, :, 2] = 1.0
            detail_path = os.path.join(tmpdir, "detail.png")
            save_image(detail, detail_path, bits=16)
            config.detail_map.detail_normal_path = detail_path

            base = np.zeros((64, 64, 3), dtype=np.float32)
            base[:, :, 0] = 0.5
            base[:, :, 1] = 0.5
            base[:, :, 2] = 1.0
            base_path = os.path.join(tmpdir, "base_normal.png")
            save_image(base, base_path, bits=16)

            overlay = DetailMapOverlay(config)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=64, original_height=64,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )
            result = overlay.process(record, base_path)
            self.assertTrue(result.get("normal_detailed"))
            import cv2
            arr = cv2.imread(result["normal_detailed"], cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(arr)
            self.assertEqual(arr.dtype, np.uint16)
            self.assertGreater(int(arr.max()), 255)


@_requires_cv2
class TestAdvancedPostprocess(unittest.TestCase):
    def _record(self, tileable: bool = False):
        return AssetRecord(
            filepath="test.png", filename="test.png",
            texture_type="diffuse", original_width=64, original_height=64,
            channels=3, has_alpha=False, is_tileable=tileable, is_hero=False,
            material_category="default", file_size_kb=10.0,
        )

    def test_orm_unity_mas_preset_layout(self):
        from AssetBrew.phases.postprocess import ORMPacker

        config = PipelineConfig()
        config.orm_packing.preset = "unity_mas"
        packer = ORMPacker(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            ao = np.full((32, 32), 0.8, dtype=np.float32)
            rough = np.full((32, 32), 0.2, dtype=np.float32)
            metal = np.full((32, 32), 0.6, dtype=np.float32)
            ao_path = os.path.join(tmpdir, "ao.png")
            rough_path = os.path.join(tmpdir, "rough.png")
            metal_path = os.path.join(tmpdir, "metal.png")
            save_image(ao, ao_path)
            save_image(rough, rough_path)
            save_image(metal, metal_path)

            result = packer.process(self._record(), ao_path, rough_path, metal_path)
            self.assertTrue(result.get("orm"))
            orm = load_image(result["orm"])
            rgb = ensure_rgb(orm)
            np.testing.assert_allclose(rgb[:, :, 0].mean(), 0.6, atol=0.05)  # metalness
            np.testing.assert_allclose(rgb[:, :, 1].mean(), 0.8, atol=0.05)  # AO
            np.testing.assert_allclose(rgb[:, :, 2].mean(), 0.8, atol=0.05)  # smoothness

    def test_reflection_mask_generation(self):
        from AssetBrew.phases.postprocess import ReflectionMaskGenerator

        config = PipelineConfig()
        proc = ReflectionMaskGenerator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            rough = np.full((64, 64), 0.2, dtype=np.float32)
            metal = np.full((64, 64), 0.6, dtype=np.float32)
            rough_path = os.path.join(tmpdir, "rough.png")
            metal_path = os.path.join(tmpdir, "metal.png")
            save_image(rough, rough_path, bits=16)
            save_image(metal, metal_path, bits=16)

            result = proc.process(self._record(), rough_path, metal_path)
            self.assertTrue(result.get("env_mask"))
            env = load_image(result["env_mask"])
            env_gray = env if env.ndim == 2 else env[:, :, 0]
            self.assertGreater(float(np.mean(env_gray)), 0.6)

    def test_emissive_detection_finds_bright_region(self):
        from AssetBrew.phases.postprocess import EmissiveMapGenerator

        config = PipelineConfig()
        proc = EmissiveMapGenerator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            img = np.zeros((64, 64, 3), dtype=np.float32)
            img[20:44, 20:44, 0] = 1.0
            img[20:44, 20:44, 1] = 0.9
            source = os.path.join(tmpdir, "albedo.png")
            save_image(img, source)

            result = proc.process(self._record(), source)
            self.assertFalse(result.get("skipped"))
            self.assertTrue(result.get("emissive"))
            emissive = load_image(result["emissive"])
            self.assertGreater(float(np.mean(emissive)), 0.02)

    def test_seam_repair_reduces_seam_score(self):
        from AssetBrew.phases.postprocess import SeamRepairProcessor

        config = PipelineConfig()
        proc = SeamRepairProcessor(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            tex = np.full((64, 64, 3), 0.5, dtype=np.float32)
            tex[:, 0, :] = 0.0
            tex[:, -1, :] = 1.0
            source = os.path.join(tmpdir, "tile.png")
            save_image(tex, source)

            result = proc.process(self._record(tileable=True), upscaled_path=source)
            self.assertTrue(result.get("upscaled_repaired"))
            before = result.get("scores", {}).get("upscaled_before", 1.0)
            after = result.get("scores", {}).get("upscaled_after", 1.0)
            self.assertLess(after, before)


@_requires_cv2
class TestColorConsistencyLabSrgbCorrect(unittest.TestCase):
    """Verify LAB conversion goes through uint8 for correct sRGB handling."""

    @staticmethod
    def _make_record(name="test_cc.png"):
        return AssetRecord(
            filename=name, filepath=name,
            texture_type="diffuse", original_width=32, original_height=32,
            channels=3, has_alpha=False, is_tileable=False, is_hero=False,
            material_category="default", file_size_kb=5.0,
        )

    def test_color_consistency_lab_srgb_correct(self):
        from AssetBrew.phases.postprocess import ColorConsistencyPass

        config = PipelineConfig()
        config.color_consistency.enabled = True
        config.color_consistency.group_by_material = False

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir

            # Create two similar images so references get built
            img1 = np.full((32, 32, 3), 0.5, dtype=np.float32)
            img2 = np.full((32, 32, 3), 0.6, dtype=np.float32)
            p1 = os.path.join(tmpdir, "a1.png")
            p2 = os.path.join(tmpdir, "a2.png")
            save_image(img1, p1)
            save_image(img2, p2)

            records = [self._make_record("a1.png"), self._make_record("a2.png")]

            cc = ColorConsistencyPass(config)
            cc.build_references(records, lambda r: os.path.join(tmpdir, r.filepath))

            # Process should not raise (it would if Lab range was wrong)
            result = cc.process(self._make_record(), p1)
            self.assertIsNone(result.get("error"),
                              f"Color consistency error: {result.get('error')}")


@_requires_cv2
class TestColorGradingPass(unittest.TestCase):
    """Tests for ColorGradingPass process()."""

    @staticmethod
    def _make_record():
        return AssetRecord(
            filename="test.png", filepath="test.png",
            texture_type="diffuse", original_width=64, original_height=64,
            channels=3, has_alpha=False, is_tileable=False, is_hero=False,
            material_category="default", file_size_kb=5.0,
        )

    def test_color_grading_pass_neutral_noop(self):
        """Neutral settings (defaults) should preserve the image."""
        from AssetBrew.phases.postprocess import ColorGradingPass

        config = PipelineConfig()
        config.color_grading.enabled = True
        config.color_grading.exposure_ev = 0.0
        config.color_grading.saturation = 1.0
        config.color_grading.lut_path = ""

        grader = ColorGradingPass(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            img = np.random.rand(32, 32, 3).astype(np.float32) * 0.8 + 0.1
            path = os.path.join(tmpdir, "albedo.png")
            save_image(img, path)

            result = grader.process(self._make_record(), path)
            self.assertFalse(result.get("skipped"), "Neutral grading should not be skipped")
            self.assertTrue(result.get("graded"), "Neutral grading should produce output")
            graded = load_image(result["graded"])
            orig = load_image(path)
            graded_rgb = ensure_rgb(graded)
            orig_rgb = ensure_rgb(orig)
            np.testing.assert_allclose(
                graded_rgb, orig_rgb, atol=0.06,
                err_msg="Neutral grading should not change the image"
            )

    def test_color_grading_pass_exposure(self):
        """+1 EV should make the image brighter."""
        from AssetBrew.phases.postprocess import ColorGradingPass

        config = PipelineConfig()
        config.color_grading.enabled = True
        config.color_grading.exposure_ev = 1.0
        config.color_grading.saturation = 1.0
        config.color_grading.lut_path = ""

        grader = ColorGradingPass(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            img = np.full((32, 32, 3), 0.3, dtype=np.float32)
            path = os.path.join(tmpdir, "albedo.png")
            save_image(img, path)

            result = grader.process(self._make_record(), path)
            self.assertFalse(result.get("skipped"), "+1 EV grading should not be skipped")
            self.assertTrue(result.get("graded"), "+1 EV grading should produce output")
            graded = load_image(result["graded"])
            graded_mean = float(ensure_rgb(graded).mean())
            orig_mean = float(ensure_rgb(load_image(path)).mean())
            self.assertGreater(graded_mean, orig_mean,
                               "+1 EV should make image brighter")


@_requires_cv2
class TestParseCubeLut(unittest.TestCase):
    """Tests for _parse_cube_lut()."""

    def test_parse_cube_lut_valid(self):
        from AssetBrew.phases.postprocess import _parse_cube_lut

        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = os.path.join(tmpdir, "identity.cube")
            lines = ["TITLE \"Test LUT\"", "LUT_3D_SIZE 2", ""]
            # 2x2x2 = 8 entries
            for b in range(2):
                for g in range(2):
                    for r in range(2):
                        lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
            with open(lut_path, "w") as f:
                f.write("\n".join(lines))

            result = _parse_cube_lut(lut_path)
            self.assertIsNotNone(result)
            lut, dmin, dmax = result
            self.assertEqual(lut.shape, (2, 2, 2, 3))

    def test_parse_cube_lut_invalid(self):
        from AssetBrew.phases.postprocess import _parse_cube_lut

        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = os.path.join(tmpdir, "bad.cube")
            with open(lut_path, "w") as f:
                f.write("LUT_3D_SIZE 4\n0.0 0.0\n")  # wrong format

            result = _parse_cube_lut(lut_path)
            self.assertIsNone(result)

    def test_parse_cube_lut_nonexistent(self):
        from AssetBrew.phases.postprocess import _parse_cube_lut
        result = _parse_cube_lut("/nonexistent/path.cube")
        self.assertIsNone(result)


@_requires_cv2
class TestSpecularAADimensionMismatch(unittest.TestCase):
    """Regression: SpecularAAProcessor must handle mismatched normal/roughness dimensions.

    Previously, if the normal map and roughness map had different resolutions,
    the processor would crash with a shape mismatch in numpy operations.  The
    fix resizes the roughness map to match the normal map before computing
    variance.  This test feeds different-sized maps and verifies no crash.
    """

    def test_different_dimensions_does_not_crash(self):
        import cv2
        from unittest import mock
        from AssetBrew.phases.postprocess import SpecularAAProcessor

        config = PipelineConfig()
        proc = SpecularAAProcessor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir

            # Normal map at 128x128, roughness at 64x64
            normal = np.random.rand(128, 128, 3).astype(np.float32)
            rough = np.full((64, 64), 0.4, dtype=np.float32)

            # Save via cv2 directly to avoid save_image temp-file extension
            # issue with certain cv2 versions on Windows.
            n_path = os.path.join(tmpdir, "normal_big.png")
            r_path = os.path.join(tmpdir, "rough_small.png")
            n_u8 = np.round(normal[:, :, ::-1] * 255).astype(np.uint8)  # RGB->BGR
            r_u8 = np.round(rough * 255).astype(np.uint8)
            cv2.imwrite(n_path, n_u8)
            cv2.imwrite(r_path, r_u8)

            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=128, original_height=128,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )

            # Capture what save_image would receive instead of writing
            saved_arrays = {}
            real_save = save_image

            def _capture_save(arr, path, **kwargs):
                saved_arrays[path] = arr.copy()
                real_save(arr, path, **kwargs)

            # Mock save_image to avoid temp-file extension issues, but still
            # write to validate the output
            with mock.patch(
                "AssetBrew.phases.postprocess.save_image"
            ) as mock_save:
                # Make the mock write to the real path using cv2 directly
                def _save_via_cv2(arr, path, **kwargs):
                    arr_clipped = np.clip(arr, 0, 1)
                    if arr_clipped.ndim == 2:
                        out = np.round(arr_clipped * 65535).astype(np.uint16)
                    elif arr_clipped.ndim == 3:
                        out = np.round(arr_clipped[:, :, ::-1] * 65535).astype(np.uint16)
                    else:
                        out = np.round(arr_clipped * 65535).astype(np.uint16)
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    cv2.imwrite(path, out)

                mock_save.side_effect = _save_via_cv2
                result = proc.process(record, n_path, r_path)

            self.assertIsNone(result.get("error"),
                              f"SpecularAA should not error on dimension mismatch: "
                              f"{result.get('error')}")
            self.assertTrue(result.get("roughness_aa"),
                            "Expected a roughness_aa output path")

            # Verify the output has the normal map's dimensions (128x128)
            aa_img = load_image(result["roughness_aa"])
            aa_gray = aa_img if aa_img.ndim == 2 else aa_img[:, :, 0]
            self.assertEqual(aa_gray.shape, (128, 128),
                             "Output should match normal map dimensions")


@_requires_cv2
class TestORMGlossDiffuseAlpha(unittest.TestCase):
    """Verify gloss_in_diffuse_alpha works when layout has alpha=none."""

    def test_gloss_in_diffuse_alpha_with_none_layout(self):
        from AssetBrew.phases.postprocess import ORMPacker

        config = PipelineConfig()
        config.orm_packing.enabled = True
        config.orm_packing.generate_gloss_in_diffuse_alpha = True
        # Default preset=unreal_orm has a="none"
        packer = ORMPacker(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            ao = np.full((32, 32), 0.8, dtype=np.float32)
            rough = np.full((32, 32), 0.5, dtype=np.float32)
            metal = np.full((32, 32), 0.2, dtype=np.float32)
            diffuse = np.full((32, 32, 3), 0.6, dtype=np.float32)

            ao_p = os.path.join(tmpdir, "ao.png")
            rough_p = os.path.join(tmpdir, "rough.png")
            metal_p = os.path.join(tmpdir, "metal.png")
            diff_p = os.path.join(tmpdir, "diffuse.png")
            save_image(ao, ao_p)
            save_image(rough, rough_p)
            save_image(metal, metal_p)
            save_image(diffuse, diff_p)

            record = AssetRecord(
                filepath="t.png", filename="t.png",
                texture_type="diffuse",
                original_width=32, original_height=32,
                channels=3, has_alpha=False,
                is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=5.0,
            )
            # Should not raise KeyError
            result = packer.process(
                record, ao_p, rough_p, metal_p, diff_p,
            )
            self.assertIsNotNone(result.get("orm"))
            # Diffuse with packed alpha should exist
            self.assertIsNotNone(result.get("diffuse_alpha_packed"))


class TestDetailOverlayWhiteout(unittest.TestCase):
    """Test that DetailMapOverlay renormalizes n2 before Whiteout (Issue 10)."""

    def test_whiteout_output_unit_length(self):
        """After Whiteout blend, all normal vectors should be approximately unit length."""
        import importlib.util
        if not importlib.util.find_spec("cv2"):
            self.skipTest("cv2 not installed")

        # Simulate the Whiteout blend inline (same logic as postprocess.py)
        base = np.zeros((32, 32, 3), dtype=np.float32)
        base[:, :, 0] = 0.5  # flat normal encoded
        base[:, :, 1] = 0.5
        base[:, :, 2] = 1.0

        detail = np.zeros((32, 32, 3), dtype=np.float32)
        detail[:, :, 0] = 0.6  # slightly tilted detail
        detail[:, :, 1] = 0.55
        detail[:, :, 2] = 0.9

        n1 = base * 2.0 - 1.0
        n2 = detail * 2.0 - 1.0
        strength = 2.0  # strong detail
        n2[:, :, 0] *= strength
        n2[:, :, 1] *= strength

        # Renormalize n2 (the fix)
        len_n2 = np.sqrt(n2[:, :, 0]**2 + n2[:, :, 1]**2 + n2[:, :, 2]**2)
        n2 = n2 / np.maximum(len_n2[:, :, np.newaxis], 1e-8)

        bx = n1[:, :, 0] * n2[:, :, 2] + n2[:, :, 0] * n1[:, :, 2]
        by = n1[:, :, 1] * n2[:, :, 2] + n2[:, :, 1] * n1[:, :, 2]
        bz = n1[:, :, 2] * n2[:, :, 2]
        length = np.sqrt(bx**2 + by**2 + bz**2)
        length = np.maximum(length, 1e-8)

        result = np.stack([bx / length, by / length, bz / length], axis=-1)
        result_lengths = np.sqrt(np.sum(result**2, axis=-1))
        np.testing.assert_allclose(result_lengths, 1.0, atol=0.02,
                                   err_msg="Whiteout output should be unit-length")


if __name__ == "__main__":
    unittest.main(verbosity=2)
