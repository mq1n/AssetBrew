"""Tests for asset scanning, manifest I/O, and path helpers."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from AssetBrew.config import PipelineConfig
from AssetBrew.core import (
    AssetRecord, scan_assets, save_manifest, load_manifest,
    get_output_path, get_intermediate_path,
)


class TestScanAssets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for name in ["brick_diff.png", "metal_normal.png", "wood_gloss.tga"]:
            path = os.path.join(self.tmpdir, name)
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scans_supported_formats(self):
        config = PipelineConfig()
        config.input_dir = self.tmpdir
        records = scan_assets(self.tmpdir, config)
        names = {r.filename for r in records}
        self.assertIn("brick_diff.png", names)
        self.assertIn("metal_normal.png", names)

    def test_classifies_types(self):
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        types = {r.filename: r.texture_type for r in records}
        self.assertEqual(types.get("brick_diff.png"), "diffuse")
        self.assertEqual(types.get("metal_normal.png"), "normal")

    def test_gloss_flag_set(self):
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        gloss_map = {r.filename: r.is_gloss for r in records}
        self.assertTrue(gloss_map.get("wood_gloss.tga", False))
        self.assertFalse(gloss_map.get("brick_diff.png", True))

    def test_opaque_rgba_not_marked_as_effective_alpha(self):
        opaque_rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        opaque_rgba[:, :, :3] = 128
        opaque_rgba[:, :, 3] = 255
        Image.fromarray(opaque_rgba).save(
            os.path.join(self.tmpdir, "opaque_rgba.png")
        )

        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        rec_map = {r.filename: r for r in records}
        self.assertIn("opaque_rgba.png", rec_map)
        self.assertFalse(rec_map["opaque_rgba.png"].has_alpha)

    def test_transparent_rgba_marked_as_effective_alpha(self):
        transparent_rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        transparent_rgba[:, :, :3] = 128
        transparent_rgba[:, :, 3] = 255
        transparent_rgba[0, 0, 3] = 0
        Image.fromarray(transparent_rgba).save(
            os.path.join(self.tmpdir, "transparent_rgba.png")
        )

        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        rec_map = {r.filename: r for r in records}
        self.assertIn("transparent_rgba.png", rec_map)
        self.assertTrue(rec_map["transparent_rgba.png"].has_alpha)


class TestManifestIO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_load_roundtrip(self):
        records = [
            AssetRecord(
                filepath="brick_diff.png", filename="brick_diff.png",
                texture_type="diffuse", original_width=256, original_height=256,
                channels=3, has_alpha=False, is_tileable=True, is_hero=False,
                material_category="brick", file_size_kb=120.5,
                file_hash="abc123", is_gloss=False,
            ),
            AssetRecord(
                filepath="metal_gloss.png", filename="metal_gloss.png",
                texture_type="roughness", original_width=512, original_height=512,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="metal", file_size_kb=256.0,
                file_hash="def456", is_gloss=True,
            ),
        ]
        path = os.path.join(self.tmpdir, "manifest.csv")
        save_manifest(records, path)
        loaded = load_manifest(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].texture_type, "diffuse")
        self.assertEqual(loaded[0].is_gloss, False)
        self.assertEqual(loaded[1].is_gloss, True)

    def test_load_manifest_wraps_parse_errors(self):
        path = os.path.join(self.tmpdir, "bad_manifest.csv")
        with open(path, "w", newline="") as f:
            f.write(
                "filepath,filename,texture_type,original_width,original_height,"
                "channels,has_alpha,is_tileable,is_hero,material_category,"
                "file_size_kb,file_hash,is_gloss\n"
            )
            f.write("a.png,a.png,diffuse,NOT_INT,256,3,False,False,False,brick,12.0,abc,False\n")
        with self.assertRaises(ValueError) as ctx:
            load_manifest(path)
        self.assertIn("row", str(ctx.exception))
        self.assertIn("bad_manifest.csv", str(ctx.exception))

    def test_load_manifest_accepts_boolean_variants(self):
        path = os.path.join(self.tmpdir, "manifest_bool_variants.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(
                "filepath,filename,texture_type,original_width,original_height,"
                "channels,has_alpha,is_tileable,is_hero,material_category,"
                "file_size_kb,file_hash,is_gloss,status\n"
            )
            f.write(
                "a.png,a.png,diffuse,64,64,3,yes,1,off,default,12.0,abc,on,pending\n"
            )
        loaded = load_manifest(path)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].has_alpha)
        self.assertTrue(loaded[0].is_tileable)
        self.assertFalse(loaded[0].is_hero)
        self.assertTrue(loaded[0].is_gloss)

    def test_asset_record_rejects_absolute_filepath(self):
        with self.assertRaises(ValueError):
            AssetRecord(
                filepath=os.path.abspath("a.png"),
                filename="a.png",
                texture_type="diffuse",
                original_width=64,
                original_height=64,
                channels=3,
                has_alpha=False,
                is_tileable=False,
                is_hero=False,
                material_category="default",
                file_size_kb=1.0,
            )


class TestMaxImagePixelsUnlimited(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_global_pixel_guard_is_disabled(self):
        """Pillow's global guard should be None (we validate per-call)."""
        self.assertIsNone(Image.MAX_IMAGE_PIXELS)

    def test_scan_respects_max_image_pixels_config(self):
        """scan_assets skips images that exceed max_image_pixels."""
        config = PipelineConfig()
        config.max_image_pixels = 10  # 10 pixels max
        img_path = os.path.join(self.tmpdir, "big_diff.png")
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

        records = scan_assets(self.tmpdir, config)
        # 8x8 = 64 pixels > 10: should be skipped
        self.assertEqual(len(records), 0)

    def test_scan_allows_images_within_limit(self):
        """scan_assets includes images within max_image_pixels."""
        config = PipelineConfig()
        config.max_image_pixels = 100  # 100 pixels max
        img_path = os.path.join(self.tmpdir, "small_diff.png")
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)

        records = scan_assets(self.tmpdir, config)
        # 8x8 = 64 pixels <= 100: should be included
        self.assertEqual(len(records), 1)


class TestScanAssetsErrorPaths(unittest.TestCase):
    """Test scan_assets gracefully handles broken/corrupt/unreadable files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_corrupt_image_file_is_skipped(self):
        corrupt_path = os.path.join(self.tmpdir, "corrupt_diff.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_REAL_IMAGE_FILE_JUST_GARBAGE")
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        self.assertEqual(len(records), 0)

    def test_empty_file_is_skipped(self):
        empty_path = os.path.join(self.tmpdir, "empty_diff.png")
        open(empty_path, "wb").close()  # zero bytes
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        self.assertEqual(len(records), 0)

    def test_empty_directory_returns_no_records(self):
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        self.assertEqual(len(records), 0)

    def test_unsupported_extension_ignored(self):
        txt_path = os.path.join(self.tmpdir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("hello")
        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        self.assertEqual(len(records), 0)


class TestManifestForwardCompat(unittest.TestCase):
    """Test that manifests with extra columns from newer versions load OK."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_unknown_csv_columns_are_ignored(self):
        manifest_path = os.path.join(self.tmpdir, "manifest.csv")
        # Write a manifest with an extra column that doesn't exist on AssetRecord
        import csv
        fieldnames = list(AssetRecord.__dataclass_fields__.keys()) + ["future_field"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {
                "filepath": "test.png",
                "filename": "test.png",
                "texture_type": "diffuse",
                "original_width": "64",
                "original_height": "64",
                "channels": "3",
                "has_alpha": "false",
                "is_tileable": "false",
                "is_hero": "false",
                "material_category": "default",
                "file_size_kb": "1.0",
                "file_hash": "abc123",
                "is_gloss": "false",
                "future_field": "some_new_data",
            }
            writer.writerow(row)

        records = load_manifest(manifest_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filepath, "test.png")


class TestPathHelpers(unittest.TestCase):
    def test_output_path(self):
        path = get_output_path("subdir/brick_diff.png", "/tmp/out", suffix="_normal", ext=".png")
        self.assertTrue(path.endswith("brick_diff_normal.png"))

    def test_intermediate_path(self):
        path = get_intermediate_path("brick.png", "02_pbr", "/tmp/inter", suffix="_ao", ext=".png")
        self.assertIn("02_pbr", path)
        self.assertTrue(path.endswith("brick_ao.png"))

    def test_helpers_do_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = get_output_path("a/b/c.png", tmpdir, suffix="_n", ext=".png")
            inter_path = get_intermediate_path("a/b/c.png", "01_x", tmpdir, suffix="_m", ext=".png")
            self.assertFalse(os.path.exists(os.path.dirname(out_path)))
            self.assertFalse(os.path.exists(os.path.dirname(inter_path)))

    def test_output_path_normalizes_parent_segments(self):
        path = get_output_path("foo/../bar/tex.png", "/tmp/out", ext=".png")
        self.assertIn(os.path.join("bar", "tex.png"), path)

    def test_output_path_accepts_windows_style_separators(self):
        path = get_output_path(r"foo\bar\tex.png", "/tmp/out", suffix="_n", ext=".png")
        self.assertTrue(path.endswith(os.path.join("foo", "bar", "tex_n.png")))


class TestScanMemoryGuardOrder(unittest.TestCase):
    """Verify pixel limit is checked before full image decode."""

    def test_huge_image_rejected_without_full_decode(self):
        """An image exceeding max_image_pixels should be skipped."""
        import tempfile
        import shutil
        from PIL import Image as PILImage
        from AssetBrew.core.scanning import scan_assets

        tmpdir = tempfile.mkdtemp()
        try:
            # Create a small but real PNG (8x8)
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, "tiny.png")
            PILImage.fromarray(arr).save(path)

            config = PipelineConfig()
            config.input_dir = tmpdir
            # Set a very low pixel limit (lower than 8x8=64)
            config.max_image_pixels = 10

            records = scan_assets(tmpdir, config)
            # The 8x8 image (64 pixels) should be rejected
            self.assertEqual(len(records), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
