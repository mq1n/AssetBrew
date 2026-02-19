"""Tests for texture classification."""

import os
import shutil
import tempfile
import unittest


from AssetBrew.config import PipelineConfig, TextureType
from AssetBrew.core import (
    classify_texture, is_gloss_texture, detect_material_category,
    is_likely_tileable, is_hero_asset, AssetRecord, scan_assets,
)


class TestTextureClassification(unittest.TestCase):
    def test_diffuse_patterns(self):
        self.assertEqual(classify_texture("brick_wall_diff.png"), TextureType.DIFFUSE)
        self.assertEqual(classify_texture("metal_diffuse.tga"), TextureType.DIFFUSE)
        self.assertEqual(classify_texture("wood_color.png"), TextureType.DIFFUSE)
        self.assertEqual(classify_texture("brick_col.png"), TextureType.DIFFUSE)
        self.assertEqual(classify_texture("armor_c.dds"), TextureType.DIFFUSE)
        self.assertEqual(classify_texture("wall_basecolor.png"), TextureType.DIFFUSE)

    def test_normal_patterns(self):
        self.assertEqual(classify_texture("brick_normal.png"), TextureType.NORMAL)
        self.assertEqual(classify_texture("wall_norm.png"), TextureType.NORMAL)
        self.assertEqual(classify_texture("floor_nrm.tga"), TextureType.NORMAL)
        self.assertEqual(classify_texture("rock_n.png"), TextureType.NORMAL)

    def test_height_patterns(self):
        self.assertEqual(classify_texture("stone_height.png"), TextureType.HEIGHT)
        self.assertEqual(classify_texture("brick_h.png"), TextureType.HEIGHT)
        self.assertEqual(classify_texture("ground_disp.tga"), TextureType.HEIGHT)
        self.assertEqual(classify_texture("rock_displacement.png"), TextureType.HEIGHT)
        self.assertEqual(classify_texture("stone_bump.png"), TextureType.HEIGHT)

    def test_roughness_patterns(self):
        self.assertEqual(classify_texture("metal_rough.png"), TextureType.ROUGHNESS)
        self.assertEqual(classify_texture("wood_roughness.tga"), TextureType.ROUGHNESS)
        self.assertEqual(classify_texture("stone_r.png"), TextureType.ROUGHNESS)
        self.assertEqual(classify_texture("test_gloss.png"), TextureType.ROUGHNESS)
        self.assertEqual(classify_texture("metal_glossiness.tga"), TextureType.ROUGHNESS)

    def test_metalness_patterns(self):
        self.assertEqual(classify_texture("gold_metalness.png"), TextureType.METALNESS)
        self.assertEqual(classify_texture("iron_metallic.tga"), TextureType.METALNESS)
        self.assertEqual(classify_texture("copper_metal.png"), TextureType.METALNESS)

    def test_ao_patterns(self):
        self.assertEqual(classify_texture("rock_ao.png"), TextureType.AO)
        self.assertEqual(classify_texture("wall_occlusion.png"), TextureType.AO)

    def test_orm_patterns(self):
        self.assertEqual(classify_texture("test_orm.png"), TextureType.ORM)
        self.assertEqual(classify_texture("wall_rma.png"), TextureType.ORM)
        self.assertEqual(classify_texture("brick_arm.png"), TextureType.ORM)

    def test_unknown_defaults(self):
        self.assertEqual(classify_texture("bear.dds"), TextureType.UNKNOWN)
        self.assertEqual(classify_texture("unnamed.png"), TextureType.UNKNOWN)

    def test_longest_match_wins(self):
        self.assertEqual(classify_texture("test_roughness.png"), TextureType.ROUGHNESS)
        self.assertEqual(classify_texture("brick_normal.png"), TextureType.NORMAL)

    def test_no_false_positives_on_short_suffixes(self):
        self.assertEqual(classify_texture("basic.png"), TextureType.UNKNOWN)
        self.assertEqual(classify_texture("electric.png"), TextureType.UNKNOWN)
        self.assertEqual(classify_texture("surface.png"), TextureType.UNKNOWN)


class TestGlossDetection(unittest.TestCase):
    def test_gloss_detected(self):
        self.assertTrue(is_gloss_texture("metal_gloss.png"))
        self.assertTrue(is_gloss_texture("wood_glossiness.tga"))

    def test_roughness_not_gloss(self):
        self.assertFalse(is_gloss_texture("metal_rough.png"))
        self.assertFalse(is_gloss_texture("metal_roughness.png"))

    def test_non_roughness_not_gloss(self):
        self.assertFalse(is_gloss_texture("brick_diff.png"))
        self.assertFalse(is_gloss_texture("wall_normal.png"))


class TestMaterialDetection(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(detect_material_category("brick_wall_diff.png"), "brick")
        self.assertEqual(detect_material_category("metal_plate.png"), "metal")
        self.assertEqual(detect_material_category("wood_floor.png"), "wood")
        self.assertEqual(detect_material_category("concrete_wall.tga"), "concrete")
        self.assertEqual(detect_material_category("gold_trim.png"), "gold")

    def test_default(self):
        self.assertEqual(detect_material_category("bear.dds"), "default")
        self.assertEqual(detect_material_category("unnamed.png"), "default")

    def test_longest_match_priority(self):
        self.assertEqual(detect_material_category("cobblestone_floor.png"), "cobblestone")


class TestAssetFlags(unittest.TestCase):
    def test_tileable_detection(self):
        config = PipelineConfig()
        self.assertTrue(is_likely_tileable("brick_wall_diff.png", config))
        self.assertTrue(is_likely_tileable("wood_floor_diff.png", config))
        self.assertFalse(is_likely_tileable("character_face.png", config))

    def test_hero_detection(self):
        config = PipelineConfig()
        self.assertTrue(is_hero_asset("character_face.png", config))
        self.assertTrue(is_hero_asset("player_body.png", config))
        self.assertFalse(is_hero_asset("brick_wall.png", config))


class TestAssetRecord(unittest.TestCase):
    def test_to_dict(self):
        record = AssetRecord(
            filepath="test.png", filename="test.png",
            texture_type="diffuse", original_width=256, original_height=256,
            channels=3, has_alpha=False, is_tileable=True, is_hero=False,
            material_category="default", file_size_kb=100.0,
            file_hash="abc", is_gloss=False,
        )
        d = record.to_dict()
        self.assertEqual(d["filepath"], "test.png")
        self.assertEqual(d["is_gloss"], False)
        self.assertIn("is_gloss", d)


class TestDDSInScanAssets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_uncompressed_dds(self, width, height):
        import struct
        DDS_MAGIC = b"DDS "
        HEADER_SIZE = 124
        FLAGS = 0x1 | 0x2 | 0x4 | 0x1000
        DDSCAPS_TEXTURE = 0x1000
        PF_FLAGS = 0x40 | 0x1
        pitch = width * 4
        header = struct.pack(
            "<4s I I I I I I", DDS_MAGIC, HEADER_SIZE,
            FLAGS, height, width, pitch, 0)
        header += struct.pack("<I", 0)
        header += b"\x00" * 44
        pf = struct.pack(
            "<I I I I I I I I", 32, PF_FLAGS, 0, 32,
            0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
        header += pf
        header += struct.pack("<I I I I", DDSCAPS_TEXTURE, 0, 0, 0)
        header += struct.pack("<I", 0)
        pixel = struct.pack("BBBB", 0, 128, 255, 255)
        data = pixel * (width * height)
        return header + data

    def test_dds_in_scan_assets(self):
        dds_bytes = self._create_uncompressed_dds(32, 32)
        dds_path = os.path.join(self.tmpdir, "brick_diff.dds")
        with open(dds_path, "wb") as f:
            f.write(dds_bytes)

        config = PipelineConfig()
        records = scan_assets(self.tmpdir, config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].filename, "brick_diff.dds")
        self.assertEqual(records[0].texture_type, "diffuse")
        self.assertEqual(records[0].original_width, 32)
        self.assertTrue(len(records[0].file_hash) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
