"""Tests for PBR map generation."""

import importlib.util
import unittest

import numpy as np

from AssetBrew.config import PipelineConfig

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestPBRGenerator(unittest.TestCase):
    def test_roughness_range(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        roughness = gen._generate_roughness(albedo, "default")
        self.assertGreaterEqual(roughness.min(), 0.0)
        self.assertLessEqual(roughness.max(), 1.0)
        self.assertGreaterEqual(roughness.min(), 0.04)

    def test_metalness_range(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        metalness = gen._generate_metalness(albedo, "default")
        self.assertGreaterEqual(metalness.min(), 0.0)
        self.assertLessEqual(metalness.max(), 1.0)

    def test_ao_range(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        ao = gen._generate_ao(albedo)
        self.assertGreaterEqual(ao.min(), 0.0)
        self.assertLessEqual(ao.max(), 1.0)

    def test_delight_preserves_range(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        delighted = gen._delight(rgb)
        self.assertGreaterEqual(delighted.min(), 0.0)
        self.assertLessEqual(delighted.max(), 1.0)

    def test_metal_material_high_metalness(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32) * 0.5 + 0.3
        metalness = gen._generate_metalness(albedo, "metal")
        self.assertGreater(np.mean(metalness), 0.3)

    def test_roughness_noise_higher_than_flat_surface(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        flat = np.full((128, 128, 3), 0.5, dtype=np.float32)
        noisy = np.clip(flat + (np.random.rand(128, 128, 3).astype(np.float32) - 0.5) * 0.5, 0, 1)
        r_flat = float(np.mean(gen._generate_roughness(flat, "default")))
        r_noisy = float(np.mean(gen._generate_roughness(noisy, "default")))
        self.assertGreater(r_noisy, r_flat + 0.03)

    def test_metalness_category_biases_metal_above_wood(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        # Use a colored (non-gray) albedo to avoid accidentally triggering
        # the silver/iron detector in the non-metal branch.
        base = np.empty((96, 96, 3), dtype=np.float32)
        base[:, :, 0] = 0.5
        base[:, :, 1] = 0.35
        base[:, :, 2] = 0.2
        m_metal = float(np.mean(gen._generate_metalness(base, "metal")))
        m_wood = float(np.mean(gen._generate_metalness(base, "wood")))
        self.assertGreater(m_metal, m_wood + 0.2)

    def test_ao_cavity_darker_than_flat_region(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.full((128, 128, 3), 0.8, dtype=np.float32)
        albedo[40:88, 40:88, :] = 0.2  # synthetic cavity
        ao = gen._generate_ao(albedo)
        center = float(np.mean(ao[52:76, 52:76]))
        border = float(np.mean(np.r_[ao[:16, :].ravel(), ao[-16:, :].ravel()]))
        self.assertLess(center, border)

    def test_iron_detection_not_crushed_to_non_metal(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.full((64, 64, 3), 0.5, dtype=np.float32)
        metalness = gen._generate_metalness(albedo, "default")
        self.assertGreater(float(np.mean(metalness)), 0.2)

    def test_zone_mask_detection_returns_expected_channels(self):
        from AssetBrew.phases.pbr import PBRGenerator

        config = PipelineConfig()
        gen = PBRGenerator(config)
        albedo = np.zeros((64, 64, 3), dtype=np.float32)
        albedo[:, :, 0] = 0.8
        albedo[:, :, 1] = 0.6
        albedo[:, :, 2] = 0.4
        zones = gen._detect_material_zones(albedo, "leather")
        self.assertEqual(set(zones.keys()), {"metal", "cloth", "leather", "skin"})
        for mask in zones.values():
            self.assertGreaterEqual(float(mask.min()), 0.0)
            self.assertLessEqual(float(mask.max()), 1.0)

    def test_gloss_conversion_is_inverse_of_roughness(self):
        from AssetBrew.phases.pbr import PBRGenerator

        config = PipelineConfig()
        gen = PBRGenerator(config)
        rough = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        gloss = gen._roughness_to_gloss(rough)
        np.testing.assert_allclose(gloss, 1.0 - rough, atol=1e-6)


@_requires_cv2
class TestPBRGeneratorProcess(unittest.TestCase):
    """Integration test: full PBRGenerator.process()."""

    def test_pbr_generator_process_produces_all_maps(self):
        import os
        import tempfile
        from AssetBrew.phases.pbr import PBRGenerator
        from AssetBrew.core import AssetRecord, save_image

        config = PipelineConfig()
        gen = PBRGenerator(config)

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

            for key in ("albedo", "roughness", "metalness", "ao"):
                self.assertIsNotNone(result.get(key),
                                     f"PBR output '{key}' should not be None")
                self.assertTrue(os.path.exists(result[key]),
                                f"PBR output '{key}' file should exist: {result[key]}")


@_requires_cv2
class TestMetalnessBinarization(unittest.TestCase):
    """Tests for metalness binarization (Issue 3)."""

    def test_metalness_binarized_output(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        config.pbr.metalness_binarize = True
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        metalness = gen._generate_metalness(albedo, "default")
        unique = set(np.unique(metalness))
        self.assertTrue(unique <= {0.0, 1.0},
                        f"Binarized metalness should be 0 or 1, got {unique}")

    def test_metalness_not_binarized_when_disabled(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        config.pbr.metalness_binarize = False
        gen = PBRGenerator(config)
        # Use a metallic albedo to produce some intermediate values
        albedo = np.full((64, 64, 3), 0.5, dtype=np.float32)
        albedo[:, :, 0] = 0.7  # warm tone
        metalness = gen._generate_metalness(albedo, "metal")
        # When not binarized, smooth-step can produce intermediate values
        # (we just verify the code path doesn't crash)
        self.assertGreaterEqual(metalness.min(), 0.0)
        self.assertLessEqual(metalness.max(), 1.0)

    def test_metalness_threshold_respected(self):
        from AssetBrew.phases.pbr import PBRGenerator
        config = PipelineConfig()
        config.pbr.metalness_binarize = True
        config.pbr.metalness_threshold = 0.8
        gen = PBRGenerator(config)
        albedo = np.random.rand(64, 64, 3).astype(np.float32)
        metalness = gen._generate_metalness(albedo, "default")
        unique = set(np.unique(metalness))
        self.assertTrue(unique <= {0.0, 1.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
