"""Tests for config validation and type safety."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from AssetBrew.config import PipelineConfig, _merge_dict_to_dataclass


class TestConfigValidation(unittest.TestCase):
    def test_default_config_valid(self):
        config = PipelineConfig()
        config.validate()

    def test_invalid_workers(self):
        config = PipelineConfig()
        config.max_workers = 0
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_tile_size(self):
        config = PipelineConfig()
        config.upscale.tile_size = 32
        with self.assertRaises(ValueError):
            config.validate()

    def test_half_precision_requires_cuda(self):
        config = PipelineConfig()
        config.device = "cpu"
        config.upscale.half_precision = True
        with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
            config.validate()
        self.assertTrue(any("half_precision" in msg for msg in cm.output))
        self.assertTrue(config.upscale.half_precision)
        config.apply_runtime_fixups()
        self.assertFalse(config.upscale.half_precision)

    def test_max_workers_upper_bound(self):
        config = PipelineConfig()
        config.max_workers = 129
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_compression_format_rejected(self):
        config = PipelineConfig()
        config.compression.format_map["diffuse"] = "bc8"
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_compression_timeout_rejected(self):
        config = PipelineConfig()
        config.compression.tool_timeout_seconds = 0
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_compressonator_encode_with_rejected(self):
        config = PipelineConfig()
        config.compression.compressonator_encode_with = "warp"
        with self.assertRaises(ValueError):
            config.validate()

    def test_compressonator_performance_string_none_normalizes(self):
        config = PipelineConfig()
        config.compression.compressonator_performance = "None"
        config.validate()
        self.assertIsNone(config.compression.compressonator_performance)

    def test_compressonator_performance_string_number_normalizes(self):
        config = PipelineConfig()
        config.compression.compressonator_performance = "0.70"
        config.validate()
        self.assertAlmostEqual(config.compression.compressonator_performance, 0.70)

    def test_invalid_specular_aa_kernel(self):
        config = PipelineConfig()
        config.specular_aa.kernel_size = 4
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_validation_semantic_thresholds(self):
        config = PipelineConfig()
        config.validation.metalness_nonmetal_max_mean = 1.2
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_validation_semantic_threshold_order(self):
        config = PipelineConfig()
        config.validation.metalness_nonmetal_max_mean = 0.7
        config.validation.metalness_metal_min_mean = 0.6
        with self.assertRaises(ValueError):
            config.validate()

    def test_orm_duplicate_channels(self):
        config = PipelineConfig()
        config.orm_packing.r_channel = "ao"
        config.orm_packing.g_channel = "ao"
        config.orm_packing.b_channel = "metalness"
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("unique", str(ctx.exception))

    def test_orm_valid_channels(self):
        config = PipelineConfig()
        config.orm_packing.r_channel = "invalid"
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_orm_preset(self):
        config = PipelineConfig()
        config.orm_packing.preset = "bad_preset"
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_color_grading_lut_strength(self):
        config = PipelineConfig()
        config.color_grading.lut_strength = 2.0
        with self.assertRaises(ValueError):
            config.validate()

    def test_hero_less_than_target(self):
        config = PipelineConfig()
        config.upscale.hero_resolution = 1024
        config.upscale.target_resolution = 2048
        with self.assertRaises(ValueError):
            config.validate()

    def test_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.yaml")
            config = PipelineConfig()
            config.to_yaml(path)
            loaded = PipelineConfig.from_yaml(path)
            loaded.validate()
            self.assertEqual(loaded.max_workers, config.max_workers)
            self.assertEqual(loaded.upscale.tile_size, config.upscale.tile_size)

    def test_normal_default_is_hybrid(self):
        config = PipelineConfig()
        self.assertEqual(config.normal.method, "hybrid")


class TestConfigTypeSafety(unittest.TestCase):
    def test_type_mismatch_rejected(self):
        config = PipelineConfig()
        with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
            _merge_dict_to_dataclass(config, {"max_workers": "four"})
        found = any("type mismatch" in msg for msg in cm.output)
        self.assertTrue(found, f"Expected type mismatch warning: {cm.output}")
        self.assertEqual(config.max_workers, 4)

    def test_int_to_float_promotion_allowed(self):
        config = PipelineConfig()
        _merge_dict_to_dataclass(config.pbr, {"ao_strength": 1})
        self.assertEqual(config.pbr.ao_strength, 1)

    def test_yaml_bool_quirk_rejected(self):
        config = PipelineConfig()
        config.normal.method = "sobel"
        with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
            _merge_dict_to_dataclass(config.normal, {"method": True})
        found = any("type mismatch" in msg for msg in cm.output)
        self.assertTrue(found, f"Expected type mismatch warning: {cm.output}")
        self.assertEqual(config.normal.method, "sobel")

    def test_device_validation_rejects_typo(self):
        config = PipelineConfig()
        config.device = "cuuda"
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("device", str(ctx.exception))

    def test_device_validation_accepts_valid(self):
        for device in ("auto", "cuda", "cuda:1", "cpu"):
            config = PipelineConfig()
            config.device = device
            if device == "cpu":
                config.upscale.half_precision = False
            config.validate()

    def test_device_validation_rejects_malformed_cuda_index(self):
        for device in ("cuda:", "cuda:abc", "cuda:-1", "cuda:1x"):
            config = PipelineConfig()
            config.device = device
            with self.assertRaises(ValueError):
                config.validate()


class TestLogLevelValidation(unittest.TestCase):
    def test_setup_logging_invalid_level_defaults_to_info(self):
        import logging
        from AssetBrew.core import setup_logging
        setup_logging("INVALID_LEVEL")
        # In embedded mode (root already has handlers), level is set on the
        # pipeline logger hierarchy; in standalone mode, on the root logger.
        pipeline_logger = logging.getLogger("asset_pipeline")
        effective = pipeline_logger.getEffectiveLevel()
        self.assertEqual(effective, logging.INFO)

    def test_config_validate_rejects_invalid_log_level(self):
        config = PipelineConfig()
        config.log_level = "VERBOSE"
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("log_level", str(ctx.exception))


class TestCLIYAMLPrecedence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_yaml_device_preserved(self):
        yaml_path = os.path.join(self.tmpdir, "config.yaml")
        config = PipelineConfig()
        config.device = "cpu"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        self.assertEqual(loaded.device, "cpu")

    def test_cli_device_overrides_yaml(self):
        yaml_path = os.path.join(self.tmpdir, "config.yaml")
        config = PipelineConfig()
        config.device = "cpu"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        device_from_cli = "auto"
        loaded.device = device_from_cli
        self.assertEqual(loaded.device, "auto")

    def test_yaml_log_level_preserved(self):
        yaml_path = os.path.join(self.tmpdir, "config.yaml")
        config = PipelineConfig()
        config.log_level = "DEBUG"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        self.assertEqual(loaded.log_level, "DEBUG")

    def test_cli_log_level_overrides_yaml(self):
        yaml_path = os.path.join(self.tmpdir, "config.yaml")
        config = PipelineConfig()
        config.log_level = "DEBUG"
        config.to_yaml(yaml_path)
        loaded = PipelineConfig.from_yaml(yaml_path)
        loaded.log_level = "WARNING"
        self.assertEqual(loaded.log_level, "WARNING")


class TestValidateNoMutation(unittest.TestCase):
    """Verify that validate() does not mutate state on failure."""

    def test_validate_does_not_mutate_format_map_on_error(self):
        config = PipelineConfig()
        config.compression.format_map["albedo"] = "BC7"
        config.max_workers = 0  # Invalid: triggers validation error
        with self.assertRaises(ValueError):
            config.validate()
        # format_map should NOT have been normalized to lowercase
        self.assertEqual(config.compression.format_map["albedo"], "BC7")

    def test_validate_normalizes_format_map_on_success(self):
        config = PipelineConfig()
        config.compression.format_map["albedo"] = "BC7"
        config.validate()
        self.assertEqual(config.compression.format_map["albedo"], "bc7")


class TestFailOnHeuristicMapsDefault(unittest.TestCase):
    def test_default_is_false(self):
        config = PipelineConfig()
        self.assertFalse(config.validation.fail_on_heuristic_maps)


class TestMetalnessMidbandDefault(unittest.TestCase):
    def test_metalness_midband_default(self):
        config = PipelineConfig()
        self.assertAlmostEqual(config.validation.metalness_midband_max_ratio, 0.25)


class TestConfigHardening(unittest.TestCase):
    """Tests for Phase 5 config hardening changes."""

    def test_phase_failure_abort_ratio_default(self):
        config = PipelineConfig()
        self.assertAlmostEqual(config.phase_failure_abort_ratio, 0.50)

    def test_enforce_pot_default_true(self):
        config = PipelineConfig()
        self.assertTrue(config.upscale.enforce_power_of_two)

    def test_cross_validation_orm_pbr_warning(self):
        config = PipelineConfig()
        config.orm_packing.enabled = True
        config.pbr.enabled = False
        with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
            config.validate()
        self.assertTrue(
            any("ORM packing is enabled but PBR generation is disabled" in msg
                for msg in cm.output),
            f"Expected ORM/PBR warning in: {cm.output}"
        )

    def test_cross_validation_specular_aa_normal_warning(self):
        config = PipelineConfig()
        config.specular_aa.enabled = True
        config.normal.enabled = False
        with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
            config.validate()
        self.assertTrue(
            any("Specular AA is enabled but normal map generation is disabled" in msg
                for msg in cm.output),
            f"Expected specular AA / normal warning in: {cm.output}"
        )

    def test_config_version_field_exists(self):
        config = PipelineConfig()
        self.assertEqual(config.config_version, 1)

    def test_config_version_future_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            config = PipelineConfig()
            config.config_version = 99
            config.to_yaml(path)
            with self.assertLogs("asset_pipeline.config", level="WARNING") as cm:
                PipelineConfig.from_yaml(path)
            self.assertTrue(
                any("config_version=99" in msg for msg in cm.output),
                f"Expected version warning in: {cm.output}"
            )

    def test_filter_method_validation(self):
        config = PipelineConfig()
        config.mipmap.filter_method = "invalid_filter"
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("mipmap.filter_method", str(ctx.exception))

    def test_filter_method_valid_values_accepted(self):
        for method in ("lanczos", "area", "bilinear", "linear", "nearest", "box", "cubic"):
            config = PipelineConfig()
            config.mipmap.filter_method = method
            config.validate()  # Should not raise

    def test_exposure_ev_range(self):
        config = PipelineConfig()
        config.color_grading.exposure_ev = 6.0
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("exposure_ev", str(ctx.exception))

        config2 = PipelineConfig()
        config2.color_grading.exposure_ev = -6.0
        with self.assertRaises(ValueError):
            config2.validate()

    def test_exposure_ev_boundary_accepted(self):
        for val in (-5.0, 0.0, 5.0):
            config = PipelineConfig()
            config.color_grading.exposure_ev = val
            config.validate()

    def test_detail_uv_scale_must_be_positive(self):
        config = PipelineConfig()
        config.detail_map.detail_uv_scale = 0
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("detail_uv_scale", str(ctx.exception))

    def test_detail_strength_bounded(self):
        config = PipelineConfig()
        config.detail_map.detail_strength = 2.5
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("detail_strength", str(ctx.exception))


class TestBatch3ConfigValidation(unittest.TestCase):
    """Tests for batch 3 config validation additions."""

    def test_metalness_binarize_defaults(self):
        config = PipelineConfig()
        self.assertTrue(config.pbr.metalness_binarize)
        self.assertEqual(config.pbr.metalness_threshold, 0.5)

    def test_pom_sigma_space_default(self):
        config = PipelineConfig()
        self.assertEqual(config.pom.bilateral_sigma_space, 10.0)

    def test_emissive_thresholds_defaults(self):
        config = PipelineConfig()
        self.assertEqual(config.emissive.saturation_threshold, 0.50)
        self.assertEqual(config.emissive.value_threshold, 0.90)

    def test_negative_pbr_weight_rejected(self):
        config = PipelineConfig()
        config.pbr.roughness_variance_weight = -0.1
        with self.assertRaises(ValueError):
            config.validate()

    def test_negative_gpu_max_vram_mb_rejected(self):
        config = PipelineConfig()
        config.gpu.max_vram_mb = -1
        with self.assertRaises(ValueError):
            config.validate()

    def test_normal_strength_upper_bound_rejected(self):
        config = PipelineConfig()
        config.normal.strength = 200.0
        with self.assertRaises(ValueError):
            config.validate()

    def test_roughness_base_value_out_of_range(self):
        config = PipelineConfig()
        config.pbr.roughness_base_value = -0.1
        with self.assertRaises(ValueError):
            config.validate()
        config.pbr.roughness_base_value = 1.5
        with self.assertRaises(ValueError):
            config.validate()

    def test_pom_sigma_space_upper_bound_rejected(self):
        config = PipelineConfig()
        config.pom.bilateral_sigma_space = 200.0
        with self.assertRaises(ValueError):
            config.validate()

    def test_metalness_threshold_out_of_range(self):
        config = PipelineConfig()
        config.pbr.metalness_threshold = 1.5
        with self.assertRaises(ValueError):
            config.validate()

    def test_empty_output_suffix_rejected(self):
        config = PipelineConfig()
        config.orm_packing.output_suffix = ""
        with self.assertRaises(ValueError):
            config.validate()

    def test_compression_tool_unknown_rejected(self):
        config = PipelineConfig()
        config.compression.tool = "foobar"
        with self.assertRaises(ValueError):
            config.validate()


class TestShippedConfigValid(unittest.TestCase):
    """Verify shipped config.yaml passes validation."""

    def test_shipped_config_yaml_validates(self):
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.yaml",
        )
        if not os.path.exists(yaml_path):
            self.skipTest("config.yaml not found at project root")
        config = PipelineConfig.from_yaml(yaml_path)
        config.validate()  # Should not raise


class TestAutoDeviceProbe(unittest.TestCase):
    def tearDown(self):
        PipelineConfig._resolved_auto_device = None

    def test_auto_device_uses_subprocess_probe_result(self):
        config = PipelineConfig()
        config.device = "auto"
        PipelineConfig._resolved_auto_device = None
        fake_proc = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=0,
            stdout='{"status": "ok", "cuda": true}\n',
            stderr="",
        )

        torch_module = sys.modules.pop("torch", None)
        try:
            with patch("subprocess.run", return_value=fake_proc) as run_mock:
                resolved = config.resolve_device()
            self.assertEqual(resolved, "cuda")
            run_mock.assert_called_once()
        finally:
            if torch_module is not None:
                sys.modules["torch"] = torch_module

    def test_auto_device_subprocess_failure_falls_back_to_cpu(self):
        config = PipelineConfig()
        config.device = "auto"
        PipelineConfig._resolved_auto_device = None
        fake_proc = subprocess.CompletedProcess(
            args=["python", "-c", "..."],
            returncode=-1073741819,
            stdout="",
            stderr="",
        )

        torch_module = sys.modules.pop("torch", None)
        try:
            with patch("subprocess.run", return_value=fake_proc):
                resolved = config.resolve_device()
            self.assertEqual(resolved, "cpu")
        finally:
            if torch_module is not None:
                sys.modules["torch"] = torch_module


if __name__ == "__main__":
    unittest.main(verbosity=2)
