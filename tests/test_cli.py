"""Tests for CLI argument handling."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock


class TestCLI(unittest.TestCase):
    def test_ui_flag_uses_ui_entrypoint(self):
        from AssetBrew import cli

        called = {"ui": False}
        fake_ui_module = types.ModuleType("AssetBrew.ui.app")

        def _fake_ui_main():
            called["ui"] = True

        fake_ui_module.main = _fake_ui_main

        with mock.patch.dict(sys.modules, {"AssetBrew.ui.app": fake_ui_module}):
            with mock.patch.object(sys, "argv", ["AssetBrew", "--ui"]):
                cli.main()

        self.assertTrue(called["ui"])

    def test_device_cuda_index_is_accepted(self):
        from AssetBrew import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "in")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(input_dir, exist_ok=True)

            with mock.patch.object(
                sys, "argv",
                [
                    "AssetBrew",
                    "--input", input_dir,
                    "--output", output_dir,
                    "--device", "cuda:1",
                    "--dry-run",
                ],
            ):
                with mock.patch("AssetBrew.pipeline.AssetPipeline") as pipeline_cls:
                    pipe = pipeline_cls.return_value
                    pipe.run.return_value = None
                    pipe._failed_assets = 0
                    with mock.patch("AssetBrew.cli.setup_logging"):
                        cli.main()

                    cfg = pipeline_cls.call_args[0][0]
                    self.assertEqual(cfg.device, "cuda:1")

    def test_device_invalid_index_is_rejected(self):
        from AssetBrew import cli

        with mock.patch.object(sys, "argv", ["AssetBrew", "--device", "cuda:foo"]):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
            self.assertEqual(ctx.exception.code, 1)

    def test_workers_zero_is_not_ignored(self):
        from AssetBrew import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "in")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(input_dir, exist_ok=True)

            argv = [
                "AssetBrew",
                "--input", input_dir,
                "--output", output_dir,
                "--workers", "0",
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("AssetBrew.cli.setup_logging"):
                    with self.assertRaises(SystemExit) as ctx:
                        cli.main()
            self.assertEqual(ctx.exception.code, 1)

    def test_exits_nonzero_when_pipeline_reports_failed_assets(self):
        from AssetBrew import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "in")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(input_dir, exist_ok=True)

            argv = [
                "AssetBrew",
                "--input", input_dir,
                "--output", output_dir,
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("AssetBrew.cli.setup_logging"):
                    with mock.patch("AssetBrew.pipeline.AssetPipeline") as pipeline_cls:
                        pipe = pipeline_cls.return_value
                        pipe.run.return_value = None
                        pipe._failed_assets = 1
                        with self.assertRaises(SystemExit) as ctx:
                            cli.main()
            self.assertEqual(ctx.exception.code, 1)

    def test_generate_config_respects_output_path(self):
        from AssetBrew import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            out_cfg = os.path.join(tmpdir, "generated.yaml")
            with mock.patch.object(
                sys, "argv",
                ["AssetBrew", "--generate-config", "--output", out_cfg],
            ):
                cli.main()
            self.assertTrue(os.path.exists(out_cfg))


if __name__ == "__main__":
    unittest.main(verbosity=2)
