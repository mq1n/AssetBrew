"""Targeted tests for GPU monitor behavior and adaptive tiling."""

import types
import unittest
from unittest.mock import patch

from AssetBrew.config import PipelineConfig
from AssetBrew.core.gpu import GPUMonitor


class _Props:
    total_memory = 8 * 1024 * 1024 * 1024


class _FakeCudaAvailable:
    def __init__(self, allocated=256, reserved=1024):
        self._allocated = allocated * 1024 * 1024
        self._reserved = reserved * 1024 * 1024
        self.empty_cache_calls = 0

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 2

    @staticmethod
    def get_device_properties(device_id):
        if device_id not in (0, 1):
            raise RuntimeError("invalid device")
        return _Props()

    def memory_allocated(self, device_id):
        del device_id
        return self._allocated

    def memory_reserved(self, device_id):
        del device_id
        return self._reserved

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeCudaUnavailable:
    @staticmethod
    def is_available():
        return False


class _FakeCudaFailing:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_properties(device_id):
        del device_id
        return _Props()

    @staticmethod
    def memory_allocated(device_id):
        del device_id
        raise RuntimeError("query failed")

    @staticmethod
    def memory_reserved(device_id):
        del device_id
        raise RuntimeError("query failed")

    @staticmethod
    def empty_cache():
        raise RuntimeError("clear failed")


def _fake_torch(cuda_obj):
    module = types.ModuleType("torch")
    module.cuda = cuda_obj
    return module


class TestGPUMonitor(unittest.TestCase):
    def test_available_usage_oom_and_tiling(self):
        cfg = PipelineConfig()
        cfg.device = "cuda:1"
        cfg.gpu.enabled = True
        cfg.gpu.max_vram_mb = 0
        cfg.gpu.min_tile_size = 64
        fake_cuda = _FakeCudaAvailable(allocated=512, reserved=7000)
        fake_torch = _fake_torch(fake_cuda)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            monitor = GPUMonitor(cfg)
            self.assertTrue(monitor.available)
            usage = monitor.get_vram_usage()
            self.assertGreater(usage["allocated_mb"], 0)
            self.assertGreater(usage["reserved_mb"], 0)
            self.assertGreater(usage["max_mb"], usage["reserved_mb"])
            self.assertTrue(monitor.check_oom_risk(needed_mb=400))

            suggested = monitor.suggest_tile_size(current_tile=512, image_h=2048, image_w=2048)
            self.assertGreaterEqual(suggested, cfg.gpu.min_tile_size)
            self.assertLessEqual(suggested, 512)

            monitor.log_usage("unit")
            monitor.clear_cache()
            self.assertEqual(fake_cuda.empty_cache_calls, 1)

    def test_unavailable_gpu_returns_safe_defaults(self):
        cfg = PipelineConfig()
        cfg.device = "cuda"
        cfg.gpu.enabled = True
        fake_torch = _fake_torch(_FakeCudaUnavailable())

        with patch.dict("sys.modules", {"torch": fake_torch}):
            monitor = GPUMonitor(cfg)
            self.assertFalse(monitor.available)
            self.assertEqual(
                monitor.get_vram_usage(),
                {"allocated_mb": 0, "reserved_mb": 0, "max_mb": 0, "free_mb": 0},
            )
            self.assertFalse(monitor.check_oom_risk(needed_mb=100))
            self.assertEqual(monitor.suggest_tile_size(256, 1024, 1024), 256)
            monitor.clear_cache()

    def test_gpu_query_failures_do_not_crash(self):
        cfg = PipelineConfig()
        cfg.device = "cuda:0"
        cfg.gpu.enabled = True
        fake_torch = _fake_torch(_FakeCudaFailing())

        with patch.dict("sys.modules", {"torch": fake_torch}):
            monitor = GPUMonitor(cfg)
            usage = monitor.get_vram_usage()
            self.assertEqual(
                usage,
                {"allocated_mb": 0, "reserved_mb": 0, "max_mb": 0, "free_mb": 0},
            )
            self.assertFalse(monitor.check_oom_risk(needed_mb=1))
            monitor.clear_cache()

    def test_cpu_device_does_not_import_torch(self):
        cfg = PipelineConfig()
        cfg.device = "cpu"
        cfg.gpu.enabled = True

        import builtins

        original_import = builtins.__import__

        def _guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("torch import must not be attempted for CPU monitor setup")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_guarded_import):
            monitor = GPUMonitor(cfg)
            self.assertFalse(monitor.available)
