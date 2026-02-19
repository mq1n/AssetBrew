"""Tests for checkpoint manager and file hashing."""

import json
import os
import shutil
import tempfile
import threading
import unittest
import warnings
from unittest import mock

from AssetBrew.config import PipelineConfig
from AssetBrew.core import CheckpointManager, file_hash, file_hash_md5, file_hash_sha256
from AssetBrew.core import checkpoint as checkpoint_module
from AssetBrew.core.checkpoint import config_fingerprint


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "test_checkpoint.json"
        )
        self.config.checkpoint.save_interval = 1

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mark_and_check(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("file1.png", "abc123", "upscale", {"upscaled": "/tmp/out.png"})
        self.assertTrue(cm.is_completed("file1.png", "abc123", ["upscale"]))
        self.assertFalse(cm.is_completed("file1.png", "abc123", ["pbr"]))

    def test_hash_mismatch(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("file1.png", "abc123", "upscale")
        self.assertFalse(cm.is_completed("file1.png", "different_hash", ["upscale"]))

    def test_save_and_reload(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("file1.png", "abc123", "upscale", {"path": "/out.png"})
        cm.save()
        cm2 = CheckpointManager(self.config)
        self.assertTrue(cm2.is_completed("file1.png", "abc123", ["upscale"]))

    def test_prune_stale(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("keep.png", "hash1", "upscale")
        cm.mark_completed("remove.png", "hash2", "upscale")
        cm.prune({"keep.png"})
        self.assertTrue(cm.is_completed("keep.png", "hash1", ["upscale"]))
        self.assertFalse(cm.is_completed("remove.png", "hash2", ["upscale"]))

    def test_prune_stale_persists_to_disk(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("keep.png", "hash1", "upscale")
        cm.mark_completed("remove.png", "hash2", "upscale")
        cm.save()

        cm.prune({"keep.png"})
        cm_reloaded = CheckpointManager(self.config)
        self.assertTrue(cm_reloaded.is_completed("keep.png", "hash1", ["upscale"]))
        self.assertFalse(cm_reloaded.is_completed("remove.png", "hash2", ["upscale"]))

    def test_clear(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("file.png", "hash", "upscale")
        cm.save()
        cm.clear()
        self.assertFalse(cm.is_completed("file.png", "hash", ["upscale"]))
        self.assertFalse(os.path.exists(self.config.checkpoint.checkpoint_path))

    def test_get_result(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("f.png", "h", "upscale", {"upscaled": "/path"})
        result = cm.get_result("f.png", "upscale")
        self.assertIsNotNone(result)
        self.assertEqual(result["upscaled"], "/path")

    def test_init_cleans_orphaned_temp_files(self):
        orphan = f"{self.config.checkpoint.checkpoint_path}.tmp.999"
        with open(orphan, "w", encoding="utf-8") as f:
            f.write("{}")
        self.assertTrue(os.path.exists(orphan))
        _ = CheckpointManager(self.config)
        self.assertFalse(os.path.exists(orphan))

    def test_thread_safety(self):
        cm = CheckpointManager(self.config)
        errors = []

        def worker(idx):
            try:
                for i in range(50):
                    fp = f"file_{idx}_{i}.png"
                    cm.mark_completed(fp, f"hash_{idx}_{i}", "upscale")
                    cm.is_completed(fp, f"hash_{idx}_{i}", ["upscale"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_save_uses_thread_specific_temp_path(self):
        cm = CheckpointManager(self.config)
        cm.mark_completed("file.png", "hash", "upscale")

        captured = {}
        real_replace = os.replace

        def _capture_replace(src, dst):
            captured["src"] = src
            return real_replace(src, dst)

        with mock.patch("os.replace", side_effect=_capture_replace):
            cm.save()

        self.assertIn(".tmp.", captured["src"])
        self.assertIn(str(os.getpid()), captured["src"])


class TestFileHash(unittest.TestCase):
    def test_consistent_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            h1 = file_hash(path)
            h2 = file_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
        finally:
            os.unlink(path)

    def test_missing_file_returns_unique_sentinel(self):
        h = file_hash("/nonexistent/path.png")
        self.assertTrue(h.startswith("unhashable-"), f"Expected 'unhashable-' prefix, got {h!r}")
        # Each call should produce a different sentinel (non-cacheable)
        h2 = file_hash("/nonexistent/path2.png")
        self.assertNotEqual(h, h2)

    def test_backward_compat_alias(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                old = file_hash_md5(path)
            new = file_hash(path)
            explicit = file_hash_sha256(path)
            self.assertEqual(old, new)
            self.assertEqual(explicit, new)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        finally:
            os.unlink(path)


class TestCheckpointStructuredResults(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "test_checkpoint.json"
        )
        self.config.checkpoint.save_interval = 1

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_preserved(self):
        cm = CheckpointManager(self.config)
        result = {
            "main": {
                "mips": ["/tmp/mip0.png", "/tmp/mip1.png", "/tmp/mip2.png"],
                "count": 3,
            }
        }
        cm.mark_completed("file.png", "hash", "mipmap", result)
        cm.save()
        cm2 = CheckpointManager(self.config)
        restored = cm2.get_result("file.png", "mipmap")
        self.assertIsNotNone(restored)
        self.assertEqual(restored["main"]["mips"],
                         ["/tmp/mip0.png", "/tmp/mip1.png", "/tmp/mip2.png"])

    def test_nested_dict_preserved(self):
        cm = CheckpointManager(self.config)
        result = {
            "normal": "/tmp/normal.png",
            "height": "/tmp/height.png",
            "stats": {"mean_z": 0.95, "valid_ratio": 0.99},
        }
        cm.mark_completed("file.png", "hash", "normal", result)
        cm.save()
        cm2 = CheckpointManager(self.config)
        restored = cm2.get_result("file.png", "normal")
        self.assertEqual(restored["stats"]["mean_z"], 0.95)


class TestCheckpointFingerprint(unittest.TestCase):
    def test_device_change_affects_fingerprint(self):
        cfg_cpu = PipelineConfig()
        cfg_cpu.device = "cpu"

        cfg_cuda = PipelineConfig()
        cfg_cuda.device = "cuda"
        cfg_cuda.upscale.half_precision = False

        self.assertNotEqual(
            config_fingerprint(cfg_cpu),
            config_fingerprint(cfg_cuda),
        )

    def test_checkpoint_schema_version_affects_fingerprint(self):
        cfg = PipelineConfig()
        fp1 = config_fingerprint(cfg)
        with mock.patch.object(
            checkpoint_module,
            "CHECKPOINT_SCHEMA_VERSION",
            checkpoint_module.CHECKPOINT_SCHEMA_VERSION + 1,
        ):
            fp2 = config_fingerprint(cfg)
        self.assertNotEqual(fp1, fp2)


class TestCorruptCheckpointRecovery(unittest.TestCase):
    """Verify CheckpointManager handles corrupt JSON gracefully."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "test_checkpoint.json"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_corrupt_checkpoint_recovery(self):
        # Write invalid JSON to checkpoint file
        with open(self.config.checkpoint.checkpoint_path, "w") as f:
            f.write("{invalid json content!!!}")

        # Should not raise; should start with empty state
        cm = CheckpointManager(self.config)
        self.assertFalse(cm.is_completed("any.png", "hash", ["upscale"]))


class TestCheckpointResumeAcrossRuns(unittest.TestCase):
    """Verify checkpoint resume works across separate manager instances."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "test_checkpoint.json"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_checkpoint_resume_across_runs(self):
        # Run 1: mark assets complete
        cm1 = CheckpointManager(self.config)
        cm1.mark_completed("asset1.png", "hash1", "upscale", {"upscaled": "/out/a1.png"})
        cm1.mark_completed("asset1.png", "hash1", "pbr", {"albedo": "/out/a1_albedo.png"})
        cm1.mark_completed("asset2.png", "hash2", "upscale", {"upscaled": "/out/a2.png"})
        cm1.save()

        # Run 2: new manager instance should see all completions
        cm2 = CheckpointManager(self.config)
        self.assertTrue(cm2.is_completed("asset1.png", "hash1", ["upscale"]))
        self.assertTrue(cm2.is_completed("asset1.png", "hash1", ["pbr"]))
        self.assertTrue(cm2.is_completed("asset2.png", "hash2", ["upscale"]))
        # Not completed phases should return False
        self.assertFalse(cm2.is_completed("asset2.png", "hash2", ["pbr"]))

        # Results should be preserved
        r1 = cm2.get_result("asset1.png", "upscale")
        self.assertIsNotNone(r1)
        self.assertEqual(r1["upscaled"], "/out/a1.png")


class TestCheckpointSchemaValidation(unittest.TestCase):
    """Regression: invalid checkpoint structure must be discarded, not crash.

    If a checkpoint file has valid JSON but an invalid schema (e.g.
    ``completed`` is a list instead of a dict), the old code would crash
    with a TypeError when iterating.  The fix adds _validate_schema() which
    rejects malformed structures on load, falling back to empty state.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = PipelineConfig()
        self.config.checkpoint.checkpoint_path = os.path.join(
            self.tmpdir, "test_checkpoint.json"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_completed_as_list_discarded(self):
        """``completed`` stored as list instead of dict should be discarded."""
        invalid_data = {
            "config_fingerprint": "abc",
            "completed": ["file1.png", "file2.png"],
        }
        with open(self.config.checkpoint.checkpoint_path, "w") as f:
            json.dump(invalid_data, f)

        cm = CheckpointManager(self.config)
        # Should start with empty state, not crash
        self.assertFalse(cm.is_completed("file1.png", "hash", ["upscale"]))

    def test_completed_entry_as_string_discarded(self):
        """An entry in ``completed`` that is a string instead of dict should be discarded."""
        invalid_data = {
            "config_fingerprint": "abc",
            "completed": {"file1.png": "just_a_string"},
        }
        with open(self.config.checkpoint.checkpoint_path, "w") as f:
            json.dump(invalid_data, f)

        cm = CheckpointManager(self.config)
        self.assertFalse(cm.is_completed("file1.png", "hash", ["upscale"]))

    def test_phases_as_dict_discarded(self):
        """``phases`` stored as dict instead of list should be discarded."""
        invalid_data = {
            "config_fingerprint": "abc",
            "completed": {
                "file1.png": {
                    "hash": "abc123",
                    "phases": {"upscale": True},
                    "results": {},
                }
            },
        }
        with open(self.config.checkpoint.checkpoint_path, "w") as f:
            json.dump(invalid_data, f)

        cm = CheckpointManager(self.config)
        self.assertFalse(cm.is_completed("file1.png", "abc123", ["upscale"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
