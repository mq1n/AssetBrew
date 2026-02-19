"""Tests for packaging and pyproject.toml correctness."""

import os
import unittest


class TestOnnxruntimeExtras(unittest.TestCase):
    def test_base_deps_no_onnxruntime(self):
        import tomllib
        toml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
        )
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        base_deps = data["project"]["dependencies"]
        onnx_in_base = any("onnxruntime" in d for d in base_deps)
        self.assertFalse(onnx_in_base)

    def test_gpu_extra_has_onnxruntime_gpu(self):
        import tomllib
        toml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
        )
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        gpu_deps = data["project"]["optional-dependencies"]["gpu"]
        has_gpu = any("onnxruntime-gpu" in d for d in gpu_deps)
        self.assertTrue(has_gpu)

    def test_gpu_extra_uses_torch_version_range(self):
        import tomllib
        toml_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
        )
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        gpu_deps = data["project"]["optional-dependencies"]["gpu"]
        torch_specs = [d for d in gpu_deps if d.startswith("torch")]
        self.assertTrue(any(">=" in d and "<" in d for d in torch_specs))

    def test_requirements_txt_no_onnxruntime(self):
        req_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "requirements.txt"
        )
        with open(req_path, "r", encoding="utf-8") as f:
            lines = [
                ln.strip()
                for ln in f.readlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        self.assertFalse(any("onnxruntime" in ln for ln in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
