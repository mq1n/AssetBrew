"""Optional smoke tests for the PyQt desktop UI."""

import importlib
import importlib.util
import os
import unittest


HAS_PYQT6 = importlib.util.find_spec("PyQt6") is not None


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not installed")
class TestUISmoke(unittest.TestCase):
    def test_ui_module_imports(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        module = importlib.import_module("AssetBrew.ui.app")
        self.assertTrue(hasattr(module, "MainWindow"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
