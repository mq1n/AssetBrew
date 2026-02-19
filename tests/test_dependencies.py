"""Fail fast when mandatory runtime dependencies are missing."""

import importlib.util


def test_mandatory_cv2_installed() -> None:
    assert importlib.util.find_spec("cv2") is not None


def test_mandatory_scipy_installed() -> None:
    assert importlib.util.find_spec("scipy") is not None
