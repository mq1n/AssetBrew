"""Tests for optional-dependency compatibility shims."""

from AssetBrew.core.compat import tqdm


def test_tqdm_interface_supports_common_methods() -> None:
    bar = tqdm([])
    for name in ("update", "set_description", "set_postfix", "close"):
        assert hasattr(bar, name), f"tqdm object missing method: {name}"
