"""Compatibility stubs for optional dependencies."""

import sys
import types
import logging

logger = logging.getLogger("asset_pipeline.compat")


def ensure_torchvision_functional_tensor_alias() -> bool:
    """Provide deprecated torchvision alias expected by BasicSR.

    BasicSR 1.4.2 imports:
      torchvision.transforms.functional_tensor
    while newer torchvision exposes:
      torchvision.transforms._functional_tensor
    """
    alias = "torchvision.transforms.functional_tensor"

    try:
        __import__(alias)
        logger.debug("Torchvision alias %s already available.", alias)
        return True
    except Exception:
        pass

    try:
        from torchvision.transforms import _functional_tensor as tv_functional_tensor
    except Exception:
        logger.debug(
            "Torchvision functional tensor compatibility aliases unavailable. "
            "Continuing without alias."
        )
        return False

    module = types.ModuleType(alias)
    for name in dir(tv_functional_tensor):
        setattr(module, name, getattr(tv_functional_tensor, name))
    sys.modules[alias] = module
    logger.debug("Installed torchvision alias compatibility module: %s", alias)
    return True


try:
    from tqdm import tqdm  # noqa: F401
except ImportError:
    class tqdm:
        """Minimal fallback when tqdm is not installed."""

        def __init__(self, iterable=None, **kwargs):
            """Store iterable argument to mimic tqdm(iterable)."""
            self._iterable = iterable
            self.total = kwargs.get("total", None)
            self.n = 0

        def __iter__(self):
            """Iterate over wrapped iterable or an empty iterator."""
            return iter(self._iterable) if self._iterable is not None else iter([])

        def __enter__(self):
            """Return self to support context-manager usage."""
            return self

        def __exit__(self, *args):
            """No-op context-manager exit."""
            pass

        def update(self, n=1):
            """Accept progress updates and track count."""
            self.n += n

        def set_description(self, desc=None, refresh=True):
            """Compatibility no-op for tqdm.set_description()."""
            pass

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            """Compatibility no-op for tqdm.set_postfix()."""
            pass

        def close(self):
            """Compatibility no-op for tqdm.close()."""
            pass
