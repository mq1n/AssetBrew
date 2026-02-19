"""Entrypoint for `python -m AssetBrew`.

Usage:
  - CLI pipeline: `python -m AssetBrew [pipeline args]`
  - Desktop UI:   `python -m AssetBrew --ui`
"""
import sys
import logging

logger = logging.getLogger("asset_pipeline")

def _run_ui():
    from .ui.app import main as ui_main
    logger.debug("Dispatching to desktop UI entrypoint.")
    ui_main()


def _run_cli():
    from .cli import main as cli_main
    logger.debug("Dispatching to CLI entrypoint.")
    cli_main()


if __name__ == "__main__":
    if "--ui" in sys.argv:
        _run_ui()
    else:
        _run_cli()
