"""Command-line interface for the asset pipeline."""

import argparse
import atexit
import logging
import os
import re
import signal
import sys

from .config import PipelineConfig
from .core import setup_logging, CheckpointManager

logger = logging.getLogger("asset_pipeline")


def main():
    """Parse CLI arguments, run the pipeline, and handle graceful shutdown."""
    parser = argparse.ArgumentParser(
        description="Game Asset Visual Upgrade Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  AssetBrew --input ./textures --output ./textures_hq
  AssetBrew --config config.yaml
  AssetBrew -i ./textures --phases upscale,pbr,normal
  AssetBrew -i ./textures --dry-run
  AssetBrew --generate-config
  AssetBrew -i ./textures --reset-checkpoint
        """
    )
    parser.add_argument("--input", "-i", help="Input assets directory")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--config", "-c", help="Path to config YAML")
    parser.add_argument("--phases", "-p",
                        help="Comma-separated: upscale,pbr,normal,pom,mipmap,postprocess,validate")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--ui", action="store_true",
                        help="Launch desktop UI")
    parser.add_argument("--workers", type=int, help="Max parallel workers")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate default config.yaml")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Clear checkpoint and reprocess everything")
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()

    if args.ui:
        from .ui.app import main as ui_main
        logger.info("Launching desktop UI...")
        ui_main()
        return

    if args.generate_config:
        config = PipelineConfig()
        dest = args.config or args.output or "config.yaml"
        if os.path.isdir(dest):
            dest = os.path.join(dest, "config.yaml")
        config.to_yaml(dest)
        logger.info("Generated default %s", dest)
        print(f"Generated default {dest}")
        return

    # Ensure early validation warnings from from_yaml() are visible on stderr
    # before the full file-based logging is configured.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Load config
    if args.config:
        if not os.path.exists(args.config):
            logger.error("Config file not found: %s", args.config)
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        try:
            config = PipelineConfig.from_yaml(args.config)
        except ValueError as e:
            logger.error("Invalid config file '%s': %s", args.config, e)
            print(f"Error: Invalid config: {e}")
            sys.exit(1)
    else:
        config = PipelineConfig()

    # CLI overrides
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.device:
        if not re.fullmatch(r"(auto|cpu|cuda(?::\d+)?)", args.device):
            print(
                "Error: --device must be one of auto, cpu, cuda, or cuda:N "
                f"(got '{args.device}')"
            )
            logger.error("Invalid --device value '%s'", args.device)
            sys.exit(1)
        config.device = args.device
    if args.workers is not None:
        config.max_workers = args.workers
    if args.dry_run:
        config.dry_run = True
    if args.log_level:
        config.log_level = args.log_level

    if not config.input_dir or not os.path.isdir(config.input_dir):
        logger.error("Input directory invalid or not found: %s", config.input_dir)
        print(f"Error: Input directory not found: {config.input_dir}")
        sys.exit(1)

    log_file = os.path.join(config.output_dir, "pipeline.log")
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.log_level, log_file)

    # Validate config
    try:
        config.validate()
        config.apply_runtime_fixups()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Handle checkpoint reset
    if args.reset_checkpoint:
        cm = CheckpointManager(config)
        cm.clear()
        logger.info("Checkpoint cleared -- all assets will be reprocessed.")

    # Parse phases
    phases = None
    if args.phases:
        phases = [p.strip() for p in args.phases.split(",")]
        valid = {"scan", "upscale", "pbr", "normal", "pom", "mipmap", "postprocess", "validate"}
        invalid = set(phases) - valid
        if invalid:
            print(f"Error: Invalid phases: {invalid}")
            print(f"Valid: {sorted(valid)}")
            logger.error("Invalid phase list requested: %s", sorted(invalid))
            sys.exit(1)

    from .pipeline import AssetPipeline
    pipeline = AssetPipeline(config)

    # Track whether checkpoint has already been saved during shutdown to
    # avoid redundant I/O (e.g. SIGTERM handler followed by atexit).
    _checkpoint_saved = False

    def _emergency_save():
        nonlocal _checkpoint_saved
        if _checkpoint_saved:
            return
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
        except Exception:
            pass  # Best-effort on exit

    atexit.register(_emergency_save)

    # Handle SIGTERM (e.g. from process managers)
    def _sigterm_handler(signum, frame):
        nonlocal _checkpoint_saved
        logger.warning("Received SIGTERM. Saving checkpoint...")
        if hasattr(pipeline, '_cancel_event'):
            pipeline._cancel_event.set()
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
        except Exception:
            pass
        sys.exit(143)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # Import pipeline-specific exceptions for clean error handling
    from .pipeline import (
        PhaseTimeoutError,
        PhaseFailureThresholdError,
        PipelineCancelledError,
    )

    # Graceful shutdown: save checkpoint on Ctrl+C
    try:
        pipeline.run(phases)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving checkpoint...")
        if hasattr(pipeline, '_cancel_event'):
            pipeline._cancel_event.set()
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
            logger.info("Checkpoint saved. Resume by re-running the same command.")
        except Exception as exc:
            logger.error("Failed to save checkpoint on interrupt: %s", exc)
        sys.exit(130)
    except PhaseTimeoutError as exc:
        logger.error("Pipeline aborted: %s", exc)
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
        except Exception:
            pass
        print(f"Error: Phase timed out: {exc}")
        sys.exit(2)
    except PhaseFailureThresholdError as exc:
        logger.error("Pipeline aborted: %s", exc)
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
        except Exception:
            pass
        print(f"Error: Too many failures: {exc}")
        sys.exit(3)
    except PipelineCancelledError as exc:
        logger.warning("Pipeline cancelled: %s", exc)
        try:
            pipeline.checkpoint.save()
            _checkpoint_saved = True
        except Exception:
            pass
        sys.exit(130)
    finally:
        # Unregister atexit handler to release pipeline reference
        atexit.unregister(_emergency_save)

    if getattr(pipeline, "_failed_assets", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
