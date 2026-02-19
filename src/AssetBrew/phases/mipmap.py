"""Generate high-quality mipmap chains for processed textures.

This phase applies type-aware downsampling, optional roughness compensation,
and optional sharpening before writing per-level mip outputs.
"""

import logging
import os
import math
import subprocess
import sys
import time
from typing import Optional
from uuid import uuid4

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord, load_image, save_image, ensure_rgb,
    get_intermediate_path, luminance_bt709, srgb_to_linear, linear_to_srgb
)

logger = logging.getLogger("asset_pipeline.mipmap")

# Known Windows NTSTATUS crash codes (as signed int32)
_CRASH_CODES_WIN = {
    -1073741819: "ACCESS_VIOLATION (0xC0000005)",
    -1073741795: "ILLEGAL_INSTRUCTION (0xC000001D)",
    -1073740791: "STACK_BUFFER_OVERRUN (0xC0000409)",
    -1073741571: "STACK_OVERFLOW (0xC00000FD)",
    -2147483645: "BREAKPOINT (0x80000003)",
    -1073741515: "DLL_NOT_FOUND (0xC0000135)",
    -1073741511: "ENTRYPOINT_NOT_FOUND (0xC0000139)",
    -1073741502: "DLL_INIT_FAILED (0xC0000142)",
}


def _is_crash_code(returncode: int) -> Optional[str]:
    """Return a human-readable crash description, or None if not a crash."""
    if sys.platform == "win32":
        desc = _CRASH_CODES_WIN.get(returncode)
        if desc:
            return desc
        if returncode < 0:
            return f"NTSTATUS 0x{returncode & 0xFFFFFFFF:08X}"
        return None
    # Unix: negative returncode means killed by signal
    if returncode < 0:
        sig_num = -returncode
        try:
            import signal
            return f"{signal.Signals(sig_num).name} (signal {sig_num})"
        except (ValueError, AttributeError):
            return f"signal {sig_num}"
    return None


def _forward_output(text: str, tool_label: str, stream_name: str,
                    level: int, max_lines: int = 120) -> None:
    """Log subprocess output line-by-line at the given level."""
    if not text or not text.strip():
        return
    lines = text.splitlines()
    if len(lines) > max_lines:
        omitted = len(lines) - max_lines
        logger.log(level, "[%s] ... %d earlier %s lines omitted",
                   tool_label, omitted, stream_name)
        lines = lines[-max_lines:]
    for line in lines:
        if len(line) > 500:
            line = line[:500] + "..."
        logger.log(level, "[%s] %s: %s", tool_label, stream_name, line)


class MipmapGenerator:
    """Generate high-quality mipmap chains with per-type filtering."""

    def __init__(self, config: PipelineConfig):
        """Initialize mipmap generator with runtime configuration."""
        self.config = config
        self.cfg = config.mipmap
        self._compression_tool_path: Optional[str] = None
        self._compression_tool_resolved = False
        self._ktx2_tool_path: Optional[str] = None
        self._ktx2_tool_resolved = False

    @staticmethod
    def _make_local_temp_dir(base_dir: str, prefix: str = "tmp_") -> str:
        """Create a writable temp directory without relying on tempfile ACL quirks."""
        os.makedirs(base_dir, exist_ok=True)
        for _ in range(256):
            candidate = os.path.join(base_dir, f"{prefix}{uuid4().hex}")
            try:
                os.makedirs(candidate, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"Unable to allocate temp directory under {base_dir}")

    def process(
        self,
        record: AssetRecord,
        source_path: str,
        tex_type_override: TextureType = None,
        name_tag: Optional[str] = None,
    ) -> dict:
        """Build and persist mip levels for a source texture asset."""
        result = {"mips": [], "error": None}
        tex_type = tex_type_override or TextureType(record.texture_type)
        tag = (name_tag or tex_type.value or "map").lower()
        safe_tag = "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_") or "map"

        if not os.path.exists(source_path):
            result["error"] = f"Source not found: {source_path}"
            logger.warning(
                "Mipmap generation skipped for %s: source missing (%s).",
                record.filename, source_path
            )
            return result

        try:
            img = load_image(source_path, max_pixels=self.config.max_image_pixels)

            # Keep grayscale maps as single-channel to avoid RGB inflation
            # (important for BC4/BC5 compression and 1/3 file size)
            _GRAYSCALE_TYPES = {
                TextureType.HEIGHT, TextureType.ROUGHNESS,
                TextureType.METALNESS, TextureType.AO,
                TextureType.MASK, TextureType.OPACITY,
            }
            if tex_type in _GRAYSCALE_TYPES and img.ndim == 3:
                img = luminance_bt709(img, assume_srgb=False)

            h, w = img.shape[:2]
            max_dim = max(h, w)
            total_mips = int(math.log2(max_dim)) + 1

            # Limit mip count: stop when smallest dim <= 2^min_resident_mips
            # e.g. min_resident_mips=3 stops at 8x8 (level where min dim is 8)
            min_mip_dim = max(1 << self.cfg.min_resident_mips, 1)
            mip_count = total_mips
            for lvl in range(total_mips):
                if max(h >> lvl, 1) < min_mip_dim and max(w >> lvl, 1) < min_mip_dim:
                    mip_count = lvl
                    break

            mip_paths = []

            for level in range(mip_count):
                mip_h = max(h >> level, 1)
                mip_w = max(w >> level, 1)

                if level == 0:
                    mip = img
                else:
                    # Always downsample from original (mip 0) to avoid
                    # progressive quality loss from cascade downscaling.
                    mip = self._generate_mip(img, mip_w, mip_h, tex_type, level)

                if tex_type == TextureType.NORMAL and self.cfg.renormalize_normals:
                    mip = self._renormalize_normal_mip(mip)

                if tex_type == TextureType.ROUGHNESS and self.cfg.increase_roughness_per_mip:
                    mip = self._increase_roughness(mip, level)

                if self.cfg.sharpen_mips and level in self.cfg.sharpen_levels:
                    if tex_type not in (
                        TextureType.NORMAL, TextureType.HEIGHT, TextureType.MASK,
                        TextureType.ROUGHNESS, TextureType.METALNESS,
                        TextureType.AO, TextureType.ORM,
                    ):
                        mip = self._sharpen(mip)

                suffix = f"_{safe_tag}_mip{level}"
                out_path = get_intermediate_path(
                    record.filepath, "05_mipmaps",
                    self.config.intermediate_dir, suffix=suffix, ext=".png"
                )
                # 16-bit for precision-critical types (avoids banding)
                mip_bits = 16 if tex_type in (
                    TextureType.HEIGHT, TextureType.ROUGHNESS,
                    TextureType.METALNESS, TextureType.AO,
                    TextureType.NORMAL,
                ) else 8
                save_image(mip, out_path, bits=mip_bits)
                mip_paths.append({
                    "level": level, "width": mip_w,
                    "height": mip_h, "path": out_path
                })

                if mip_w <= 1 and mip_h <= 1:
                    break

            result["mips"] = mip_paths
            logger.info(
                f"Generated {len(mip_paths)} mip levels for {record.filename} "
                f"(type={tex_type.value})"
            )

        except Exception as e:
            logger.error(f"Mipmap generation failed for {record.filename}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _generate_mip(self, source: np.ndarray, target_w: int, target_h: int,
                      tex_type: TextureType, level: int) -> np.ndarray:
        if tex_type in (TextureType.MASK, TextureType.OPACITY):
            interp = cv2.INTER_NEAREST
        elif tex_type == TextureType.NORMAL:
            # Downsample normals in vector space, not encoded [0,1] color space.
            rgb = ensure_rgb(np.clip(source, 0, 1).astype(np.float32))
            src = rgb * 2.0 - 1.0
            curr_h, curr_w = src.shape[:2]
            while curr_w > target_w * 2 or curr_h > target_h * 2:
                next_w = max(curr_w // 2, target_w)
                next_h = max(curr_h // 2, target_h)
                src = cv2.resize(src, (next_w, next_h), interpolation=cv2.INTER_AREA)
                src = self._renormalize_normal_decoded(src)
                curr_w, curr_h = next_w, next_h
            src = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA)
            src = self._renormalize_normal_decoded(src)
            return np.clip(src * 0.5 + 0.5, 0, 1).astype(np.float32)
        else:
            filter_map = {
                "lanczos": cv2.INTER_LANCZOS4,
                "area": cv2.INTER_AREA,
                "cubic": cv2.INTER_CUBIC,
                "linear": cv2.INTER_LINEAR,
                "bilinear": cv2.INTER_LINEAR,
                "nearest": cv2.INTER_NEAREST,
                "box": cv2.INTER_AREA,  # INTER_AREA is the box filter equivalent
            }
            interp = filter_map.get(
                self.cfg.filter_method, cv2.INTER_LANCZOS4
            )

        # Work in float32 throughout to avoid 8-bit quantization/banding
        src = np.clip(source, 0, 1).astype(np.float32)
        # Use the config's srgb_texture_types rather than a hardcoded set so
        # users can remove e.g. "specular" if their workflow uses linear data.
        srgb_types = set(self.config.compression.srgb_texture_types)
        use_srgb_downsampling = (
            bool(getattr(self.cfg, "srgb_downsampling", True))
            and tex_type.value in srgb_types
            and src.ndim == 3
            and src.shape[2] >= 3
        )
        if use_srgb_downsampling:
            linear_rgb = srgb_to_linear(src[:, :, :3])
            if src.shape[2] > 3:
                src = np.concatenate([linear_rgb, src[:, :, 3:]], axis=2)
            else:
                src = linear_rgb

        # Progressive downsample to avoid aliasing on large jumps
        curr_h, curr_w = src.shape[:2]
        while curr_w > target_w * 2 or curr_h > target_h * 2:
            next_w = max(curr_w // 2, target_w)
            next_h = max(curr_h // 2, target_h)
            src = cv2.resize(src, (next_w, next_h), interpolation=interp)
            curr_w, curr_h = next_w, next_h

        resized = cv2.resize(src, (target_w, target_h), interpolation=interp)
        if use_srgb_downsampling and resized.ndim == 3 and resized.shape[2] >= 3:
            srgb_rgb = linear_to_srgb(np.clip(resized[:, :, :3], 0.0, 1.0))
            if resized.shape[2] > 3:
                resized = np.concatenate([srgb_rgb, resized[:, :, 3:]], axis=2)
            else:
                resized = srgb_rgb
        return np.clip(resized, 0, 1).astype(np.float32)

    def _renormalize_normal_mip(self, mip: np.ndarray) -> np.ndarray:
        rgb = ensure_rgb(mip)
        decoded = rgb * 2.0 - 1.0
        decoded = self._renormalize_normal_decoded(decoded)
        return (decoded * 0.5 + 0.5).astype(np.float32)

    @staticmethod
    def _renormalize_normal_decoded(decoded: np.ndarray) -> np.ndarray:
        length = np.sqrt(np.sum(decoded**2, axis=-1, keepdims=True))
        length = np.maximum(length, 1e-8)
        return (decoded / length).astype(np.float32)

    def _increase_roughness(self, mip: np.ndarray, level: int) -> np.ndarray:
        # Work in roughness² (perceptual / GGX alpha) space per PBR convention.
        # Linear addition in α=r² space gives perceptually correct mip roughness
        # increase: new_r = sqrt(r² + delta*level).
        increase = self.cfg.roughness_mip_increase * (2.0 ** level - 1.0)
        r_sq = mip ** 2
        new_r = np.sqrt(np.clip(r_sq + increase, 0, 1))
        return new_r.astype(np.float32)

    def _sharpen(self, mip: np.ndarray) -> np.ndarray:
        blurred = gaussian_filter(mip, sigma=self.cfg.sharpen_radius)
        sharpened = mip + self.cfg.sharpen_strength * (mip - blurred)
        return np.clip(sharpened, 0, 1).astype(np.float32)

    @staticmethod
    def _is_transient_tool_failure(text: str, returncode: int) -> bool:
        """Return True when the failure likely came from temporary I/O contention."""
        msg = (text or "").lower()
        if "format is unsupported" in msg or "mipset to qt format is not supported" in msg:
            return False
        transient_markers = (
            "sharing violation",
            "being used by another process",
            "temporarily unavailable",
            "resource busy",
            "timed out",
            "access is denied",
            "permission denied",
        )
        if any(marker in msg for marker in transient_markers):
            return True
        # Often used by tools for generic failures where stderr contains lock issues.
        return returncode in (1, 2) and ("lock" in msg or "busy" in msg)

    def _run_tool(self, cmd: list, tool_label: str, source_info: str,
                  timeout: int = 120) -> tuple:
        """Run an external CLI tool with output forwarding and crash detection.

        Returns (success: bool, proc: CompletedProcess | None).
        """
        logger.debug("Running %s: %s", tool_label, " ".join(cmd))
        max_attempts = 3
        last_proc = None
        for attempt in range(1, max_attempts + 1):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=timeout, text=True,
                    encoding="utf-8", errors="replace",
                )
            except FileNotFoundError:
                logger.error("%s tool not found: %s", tool_label, cmd[0])
                return (False, None)
            except PermissionError:
                logger.error(
                    "%s tool is not executable: %s", tool_label, cmd[0]
                )
                return (False, None)
            except subprocess.TimeoutExpired as e:
                logger.error("%s timed out after %ds for %s",
                             tool_label, timeout, source_info)
                stdout_text = ""
                stderr_text = ""
                for stream, output in [("stdout", e.stdout), ("stderr", e.stderr)]:
                    if output:
                        text = (
                            output
                            if isinstance(output, str)
                            else output.decode(errors="replace")
                        )
                        if stream == "stdout":
                            stdout_text = text
                        else:
                            stderr_text = text
                        _forward_output(text, tool_label, stream, logging.ERROR, max_lines=10)
                if attempt < max_attempts and self._is_transient_tool_failure(
                    f"{stdout_text}\n{stderr_text}", returncode=1
                ):
                    delay = 0.3 * attempt
                    logger.warning(
                        "%s retrying after transient timeout (%s), attempt %d/%d in %.1fs",
                        tool_label, source_info, attempt + 1, max_attempts, delay
                    )
                    time.sleep(delay)
                    continue
                return (False, None)

            last_proc = proc
            if proc.returncode == 0:
                # Surface successful tool progress/details in normal console logs.
                _forward_output(
                    proc.stdout, tool_label, "stdout", logging.INFO, max_lines=200
                )
                _forward_output(
                    proc.stderr, tool_label, "stderr", logging.INFO, max_lines=200
                )
                return (True, proc)

            # Failure
            _forward_output(proc.stdout, tool_label, "stdout", logging.ERROR)
            _forward_output(proc.stderr, tool_label, "stderr", logging.ERROR)
            crash = _is_crash_code(proc.returncode)
            if crash:
                logger.error(
                    "%s crashed processing %s: %s (exit code %d)",
                    tool_label, source_info, crash, proc.returncode
                )
            else:
                logger.error(
                    "%s failed for %s with exit code %d",
                    tool_label, source_info, proc.returncode
                )

            merged_output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if (
                attempt < max_attempts
                and not crash
                and self._is_transient_tool_failure(merged_output, proc.returncode)
            ):
                delay = 0.3 * attempt
                logger.warning(
                    "%s retrying after transient failure (%s), attempt %d/%d in %.1fs",
                    tool_label, source_info, attempt + 1, max_attempts, delay
                )
                time.sleep(delay)
                continue
            break

        return (False, last_proc)

    def _resolve_compression_tool(self) -> Optional[str]:
        """Resolve and cache the compression tool path."""
        if self._compression_tool_resolved:
            return self._compression_tool_path

        self._compression_tool_resolved = True
        tool_path = self.config.compression.tool_path
        tool_name = self.config.compression.tool

        if not tool_path:
            import shutil as _shutil
            if tool_name == "compressonator":
                tool_path = _shutil.which("compressonatorcli")
            elif tool_name == "texconv":
                tool_path = _shutil.which("texconv")

        # Fallback: check bundled tools in bin/ (platform-aware)
        if not tool_path:
            import platform as _platform
            from .. import BIN_DIR
            bin_dir = BIN_DIR
            is_windows = _platform.system() == "Windows"
            exe_suffix = ".exe" if is_windows else ""
            candidates = []
            if tool_name == "compressonator":
                # Check platform-specific bundled directories (any version)
                for cmp_dir in sorted(bin_dir.glob("compressonatorcli*"), reverse=True):
                    if cmp_dir.is_dir():
                        candidates.append(cmp_dir / f"compressonatorcli{exe_suffix}")
                candidates.append(bin_dir / f"compressonatorcli{exe_suffix}")
            elif tool_name == "texconv":
                candidates.append(bin_dir / f"texconv{exe_suffix}")
            for candidate in candidates:
                if candidate.is_file():
                    tool_path = str(candidate)
                    break
            if tool_path:
                logger.info(f"Using bundled compression tool: {tool_path}")

        if not tool_path:
            logger.warning(
                f"Compression tool '{tool_name}' not found. "
                f"Set compression.tool_path in config or install it on PATH. "
                f"Skipping DDS generation."
            )
        self._compression_tool_path = tool_path
        return tool_path

    def _compression_timeout(self) -> int:
        """Resolve external compression timeout in seconds."""
        timeout = getattr(self.config.compression, "tool_timeout_seconds", 60)
        try:
            timeout_int = int(timeout)
        except (TypeError, ValueError):
            timeout_int = 60
        return max(1, timeout_int)

    def _compressonator_num_threads(self) -> int:
        """Resolve Compressonator thread count (0 means auto)."""
        configured = int(
            max(0, getattr(self.config.compression, "compressonator_num_threads", 0))
        )
        return min(configured, 128)

    def generate_dds(self, source_path: str, output_path: str,
                     compression: str = "bc7", srgb: bool = False) -> bool:
        """Convert a single PNG to compressed DDS using external tool.

        Supports both Compressonator CLI and Microsoft texconv.
        When *srgb* is True the output file is flagged as sRGB (e.g.
        ``BC7_UNORM_SRGB`` instead of ``BC7_UNORM``).
        Returns True on success, False on failure.
        """
        tool_path = self._resolve_compression_tool()
        if not tool_path:
            logger.warning(
                "Skipping DDS generation for %s: compression tool unavailable.",
                source_path
            )
            return False

        tool_name = self.config.compression.tool

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        quality = self.config.compression.quality
        timeout = self._compression_timeout()

        if tool_name == "texconv":
            out_dir = os.path.dirname(output_path) or "."
            cmd = [
                tool_path,
                "-f", compression.upper(),
                "-o", out_dir,
                "-y",
            ]
            if srgb:
                cmd.append("-sRGB")
            # texconv uses -bc with quality flag: q (quick), d (default),
            # u (ultra).  Map [0, 1] quality to one of these.
            if quality <= 0.33:
                cmd.extend(["-bc", "q"])
            elif quality >= 0.9:
                cmd.extend(["-bc", "u"])
            # else: omit for texconv default quality
            cmd.append(source_path)
        else:
            # Compressonator CLI (default).
            # Do NOT append _SRGB to the format name — Compressonator
            # rejects BC7_SRGB/BC3_SRGB as invalid format codes.
            # sRGB is handled after compression by patching the DX10
            # header DXGI format code (see _patch_dds_srgb).
            fmt = compression.upper()
            cmd = [
                tool_path,
                "-fd", fmt,
            ]
            encode_with = (
                getattr(self.config.compression, "compressonator_encode_with", "hpc")
                .strip()
                .lower()
            )
            if fmt in {"BC6H", "BC7"} and encode_with:
                cmd.extend(["-EncodeWith", encode_with.upper()])
                cmd.extend(["-NumThreads", str(self._compressonator_num_threads())])

            perf = getattr(self.config.compression, "compressonator_performance", None)
            if perf is not None and fmt in {"BC6H", "BC7"}:
                cmd.extend(["-Performance", f"{float(perf):.2f}"])

            # Compressonator quality: 0.0 (fastest) to 1.0 (best)
            cmd.extend(["-Quality", f"{quality:.2f}"])
            if bool(getattr(self.config.compression, "compressonator_no_progress", True)):
                cmd.append("-noprogress")
            cmd.extend([source_path, output_path])

        ok, _ = self._run_tool(cmd, tool_name, source_path, timeout=timeout)
        if not ok:
            return False

        # texconv writes {out_dir}/{input_stem}.dds, not the exact
        # output_path we requested.  Rename if necessary.
        if tool_name == "texconv":
            import pathlib as _pl
            src_stem = _pl.Path(source_path).stem
            actual = os.path.join(out_dir, f"{src_stem}.DDS")
            if not os.path.exists(actual):
                actual = os.path.join(out_dir, f"{src_stem}.dds")
            if actual != output_path and os.path.exists(actual):
                os.replace(actual, output_path)

        # For Compressonator: patch the DX10 header to sRGB after compression
        # since Compressonator doesn't accept _SRGB format suffixes.
        if srgb and tool_name != "texconv":
            self._patch_dds_srgb(output_path)

        # Some encoders emit incomplete DDS flags/caps. Normalize to keep
        # strict DDS parsers compatible with generated outputs.
        self._normalize_dds_header(output_path)

        logger.debug(f"DDS created: {output_path}")
        return True

    def generate_dds_mipchain(self, mip_paths: list, output_path: str,
                              compression: str = "bc7",
                              srgb: bool = False) -> "tuple[bool, bool]":
        """Generate a single DDS with all precomputed mip levels packed in.

        Compresses each mip level individually to a temp DDS, then assembles
        all compressed mip data into one DDS file with a proper mipmap chain.
        Game engines will load this as a single texture with pre-built mips,
        preserving the quality mip processing from Phase 5 (renormalization,
        roughness boosting, sharpening).

        Returns (success, degraded) tuple.  ``success`` is True when *any*
        DDS was written (full chain or base-only fallback).  ``degraded``
        is True when the full mipchain could not be assembled and the output
        fell back to a base-only DDS.
        """
        if not mip_paths:
            logger.warning(
                "Skipping DDS mipchain generation for %s: no mip levels provided.",
                output_path
            )
            return (False, False)

        tool_path = self._resolve_compression_tool()
        if not tool_path:
            logger.warning(
                "Skipping DDS mipchain generation for %s: compression tool unavailable.",
                output_path
            )
            return (False, False)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if len(mip_paths) <= 1:
            ok = self.generate_dds(mip_paths[0]["path"], output_path, compression, srgb=srgb)
            return (ok, False)

        # Compress each mip to temp DDS files under a project-local path.
        import shutil

        temp_root = os.path.join(
            self.config.intermediate_dir,
            "_dds_mipchain_tmp",
        )
        os.makedirs(temp_root, exist_ok=True)
        temp_dir = self._make_local_temp_dir(temp_root, prefix="dds_mips_")

        try:
            temp_dds = []
            for mip_info in mip_paths:
                level = mip_info["level"]
                mip_src = mip_info["path"]
                mip_dst = os.path.join(temp_dir, f"mip{level}.dds")
                if not self.generate_dds(mip_src, mip_dst, compression, srgb=srgb):
                    logger.warning(
                        f"Mip {level} compression failed, "
                        f"falling back to base-only DDS"
                    )
                    if self.generate_dds(
                        mip_paths[0]["path"], output_path, compression, srgb=srgb
                    ):
                        return (True, True)
                    return (False, False)
                temp_dds.append(mip_dst)

            # Assemble into single multi-mip DDS
            ok = self._assemble_dds_mipchain(temp_dds, output_path, srgb=srgb)
            if ok:
                self._normalize_dds_header(output_path)
                logger.debug(
                    f"Assembled {len(temp_dds)}-level mipchain DDS: "
                    f"{output_path}"
                )
                return (True, False)

            logger.warning(
                "DDS mipchain assembly failed for %s; falling back to base-only DDS",
                output_path,
            )
            if self.generate_dds(mip_paths[0]["path"], output_path, compression, srgb=srgb):
                return (True, True)
            return (False, False)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _patch_dds_srgb(self, dds_path: str) -> bool:
        """Patch a DDS file's DX10 header to use the sRGB DXGI format variant.

        Compressonator does not accept ``_SRGB`` format suffixes, so we
        compress with the UNORM variant and patch the DXGI format code
        afterwards.  Returns True on success.
        """
        import struct

        try:
            with open(dds_path, "r+b") as f:
                magic = f.read(4)
                if magic != b"DDS ":
                    return False
                header = f.read(124)
                if len(header) < 124:
                    return False
                pf_fourcc = header[80:84]
                if pf_fourcc != b"DX10":
                    # Legacy header — no DXGI format to patch. The sRGB flag
                    # cannot be set in legacy DDS headers, but the image data
                    # is still correct; the engine will just interpret it as
                    # linear. Log a warning and accept.
                    logger.debug(
                        "DDS at %s uses legacy header (no DX10); cannot "
                        "patch sRGB flag. Image data is still correct.",
                        dds_path,
                    )
                    return True
                dx10 = bytearray(f.read(20))
                if len(dx10) < 4:
                    return False
                dxgi_fmt = struct.unpack_from("<I", dx10, 0)[0]
                srgb_fmt = self._DXGI_SRGB_MAP.get(dxgi_fmt)
                if srgb_fmt is not None:
                    struct.pack_into("<I", dx10, 0, srgb_fmt)
                    f.seek(128)  # start of DX10 header
                    f.write(bytes(dx10))
                    logger.debug(
                        "Patched DDS sRGB: DXGI %d → %d for %s",
                        dxgi_fmt, srgb_fmt, dds_path,
                    )
        except Exception as exc:
            logger.warning("Failed to patch DDS sRGB header for %s: %s", dds_path, exc)
            return False
        return True

    # DXGI UNORM → SRGB format variant mapping
    _DXGI_SRGB_MAP = {
        71: 72,   # BC1_UNORM → BC1_UNORM_SRGB
        74: 75,   # BC2_UNORM → BC2_UNORM_SRGB
        77: 78,   # BC3_UNORM → BC3_UNORM_SRGB
        98: 99,   # BC7_UNORM → BC7_UNORM_SRGB
    }

    # Block size (bytes per 4x4 block) for DDS block-compressed formats.
    _DDS_BLOCK_BYTES_DXGI = {
        71: 8, 72: 8,    # BC1
        74: 16, 75: 16,  # BC2
        77: 16, 78: 16,  # BC3
        80: 8, 81: 8,    # BC4
        83: 16, 84: 16,  # BC5
        95: 16, 96: 16,  # BC6H
        98: 16, 99: 16,  # BC7
    }
    _DDS_BLOCK_BYTES_FOURCC = {
        b"DXT1": 8,
        b"DXT3": 16,
        b"DXT5": 16,
        b"ATI1": 8,
        b"BC4U": 8,
        b"BC4S": 8,
        b"ATI2": 16,
        b"BC5U": 16,
        b"BC5S": 16,
    }

    def _normalize_dds_header(self, dds_path: str) -> bool:
        """Normalize DDS header flags/caps for stricter loader compatibility."""
        import struct

        try:
            with open(dds_path, "r+b") as f:
                magic = f.read(4)
                if magic != b"DDS ":
                    return False
                header = bytearray(f.read(124))
                if len(header) < 124:
                    return False

                # Required fixed sizes for valid DDS header blocks.
                struct.pack_into("<I", header, 0, 124)   # dwSize
                struct.pack_into("<I", header, 72, 32)   # ddspf.dwSize

                width = struct.unpack_from("<I", header, 12)[0]
                height = struct.unpack_from("<I", header, 8)[0]
                mip_count = struct.unpack_from("<I", header, 24)[0]
                if mip_count < 1:
                    mip_count = 1
                    struct.pack_into("<I", header, 24, mip_count)

                fourcc = bytes(header[80:84])
                dxgi = None
                if fourcc == b"DX10":
                    dx10 = f.read(20)
                    if len(dx10) >= 4:
                        dxgi = struct.unpack_from("<I", dx10, 0)[0]

                flags = struct.unpack_from("<I", header, 4)[0]
                # Required DDSD bits for strict validators.
                flags |= (0x1 | 0x2 | 0x4 | 0x1000)  # CAPS|HEIGHT|WIDTH|PIXELFORMAT
                if mip_count > 1:
                    flags |= 0x00020000  # DDSD_MIPMAPCOUNT

                block_bytes = None
                if fourcc == b"DX10" and dxgi is not None:
                    block_bytes = self._DDS_BLOCK_BYTES_DXGI.get(dxgi)
                else:
                    block_bytes = self._DDS_BLOCK_BYTES_FOURCC.get(fourcc)
                if block_bytes is not None and width > 0 and height > 0:
                    blocks_w = max(1, (width + 3) // 4)
                    blocks_h = max(1, (height + 3) // 4)
                    linear_size = blocks_w * blocks_h * block_bytes
                    struct.pack_into("<I", header, 16, linear_size)
                    flags |= 0x00080000  # DDSD_LINEARSIZE
                    flags &= ~0x00000008  # clear DDSD_PITCH for BCn

                struct.pack_into("<I", header, 4, flags)

                caps = struct.unpack_from("<I", header, 104)[0]
                caps |= 0x00001000  # DDSCAPS_TEXTURE
                if mip_count > 1:
                    caps |= (0x00000008 | 0x00400000)  # COMPLEX | MIPMAP
                struct.pack_into("<I", header, 104, caps)

                f.seek(4)
                f.write(bytes(header))
            return True
        except Exception as exc:
            logger.warning("Failed to normalize DDS header for %s: %s", dds_path, exc)
            return False

    @staticmethod
    def _assemble_dds_mipchain(dds_paths: list, output_path: str,
                               srgb: bool = False) -> bool:
        """Assemble individually compressed DDS files into one multi-mip DDS.

        Reads the header from the base (mip 0) DDS, updates it to declare
        a mipmap chain, then concatenates the compressed pixel data from
        each mip level in order.  Handles both legacy and DX10-extended
        DDS headers.
        """
        import struct

        DDSD_CAPS = 0x1
        DDSD_HEIGHT = 0x2
        DDSD_WIDTH = 0x4
        DDSD_PIXELFORMAT = 0x1000
        DDSD_MIPMAPCOUNT = 0x20000
        DDSD_LINEARSIZE = 0x80000
        DDSCAPS_TEXTURE = 0x1000
        DDSCAPS_COMPLEX = 0x8
        DDSCAPS_MIPMAP = 0x400000

        def _read_dds_parts(path: str) -> Optional[dict]:
            with open(path, "rb") as f:
                raw = f.read()
            if len(raw) < 128:
                logger.error("DDS too small for header: %s", path)
                return None
            if raw[:4] != b"DDS ":
                logger.error("DDS has invalid magic bytes: %s", path)
                return None

            header = bytearray(raw[4:128])
            pf_fourcc = bytes(header[80:84])
            has_dx10 = (pf_fourcc == b"DX10")
            dx10_header = b""
            data_offset = 128
            if has_dx10:
                if len(raw) < 148:
                    logger.error("DDS DX10 header missing/truncated: %s", path)
                    return None
                dx10_header = raw[128:148]
                data_offset = 148

            data = raw[data_offset:]
            if not data:
                logger.error("DDS has no pixel payload: %s", path)
                return None

            return {
                "path": path,
                "header": header,
                "pf_fourcc": pf_fourcc,
                "has_dx10": has_dx10,
                "dx10_header": dx10_header,
                "data": data,
            }

        base = _read_dds_parts(dds_paths[0])
        if base is None:
            return False

        # --- Patch header for mipmap chain ---
        num_mips = len(dds_paths)
        header = base["header"]

        # dwMipMapCount at header offset 24
        struct.pack_into("<I", header, 24, num_mips)

        # Set required DDSD bits and mip flags in dwFlags (header offset 4)
        flags = struct.unpack_from("<I", header, 4)[0]
        flags |= (
            DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT |
            DDSD_MIPMAPCOUNT
        )
        struct.pack_into("<I", header, 4, flags)

        # Set TEXTURE | COMPLEX | MIPMAP in dwCaps (header offset 104)
        caps = struct.unpack_from("<I", header, 104)[0]
        caps |= DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
        struct.pack_into("<I", header, 104, caps)

        # Set dwPitchOrLinearSize (header offset 16) to base mip data size
        # and set DDSD_LINEARSIZE in dwFlags.
        base_data_size = len(base["data"])
        struct.pack_into("<I", header, 16, base_data_size)
        flags = struct.unpack_from("<I", header, 4)[0]
        flags |= DDSD_LINEARSIZE
        struct.pack_into("<I", header, 4, flags)

        # --- Validate successive mip dimensions ---
        # Each mip must be max(1, floor(prev/2)) in both dimensions.
        # Read base dimensions from the DDS header.
        base_w = struct.unpack_from("<I", base["header"], 12)[0]
        base_h = struct.unpack_from("<I", base["header"], 8)[0]
        for level_idx in range(1, len(dds_paths)):
            expected_w = max(base_w >> level_idx, 1)
            expected_h = max(base_h >> level_idx, 1)
            mip_part = _read_dds_parts(dds_paths[level_idx])
            if mip_part is None:
                return False
            mip_w = struct.unpack_from("<I", mip_part["header"], 12)[0]
            mip_h = struct.unpack_from("<I", mip_part["header"], 8)[0]
            if mip_w != expected_w or mip_h != expected_h:
                logger.error(
                    "DDS mip %d dimension mismatch: expected %dx%d, got %dx%d "
                    "(non-power-of-two source may cause invalid mipchain): %s",
                    level_idx, expected_w, expected_h, mip_w, mip_h,
                    dds_paths[level_idx],
                )
                return False

        # --- Collect pixel data from each mip ---
        mip_datas = [base["data"]]
        for dds_path in dds_paths[1:]:
            mip = _read_dds_parts(dds_path)
            if mip is None:
                return False
            if mip["has_dx10"] != base["has_dx10"]:
                logger.error(
                    "DDS mip header mismatch (DX10 presence differs): %s", dds_path
                )
                return False
            if mip["pf_fourcc"] != base["pf_fourcc"]:
                logger.error(
                    "DDS mip header mismatch (fourCC differs): %s", dds_path
                )
                return False
            if base["has_dx10"] and mip["dx10_header"] != base["dx10_header"]:
                logger.error(
                    "DDS mip header mismatch (DX10 header differs): %s", dds_path
                )
                return False
            mip_datas.append(mip["data"])

        # --- Patch DX10 header for sRGB when requested ---
        dx10_out = base["dx10_header"]
        if srgb and base["has_dx10"] and len(dx10_out) >= 4:
            dx10_out = bytearray(dx10_out)
            dxgi_fmt = struct.unpack_from("<I", dx10_out, 0)[0]
            srgb_fmt = MipmapGenerator._DXGI_SRGB_MAP.get(dxgi_fmt)
            if srgb_fmt is not None:
                struct.pack_into("<I", dx10_out, 0, srgb_fmt)
                logger.debug(
                    "Patched DX10 DXGI format %d → %d (sRGB) for %s",
                    dxgi_fmt, srgb_fmt, output_path,
                )
            dx10_out = bytes(dx10_out)

        # --- Write assembled DDS ---
        with open(output_path, "wb") as f:
            f.write(b"DDS ")
            f.write(bytes(header))
            if base["has_dx10"]:
                f.write(dx10_out)
            for data in mip_datas:
                f.write(data)

        # Basic structural sanity check on assembled file.
        min_size = 128 + (20 if base["has_dx10"] else 0) + sum(len(d) for d in mip_datas)
        try:
            actual_size = os.path.getsize(output_path)
        except OSError:
            logger.error("Failed to stat assembled DDS output: %s", output_path)
            return False
        if actual_size < min_size:
            logger.error(
                "Assembled DDS size is smaller than expected (%d < %d): %s",
                actual_size, min_size, output_path
            )
            return False

        return True

    # ──────────────────────────────────────────
    # KTX2 Generation
    # ──────────────────────────────────────────

    def _resolve_ktx2_tool(self) -> Optional[str]:
        """Resolve and cache the KTX2 tool path (toktx from KTX-Software)."""
        if self._ktx2_tool_resolved:
            return self._ktx2_tool_path

        self._ktx2_tool_resolved = True
        import shutil as _shutil

        # Try toktx (from KTX-Software) first, then basisu on PATH
        tool_path = None
        for name in ("toktx", "basisu"):
            tool_path = _shutil.which(name)
            if tool_path:
                break

        # Fallback: check bundled tools in bin/ (platform-aware)
        if not tool_path:
            import platform as _platform
            from .. import BIN_DIR
            bin_dir = BIN_DIR
            is_windows = _platform.system() == "Windows"
            exe_suffix = ".exe" if is_windows else ""
            candidates = []
            # Check KTX-Software-* bundled directories
            for ktx_dir in sorted(bin_dir.glob("KTX-Software*"), reverse=True):
                candidates.append(ktx_dir / f"toktx{exe_suffix}")
            candidates.append(bin_dir / f"toktx{exe_suffix}")
            candidates.append(bin_dir / f"basisu{exe_suffix}")
            for candidate in candidates:
                if candidate.is_file():
                    tool_path = str(candidate)
                    break

        if tool_path:
            self._ktx2_tool_path = tool_path
            logger.info(f"Using KTX2 tool: {tool_path}")
            return tool_path

        logger.debug("KTX2 tool not found in PATH; checking bundled/installed alternatives.")
        logger.warning(
            "KTX2 tool not found. Install KTX-Software (toktx) from "
            "https://github.com/KhronosGroup/KTX-Software or basisu. "
            "Skipping KTX2 generation."
        )
        return None

    def generate_ktx2_mipchain(self, mip_paths: list, output_path: str,
                               uastc: bool = True, srgb: bool = False) -> bool:
        """Generate a single KTX2 with all precomputed mip levels.

        Passes pre-computed mip PNGs to toktx via per-level inputs.
        Falls back to base-only on failure.

        Returns True on success, False on failure.
        """
        if not mip_paths:
            logger.warning(
                "Skipping KTX2 mipchain generation for %s: no mip levels provided.",
                output_path
            )
            return False

        tool_path = self._resolve_ktx2_tool()
        if not tool_path:
            logger.warning(
                "Skipping KTX2 mipchain generation for %s: KTX2 tool unavailable.",
                output_path
            )
            return False

        if len(mip_paths) <= 1:
            return self.generate_ktx2(mip_paths[0]["path"], output_path, uastc, srgb=srgb)

        tool_name = os.path.basename(tool_path).lower().replace(".exe", "")

        # toktx supports --levels with multiple input files for mip levels
        if "toktx" in tool_name:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            cmd = [tool_path, "--t2"]
            if uastc:
                cmd += ["--encode", "uastc", "--uastc_quality", "2"]
            else:
                cmd += ["--encode", "etc1s", "--clevel", "3"]
            cmd += ["--assign_oetf", "srgb" if srgb else "linear"]
            cmd += ["--levels", str(len(mip_paths))]
            cmd.append(output_path)
            # Add mip level source files in order (level 0, 1, 2, ...)
            for mip_info in mip_paths:
                cmd.append(mip_info["path"])

            ok, _ = self._run_tool(
                cmd, "toktx", mip_paths[0]["path"],
                timeout=self._compression_timeout(),
            )
            if ok:
                logger.debug(
                    f"KTX2 mipchain ({len(mip_paths)} levels) created: "
                    f"{output_path}"
                )
                return True
            logger.warning("Falling back to base-only KTX2.")

        # Fallback: base level only
        return self.generate_ktx2(mip_paths[0]["path"], output_path, uastc, srgb=srgb)

    def generate_ktx2(self, source_path: str, output_path: str,
                      uastc: bool = True, srgb: bool = False) -> bool:
        """Convert a PNG to KTX2 format using toktx or basisu.

        Args:
            source_path: Input PNG file path.
            output_path: Output .ktx2 file path.
            uastc: If True, use UASTC encoding (higher quality, GPU-native).
                   If False, use ETC1S (smaller files, wider compatibility).
            srgb: If True, tag the output with the sRGB transfer function.

        Returns True on success, False on failure.

        """
        tool_path = self._resolve_ktx2_tool()
        if not tool_path:
            logger.warning(
                "Skipping KTX2 conversion for %s: KTX2 tool unavailable.",
                source_path
            )
            return False

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        tool_name = os.path.basename(tool_path).lower().replace(".exe", "")

        if "toktx" in tool_name:
            cmd = [tool_path, "--t2"]
            if uastc:
                cmd += ["--encode", "uastc", "--uastc_quality", "2"]
            else:
                cmd += ["--encode", "etc1s", "--clevel", "3"]
            cmd += ["--assign_oetf", "srgb" if srgb else "linear"]
            cmd += [output_path, source_path]
        else:
            # basisu
            cmd = [tool_path, "-ktx2"]
            if uastc:
                cmd += ["-uastc"]
            cmd += ["-file", source_path, "-output_file", output_path]

        ok, _ = self._run_tool(
            cmd, tool_name, source_path, timeout=self._compression_timeout()
        )
        if ok:
            logger.debug(f"KTX2 created: {output_path}")
        return ok
