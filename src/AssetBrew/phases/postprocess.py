"""Run production post-processing passes after core texture generation."""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord,
    ensure_rgb,
    extract_alpha,
    get_intermediate_path,
    linear_to_srgb,
    load_image,
    luminance_bt709,
    merge_alpha,
    save_image,
    srgb_to_linear,
)
from ..core.compat import tqdm

logger = logging.getLogger("asset_pipeline.postprocess")


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert image array to single-channel grayscale."""
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return luminance_bt709(arr, assume_srgb=False)
    return arr[:, :, 0].astype(np.float32, copy=False)


def _resize_map(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize map (HxW or HxWxC) to target shape."""
    if arr.shape[:2] == (height, width):
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _parse_cube_lut(path: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse a .cube LUT file.

    Returns (lut, domain_min, domain_max) or None when parsing fails.
    LUT indexing convention is lut[b, g, r].
    """
    if not path or not os.path.exists(path):
        return None

    size = 0
    values: list[list[float]] = []
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                upper = line.upper()
                if upper.startswith("TITLE"):
                    continue
                if upper.startswith("LUT_3D_SIZE"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        size = int(parts[1])
                    continue
                if upper.startswith("DOMAIN_MIN"):
                    parts = line.split()
                    if len(parts) == 4:
                        domain_min = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    continue
                if upper.startswith("DOMAIN_MAX"):
                    parts = line.split()
                    if len(parts) == 4:
                        domain_max = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    continue
                parts = line.split()
                if len(parts) == 3:
                    values.append([float(parts[0]), float(parts[1]), float(parts[2])])
    except Exception as exc:
        logger.warning("Failed to parse LUT '%s': %s", path, exc)
        return None

    if size <= 1:
        logger.warning("Invalid LUT size in '%s'", path)
        return None
    expected = size * size * size
    if len(values) != expected:
        logger.warning("Invalid LUT '%s': expected %d rows, found %d", path, expected, len(values))
        return None

    lut = np.array(values, dtype=np.float32).reshape((size, size, size, 3))
    return lut, domain_min.astype(np.float32), domain_max.astype(np.float32)


def _apply_lut_trilinear(
    rgb: np.ndarray,
    lut: np.ndarray,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> np.ndarray:
    """Apply 3D LUT using trilinear interpolation."""
    rgb_in = np.clip(rgb.astype(np.float32, copy=False), 0, 1)
    lut_size = int(lut.shape[0])

    dom_scale = np.maximum(domain_max - domain_min, 1e-6)
    normalized = np.clip((rgb_in - domain_min) / dom_scale, 0.0, 1.0)
    coords = normalized * float(lut_size - 1)

    r = coords[:, :, 0]
    g = coords[:, :, 1]
    b = coords[:, :, 2]

    r0 = np.floor(r).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    b0 = np.floor(b).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, lut_size - 1)
    g1 = np.clip(g0 + 1, 0, lut_size - 1)
    b1 = np.clip(b0 + 1, 0, lut_size - 1)

    fr = (r - r0).astype(np.float32)
    fg = (g - g0).astype(np.float32)
    fb = (b - b0).astype(np.float32)

    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    c00 = c000 * (1.0 - fr[:, :, None]) + c001 * fr[:, :, None]
    del c000, c001
    c01 = c010 * (1.0 - fr[:, :, None]) + c011 * fr[:, :, None]
    del c010, c011
    c10 = c100 * (1.0 - fr[:, :, None]) + c101 * fr[:, :, None]
    del c100, c101
    c11 = c110 * (1.0 - fr[:, :, None]) + c111 * fr[:, :, None]
    del c110, c111
    c0 = c00 * (1.0 - fg[:, :, None]) + c01 * fg[:, :, None]
    c1 = c10 * (1.0 - fg[:, :, None]) + c11 * fg[:, :, None]

    out = c0 * (1.0 - fb[:, :, None]) + c1 * fb[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class ORMPacker:
    """Pack material channels using configurable engine presets."""

    _VALID_CHANNELS = {
        "ao", "roughness", "metalness", "gloss", "smoothness", "one", "zero", "none",
    }

    def __init__(self, config: PipelineConfig):  # noqa: D107
        self.config = config
        self.cfg = config.orm_packing

    def _resolve_layout(self) -> Dict[str, str]:
        preset = (self.cfg.preset or "unreal_orm").strip().lower()
        if preset == "unreal_orm":
            layout = {"r": "ao", "g": "roughness", "b": "metalness", "a": "none"}
        elif preset == "unity_mas":
            layout = {"r": "metalness", "g": "ao", "b": "smoothness", "a": "none"}
        elif preset == "source_phong":
            layout = {"r": "ao", "g": "roughness", "b": "metalness", "a": "gloss"}
        elif preset == "idtech_rma":
            layout = {"r": "roughness", "g": "metalness", "b": "ao", "a": "none"}
        else:
            layout = {
                "r": self.cfg.r_channel,
                "g": self.cfg.g_channel,
                "b": self.cfg.b_channel,
                "a": self.cfg.alpha_channel,
            }
        if self.cfg.custom_layout:
            for key, val in self.cfg.custom_layout.items():
                k = key.strip().lower()
                if k in {"r", "g", "b", "a"}:
                    layout[k] = val.strip().lower()
        return layout

    @staticmethod
    def _channel_value(channel_map: Dict[str, np.ndarray], key: str) -> np.ndarray:
        if key == "one":
            shape = next(iter(channel_map.values())).shape
            return np.ones(shape, dtype=np.float32)
        if key in ("zero", "none"):
            shape = next(iter(channel_map.values())).shape
            return np.zeros(shape, dtype=np.float32)
        return channel_map[key]

    def process(
        self,
        record: AssetRecord,
        ao_path: str = None,
        roughness_path: str = None,
        metalness_path: str = None,
        diffuse_path: str = None,
        gloss_path: str = None,
    ) -> dict:
        """Pack AO/roughness/metalness (and variants) into a compact texture."""
        result = {
            "orm": None,
            "layout": None,
            "diffuse_alpha_packed": None,
            "skipped": False,
            "error": None,
        }

        if not self.cfg.enabled:
            result["skipped"] = True
            return result
        if not roughness_path or not os.path.exists(roughness_path):
            result["skipped"] = True
            return result

        try:
            rough = load_image(roughness_path, max_pixels=self.config.max_image_pixels)
            rough_gray = _to_grayscale(rough)
            h, w = rough_gray.shape[:2]

            if gloss_path and os.path.exists(gloss_path):
                gloss_gray = _to_grayscale(
                    load_image(gloss_path, max_pixels=self.config.max_image_pixels)
                )
                gloss_gray = _resize_map(gloss_gray, w, h)
            else:
                gloss_gray = np.clip(1.0 - rough_gray, 0.0, 1.0).astype(np.float32)

            if ao_path and os.path.exists(ao_path):
                ao_img = load_image(ao_path, max_pixels=self.config.max_image_pixels)
                ao_gray = _to_grayscale(ao_img)
                ao_gray = _resize_map(ao_gray, w, h)
            else:
                ao_gray = np.ones((h, w), dtype=np.float32)

            if metalness_path and os.path.exists(metalness_path):
                metal_gray = _to_grayscale(
                    load_image(metalness_path, max_pixels=self.config.max_image_pixels)
                )
                metal_gray = _resize_map(metal_gray, w, h)
            else:
                metal_gray = np.zeros((h, w), dtype=np.float32)

            channel_map = {
                "ao": np.clip(ao_gray, 0.0, 1.0).astype(np.float32),
                "roughness": np.clip(rough_gray, 0.0, 1.0).astype(np.float32),
                "metalness": np.clip(metal_gray, 0.0, 1.0).astype(np.float32),
                "gloss": np.clip(gloss_gray, 0.0, 1.0).astype(np.float32),
                "smoothness": np.clip(gloss_gray, 0.0, 1.0).astype(np.float32),
                "one": np.ones((h, w), dtype=np.float32),
                "zero": np.zeros((h, w), dtype=np.float32),
                "none": np.zeros((h, w), dtype=np.float32),
            }

            layout = self._resolve_layout()
            for channel_key in layout.values():
                if channel_key not in self._VALID_CHANNELS:
                    raise ValueError(f"Invalid pack channel: {channel_key}")

            r = self._channel_value(channel_map, layout["r"])
            g = self._channel_value(channel_map, layout["g"])
            b = self._channel_value(channel_map, layout["b"])

            # Include alpha channel when the layout specifies a real source
            # (not "none"/"zero").  Game engines expecting 4-channel RGBA ORM
            # would otherwise get truncated 3-channel output.
            alpha_source = layout.get("a", "none")
            if alpha_source not in ("none", "zero"):
                a = self._channel_value(channel_map, alpha_source)
                packed = np.stack([r, g, b, a], axis=-1).astype(np.float32)
            else:
                packed = np.stack([r, g, b], axis=-1).astype(np.float32)

            suffix = self.cfg.output_suffix or "_orm"
            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix=suffix,
                ext=".png",
            )
            save_image(packed, out_path, bits=16)
            result["orm"] = out_path
            result["layout"] = layout

            alpha_source = layout.get("a", "none")
            pack_diffuse_alpha = (
                self.cfg.generate_gloss_in_diffuse_alpha
                or alpha_source not in ("none", "zero")
            )
            if pack_diffuse_alpha and diffuse_path and os.path.exists(diffuse_path):
                base = load_image(
                    diffuse_path, max_pixels=self.config.max_image_pixels,
                )
                base_rgb = ensure_rgb(base).astype(np.float32, copy=False)
                base_rgb = (
                    _resize_map(base_rgb, w, h)
                    if base_rgb.shape[:2] != (h, w) else base_rgb
                )
                # When generate_gloss_in_diffuse_alpha is set and the layout
                # has no explicit alpha channel, use gloss_source config.
                if (
                    self.cfg.generate_gloss_in_diffuse_alpha
                    and alpha_source in ("none", "zero")
                ):
                    effective_alpha = self.cfg.gloss_source
                else:
                    effective_alpha = alpha_source
                chosen_alpha = self._channel_value(
                    channel_map,
                    effective_alpha
                    if effective_alpha in channel_map
                    else "gloss",
                )
                chosen_alpha = _resize_map(chosen_alpha, w, h)
                existing_alpha = extract_alpha(base)
                if existing_alpha is not None and not self.cfg.overwrite_existing_alpha:
                    alpha_out = _resize_map(existing_alpha, w, h)
                else:
                    alpha_out = np.clip(chosen_alpha, 0.0, 1.0).astype(np.float32)
                rgba = merge_alpha(base_rgb, alpha_out)
                diffuse_out = get_intermediate_path(
                    record.filepath,
                    "06_postprocess",
                    self.config.intermediate_dir,
                    suffix="_diffuse_packed",
                    ext=".png",
                )
                save_image(rgba, diffuse_out, bits=16)
                result["diffuse_alpha_packed"] = diffuse_out

            del channel_map

        except Exception as exc:
            logger.error("ORM packing failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class ColorConsistencyPass:
    """Match color statistics across textures in the same material group."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.color_consistency
        self._references: Dict[str, dict] = {}

    def build_references(self, records: List[AssetRecord], get_albedo_path) -> None:
        """Build color reference statistics for material groups."""
        if not self.cfg.enabled:
            return

        groups: Dict[str, List[str]] = defaultdict(list)
        for rec in records:
            tex_type = TextureType(rec.texture_type)
            if tex_type not in (TextureType.DIFFUSE, TextureType.ALBEDO, TextureType.UNKNOWN):
                continue
            key = rec.material_category if self.cfg.group_by_material else "all"
            path = get_albedo_path(rec)
            if path and os.path.exists(path):
                groups[key].append(path)

        total_images = sum(len(paths) for paths in groups.values() if len(paths) >= 2)
        with tqdm(total=total_images, desc="Building color refs") as pbar:
            for group, paths in groups.items():
                if len(paths) < 2:
                    continue
                all_means = []
                all_stds = []
                for path in paths:
                    try:
                        img = load_image(
                            path, max_pixels=self.config.max_image_pixels,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Color consistency: skipping unreadable "
                            "image %s: %s", path, exc,
                        )
                        pbar.update(1)
                        continue
                    rgb = ensure_rgb(img)
                    # OpenCV treats float32 RGB→Lab as linear input; convert
                    # via uint8 so the sRGB transfer function is applied.
                    rgb_u8 = np.round(
                        np.clip(rgb, 0, 1) * 255
                    ).astype(np.uint8)
                    lab = cv2.cvtColor(
                        rgb_u8, cv2.COLOR_RGB2Lab,
                    ).astype(np.float32)
                    all_means.append(np.mean(lab, axis=(0, 1)))
                    all_stds.append(np.std(lab, axis=(0, 1)))
                    del img, rgb, rgb_u8, lab
                    pbar.update(1)

                if not all_means:
                    logger.warning(
                        "Color consistency: all images in group '%s' failed "
                        "to load; skipping reference computation.", group,
                    )
                    continue
                self._references[group] = {
                    "mean": np.median(all_means, axis=0),
                    "std": np.median(all_stds, axis=0),
                }

    def process(self, record: AssetRecord, albedo_path: str) -> dict:
        """Apply color consistency correction to one asset."""
        result = {"corrected": None, "skipped": False, "error": None}
        if not self.cfg.enabled or not albedo_path or not os.path.exists(albedo_path):
            result["skipped"] = True
            return result

        key = record.material_category if self.cfg.group_by_material else "all"
        if key not in self._references:
            result["skipped"] = True
            return result

        try:
            img = load_image(albedo_path, max_pixels=self.config.max_image_pixels)
            rgb = ensure_rgb(img)
            # OpenCV treats float32 RGB→Lab as linear input; convert via
            # uint8 so the sRGB transfer function is applied correctly.
            rgb_u8 = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
            src_mean = np.mean(lab, axis=(0, 1))
            src_std = np.std(lab, axis=(0, 1))
            ref = self._references[key]
            strength = self.cfg.correction_strength
            original_lab = lab.copy()

            for idx in range(3):
                if src_std[idx] > 1e-6:
                    std_ratio = np.clip(ref["std"][idx] / src_std[idx], 0.5, 1.5)
                    lab[:, :, idx] = (lab[:, :, idx] - src_mean[idx]) * std_ratio + ref["mean"][idx]

            lab = lab * strength + original_lab * (1.0 - strength)
            lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
            # Convert back via uint8 for correct sRGB encoding
            lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
            corrected_u8 = cv2.cvtColor(lab_u8, cv2.COLOR_Lab2RGB)
            corrected = corrected_u8.astype(np.float32) / 255.0

            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_cc",
                ext=".png",
            )
            save_image(corrected, out_path)
            result["corrected"] = out_path
        except Exception as exc:
            logger.error("Color consistency failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class ColorGradingPass:
    """Apply white balance, exposure, tone curve, LUT, saturation, and cleanup."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.color_grading
        self._cached_lut_path: str = ""
        self._cached_lut: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    @staticmethod
    def _material_value(table: Dict[str, float], record: AssetRecord):
        """Return per-material override or None if no entry exists."""
        cat = (record.material_category or "default").strip().lower()
        if cat in table:
            return float(table[cat])
        if "default" in table:
            return float(table["default"])
        return None

    def _load_lut(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        path = (self.cfg.lut_path or "").strip()
        if not path:
            return None
        if path == self._cached_lut_path:
            return self._cached_lut
        parsed = _parse_cube_lut(path)
        self._cached_lut_path = path
        self._cached_lut = parsed
        if parsed is None:
            logger.warning("Color LUT unavailable or invalid: %s", path)
        return parsed

    def process(self, record: AssetRecord, albedo_path: str) -> dict:
        """Apply color grading adjustments to one asset."""
        result = {"graded": None, "skipped": False, "error": None}
        if not self.cfg.enabled or not albedo_path or not os.path.exists(albedo_path):
            result["skipped"] = True
            return result
        if record.texture_type not in set(self.cfg.apply_to_texture_types):
            result["skipped"] = True
            return result

        try:
            img = load_image(albedo_path, max_pixels=self.config.max_image_pixels)
            rgb_srgb = np.clip(ensure_rgb(img).astype(np.float32, copy=False), 0.0, 1.0)
            working = srgb_to_linear(rgb_srgb) if self.cfg.process_in_linear else rgb_srgb.copy()

            wb = float(self.cfg.white_balance_shift)
            wb += self._material_value(self.cfg.white_balance_per_material, record) or 0.0
            wb = float(np.clip(wb, -1.0, 1.0))
            red_scale = 1.0 + 0.18 * wb
            blue_scale = 1.0 - 0.18 * wb
            green_scale = 1.0 - 0.06 * abs(wb)
            working[:, :, 0] *= red_scale
            working[:, :, 1] *= green_scale
            working[:, :, 2] *= blue_scale

            exposure_ev = float(self.cfg.exposure_ev)
            exposure_ev += self._material_value(self.cfg.exposure_per_material, record) or 0.0
            working *= float(2.0 ** exposure_ev)

            gamma = float(self.cfg.midtone_gamma)
            gamma_override = self._material_value(self.cfg.midtone_gamma_per_material, record)
            if gamma_override is not None:
                gamma = gamma_override
            gamma = max(gamma, 1e-3)
            working = np.power(np.clip(working, 0.0, 1.0), gamma)

            sat = float(self.cfg.saturation)
            sat_override = self._material_value(self.cfg.saturation_per_material, record)
            if sat_override is not None:
                sat *= sat_override
            luma = luminance_bt709(working, assume_srgb=False)[:, :, None]
            working = np.clip(luma + (working - luma) * sat, 0.0, 1.0)

            lut_strength = float(np.clip(self.cfg.lut_strength, 0.0, 1.0))
            lut_data = self._load_lut() if lut_strength > 0 else None
            if lut_data is not None:
                lut, dmin, dmax = lut_data
                lut_input = linear_to_srgb(working) if self.cfg.process_in_linear else working
                lut_out = _apply_lut_trilinear(lut_input, lut, dmin, dmax)
                lut_mix = lut_input * (1.0 - lut_strength) + lut_out * lut_strength
                working = srgb_to_linear(lut_mix) if self.cfg.process_in_linear else lut_mix

            denoise_strength = float(max(self.cfg.denoise_strength, 0.0))
            if denoise_strength > 0:
                denoise_input = linear_to_srgb(working) if self.cfg.process_in_linear else working
                denoise_u8 = np.clip(denoise_input * 255.0, 0, 255).astype(np.uint8)
                hval = max(int(round(6 + denoise_strength * 12)), 1)
                denoised_u8 = cv2.fastNlMeansDenoisingColored(
                    denoise_u8,
                    None,
                    h=hval,
                    hColor=hval,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
                denoised = denoised_u8.astype(np.float32) / 255.0
                working = srgb_to_linear(denoised) if self.cfg.process_in_linear else denoised

            sharpen_strength = float(max(self.cfg.sharpen_strength, 0.0))
            if sharpen_strength > 0:
                sigma = max(float(self.cfg.sharpen_radius), 0.5)
                blur = cv2.GaussianBlur(working, (0, 0), sigmaX=sigma, sigmaY=sigma)
                working = np.clip(working + sharpen_strength * (working - blur), 0.0, 1.0)

            if self.cfg.auto_plausible_albedo_clamp and self.cfg.process_in_linear:
                low = float(np.clip(self.cfg.plausible_albedo_min, 0.0, 1.0))
                high = float(np.clip(self.cfg.plausible_albedo_max, 0.0, 1.0))
                if low < high:
                    working = np.clip(working, low, high)

            graded = linear_to_srgb(working) if self.cfg.process_in_linear else working
            graded = np.clip(graded, 0.0, 1.0).astype(np.float32)

            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_graded",
                ext=".png",
            )
            save_image(graded, out_path)
            result["graded"] = out_path
        except Exception as exc:
            logger.error("Color grading failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class SeamRepairProcessor:
    """Detect and repair wrap seams after upscale/generation."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.seam_repair

    @staticmethod
    def _seam_score(rgb: np.ndarray, border_width: int) -> float:
        bw = max(1, border_width)
        left = rgb[:, :bw, :]
        right = rgb[:, -bw:, :]
        top = rgb[:bw, :, :]
        bottom = rgb[-bw:, :, :]

        lr_pixel = np.mean(np.abs(left - right))
        tb_pixel = np.mean(np.abs(top - bottom))
        gray = luminance_bt709(rgb, assume_srgb=False)

        grad_seam_lr = np.abs(gray[:, 0] - gray[:, -1])
        grad_internal_lr = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
        grad_seam_tb = np.abs(gray[0, :] - gray[-1, :])
        grad_internal_tb = np.mean(np.abs(np.diff(gray, axis=0)), axis=0)
        lr_grad = np.mean(np.maximum(grad_seam_lr - grad_internal_lr, 0))
        tb_grad = np.mean(np.maximum(grad_seam_tb - grad_internal_tb, 0))

        return float(0.4 * ((lr_pixel + tb_pixel) * 0.5) + 0.6 * ((lr_grad + tb_grad) * 0.5))

    def _repair_rgb(self, rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
        bw = int(max(self.cfg.repair_border_width, 1))
        score_before = self._seam_score(rgb, bw)
        if score_before < float(self.cfg.detect_threshold):
            return rgb.astype(np.float32, copy=False), score_before, score_before

        out = rgb.copy().astype(np.float32)
        blend_strength = float(np.clip(self.cfg.blend_strength, 0.0, 1.0))
        for idx in range(bw):
            t = (bw - idx) / max(bw, 1) * blend_strength
            lr_avg = 0.5 * (out[:, idx, :] + out[:, -idx - 1, :])
            out[:, idx, :] = out[:, idx, :] * (1.0 - t) + lr_avg * t
            out[:, -idx - 1, :] = out[:, -idx - 1, :] * (1.0 - t) + lr_avg * t

            tb_avg = 0.5 * (out[idx, :, :] + out[-idx - 1, :, :])
            out[idx, :, :] = out[idx, :, :] * (1.0 - t) + tb_avg * t
            out[-idx - 1, :, :] = out[-idx - 1, :, :] * (1.0 - t) + tb_avg * t

        inpaint_radius = int(max(self.cfg.inpaint_radius, 0))
        if inpaint_radius > 0:
            h, w = out.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            seam_thresh = float(self.cfg.detect_threshold)
            # Per-pixel diffs are on a larger scale than the composite
            # seam score, so use 2x detect_threshold for inpaint mask.
            inpaint_thresh = seam_thresh * 2.0
            lr_diff = np.mean(np.abs(out[:, :1, :] - out[:, -1:, :]), axis=2).squeeze(1)
            tb_diff = np.mean(np.abs(out[:1, :, :] - out[-1:, :, :]), axis=2).squeeze(0)
            row_idx = np.where(lr_diff > inpaint_thresh)[0]
            col_idx = np.where(tb_diff > inpaint_thresh)[0]
            mask[row_idx, 0:bw] = 255
            mask[row_idx, w - bw:w] = 255
            mask[0:bw, col_idx] = 255
            mask[h - bw:h, col_idx] = 255

            if np.any(mask):
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (inpaint_radius * 2 + 1, inpaint_radius * 2 + 1),
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
                # cv2.inpaint requires uint8; quantize high byte of 16-bit
                # to preserve some precision in the float32->uint8 round-trip.
                out_u16 = np.clip(out * 65535.0, 0, 65535).astype(np.uint16)
                # Split into high/low bytes and inpaint the high byte for
                # structure, then blend back at float precision.
                out_u8 = (out_u16 >> 8).astype(np.uint8)
                inpainted_u8 = cv2.inpaint(out_u8, mask, inpaint_radius, cv2.INPAINT_TELEA)
                inpainted_f = inpainted_u8.astype(np.float32) / 255.0
                # Only overwrite masked pixels; preserve original float32
                # precision in non-masked regions.
                mask_bool = mask > 0
                if mask_bool.ndim == 2 and out.ndim == 3:
                    mask_bool = mask_bool[:, :, None]
                out = np.where(mask_bool, inpainted_f, out)

        score_after = self._seam_score(out, bw)
        return np.clip(out, 0.0, 1.0).astype(np.float32), score_before, score_after

    def _process_path(
        self, record: AssetRecord, source_path: str, suffix: str,
    ) -> tuple[str, float, float]:
        arr = load_image(source_path, max_pixels=self.config.max_image_pixels)
        rgb = ensure_rgb(arr)
        alpha = extract_alpha(arr)
        repaired_rgb, score_before, score_after = self._repair_rgb(rgb)
        repaired = merge_alpha(repaired_rgb, alpha) if alpha is not None else repaired_rgb
        out_path = get_intermediate_path(
            record.filepath,
            "06_postprocess",
            self.config.intermediate_dir,
            suffix=suffix,
            ext=".png",
        )
        save_image(repaired, out_path, bits=16)
        return out_path, score_before, score_after

    def process(
        self, record: AssetRecord,
        upscaled_path: str = None, albedo_path: str = None,
    ) -> dict:
        """Detect and repair tiling seams in one asset."""
        result = {
            "upscaled_repaired": None,
            "albedo_repaired": None,
            "scores": {},
            "flagged": False,
            "skipped": False,
            "error": None,
        }
        if not self.cfg.enabled:
            result["skipped"] = True
            return result
        if self.cfg.only_tileable and not record.is_tileable:
            result["skipped"] = True
            return result

        try:
            any_processed = False
            if upscaled_path and os.path.exists(upscaled_path):
                path, before, after = self._process_path(record, upscaled_path, "_seamfix")
                result["upscaled_repaired"] = path
                result["scores"]["upscaled_before"] = before
                result["scores"]["upscaled_after"] = after
                any_processed = True
                if before >= self.cfg.detect_threshold:
                    result["flagged"] = True

            if albedo_path and os.path.exists(albedo_path):
                path, before, after = self._process_path(record, albedo_path, "_albedo_seamfix")
                result["albedo_repaired"] = path
                result["scores"]["albedo_before"] = before
                result["scores"]["albedo_after"] = after
                any_processed = True
                if before >= self.cfg.detect_threshold:
                    result["flagged"] = True

            if not any_processed:
                result["skipped"] = True
        except Exception as exc:
            logger.error("Seam repair failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class EmissiveMapGenerator:
    """Detect emissive regions and generate emissive maps."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.emissive

    def process(self, record: AssetRecord, albedo_path: str = None) -> dict:
        """Generate emissive map from albedo for one asset."""
        result = {"emissive": None, "emissive_mask": None, "skipped": False, "error": None}
        if not self.cfg.enabled or not albedo_path or not os.path.exists(albedo_path):
            result["skipped"] = True
            return result

        try:
            rgb = np.clip(
                ensure_rgb(load_image(albedo_path, max_pixels=self.config.max_image_pixels)),
                0,
                1,
            ).astype(np.float32)

            linear = srgb_to_linear(rgb)
            lum = luminance_bt709(linear, assume_srgb=False)
            hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            sat = hsv[:, :, 1] / 255.0
            val = hsv[:, :, 2] / 255.0

            mask = (
                ((lum > self.cfg.luminance_threshold) & (sat > self.cfg.saturation_threshold))
                | ((val > self.cfg.value_threshold) & (sat > self.cfg.saturation_threshold * 0.6))
            )
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            h, w = mask_u8.shape
            min_area = max(int(self.cfg.min_region_ratio * h * w), 1)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
            filtered = np.zeros_like(mask_u8)
            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area >= min_area:
                    filtered[labels == label] = 255

            emissive_mask = filtered.astype(np.float32) / 255.0
            if float(np.mean(emissive_mask)) <= 1e-5:
                result["skipped"] = True
                return result

            emissive = np.clip(rgb * emissive_mask[:, :, None] * float(self.cfg.boost), 0.0, 1.0)
            emissive_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_emissive",
                ext=".png",
            )
            mask_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_emissive_mask",
                ext=".png",
            )
            save_image(emissive.astype(np.float32), emissive_path, bits=16)
            save_image(emissive_mask.astype(np.float32), mask_path, bits=16)
            result["emissive"] = emissive_path
            result["emissive_mask"] = mask_path
        except Exception as exc:
            logger.error("Emissive generation failed for %s: %s",
                         record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class ReflectionMaskGenerator:
    """Generate environment reflection masks from roughness/metalness."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.reflection_mask

    def process(
        self, record: AssetRecord,
        roughness_path: str = None, metalness_path: str = None,
    ) -> dict:
        """Generate environment reflection mask for one asset."""
        result = {"env_mask": None, "skipped": False, "error": None}
        if not self.cfg.enabled or not roughness_path or not os.path.exists(roughness_path):
            result["skipped"] = True
            return result

        try:
            rough_img = load_image(
                roughness_path, max_pixels=self.config.max_image_pixels,
            )
            rough = _to_grayscale(rough_img)
            h, w = rough.shape[:2]
            if metalness_path and os.path.exists(metalness_path):
                metal_img = load_image(
                    metalness_path,
                    max_pixels=self.config.max_image_pixels,
                )
                metal = _to_grayscale(metal_img)
                metal = _resize_map(metal, w, h)
            else:
                metal = np.zeros((h, w), dtype=np.float32)

            gloss = np.clip(1.0 - rough, 0.0, 1.0)
            env_mask = (
                float(self.cfg.gloss_weight) * gloss
                + float(self.cfg.metalness_weight) * metal
                + float(self.cfg.bias)
            )
            env_mask = np.clip(env_mask, 0.0, 1.0).astype(np.float32)
            env_mask = cv2.GaussianBlur(env_mask, (0, 0), sigmaX=1.0, sigmaY=1.0)

            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_envmask",
                ext=".png",
            )
            save_image(env_mask, out_path, bits=16)
            result["env_mask"] = out_path
        except Exception as exc:
            logger.error("Reflection mask generation failed for %s: %s",
                         record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class SpecularAAProcessor:
    """Geometric specular anti-aliasing for roughness maps."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.specular_aa

    def process(
        self, record: AssetRecord,
        normal_path: str = None, roughness_path: str = None,
    ) -> dict:
        """Apply specular antialiasing to roughness for one asset."""
        result = {"roughness_aa": None, "skipped": False, "error": None}
        if not self.cfg.enabled:
            result["skipped"] = True
            return result
        if not normal_path or not os.path.exists(normal_path):
            result["skipped"] = True
            return result
        if not roughness_path or not os.path.exists(roughness_path):
            result["skipped"] = True
            return result

        try:
            normal = load_image(normal_path, max_pixels=self.config.max_image_pixels)
            roughness = load_image(roughness_path, max_pixels=self.config.max_image_pixels)
            normal_rgb = ensure_rgb(normal)
            rough_gray = _to_grayscale(roughness)
            # Align dimensions — normal is the reference resolution
            h, w = normal_rgb.shape[:2]
            rough_gray = _resize_map(rough_gray, w, h)
            n = normal_rgb * 2.0 - 1.0
            k = self.cfg.kernel_size

            nx, ny = n[:, :, 0], n[:, :, 1]
            nx_var = uniform_filter(nx**2, size=k) - uniform_filter(nx, size=k)**2
            ny_var = uniform_filter(ny**2, size=k) - uniform_filter(ny, size=k)**2
            variance = np.maximum(nx_var + ny_var, 0.0)

            roughness_increase = np.clip(
                variance - self.cfg.variance_threshold,
                0,
                self.cfg.max_roughness_increase,
            )
            rough_sq = rough_gray**2
            new_rough = np.sqrt(np.clip(rough_sq + roughness_increase, 0, 1)).astype(np.float32)

            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_roughness_aa",
                ext=".png",
            )
            save_image(new_rough, out_path, bits=16)
            result["roughness_aa"] = out_path
        except Exception as exc:
            logger.error("Specular AA failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result


class DetailMapOverlay:
    """Overlay a tiled detail normal map onto asset normals."""

    def __init__(self, config: PipelineConfig):
        """Initialize with pipeline config."""
        self.config = config
        self.cfg = config.detail_map
        self._detail_normal: Optional[np.ndarray] = None

    def _load_detail_normal(self) -> Optional[np.ndarray]:
        if self._detail_normal is not None:
            return self._detail_normal

        path = self.cfg.detail_normal_path
        if not path or not os.path.exists(path):
            logger.warning("Detail normal not found: %s", path)
            return None
        self._detail_normal = load_image(path, max_pixels=self.config.max_image_pixels)
        return self._detail_normal

    def process(self, record: AssetRecord, normal_path: str = None) -> dict:
        """Overlay tiled detail normal onto asset normal for one asset."""
        result = {"normal_detailed": None, "skipped": False, "error": None}
        if not self.cfg.enabled:
            result["skipped"] = True
            return result
        if record.material_category not in self.cfg.apply_to_categories:
            result["skipped"] = True
            return result
        if not normal_path or not os.path.exists(normal_path):
            result["skipped"] = True
            return result

        detail = self._load_detail_normal()
        if detail is None:
            result["skipped"] = True
            return result

        try:
            base_normal = load_image(normal_path, max_pixels=self.config.max_image_pixels)
            base_rgb = ensure_rgb(base_normal)
            h, w = base_rgb.shape[:2]

            detail_rgb = ensure_rgb(detail)
            scale = self.cfg.detail_uv_scale
            detail_h = int(h / scale)
            detail_w = int(w / scale)
            if detail_h < 1 or detail_w < 1:
                result["skipped"] = True
                return result

            detail_tile = cv2.resize(
                detail_rgb, (detail_w, detail_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
            detail_tile = np.clip(detail_tile, 0, 1).astype(np.float32)

            map_x = np.mod(np.arange(w, dtype=np.float32)[None, :], float(detail_w))
            map_y = np.mod(np.arange(h, dtype=np.float32)[:, None], float(detail_h))
            map_x = np.broadcast_to(map_x, (h, w))
            map_y = np.broadcast_to(map_y, (h, w))
            tiled = cv2.remap(
                detail_tile,
                map_x.astype(np.float32, copy=False),
                map_y.astype(np.float32, copy=False),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )

            n1 = base_rgb * 2.0 - 1.0
            n2 = tiled * 2.0 - 1.0
            strength = self.cfg.detail_strength
            n2[:, :, 0] *= strength
            n2[:, :, 1] *= strength
            # Renormalize n2 after XY scaling so the Whiteout blend
            # receives unit-length inputs (avoids directional bias).
            len_n2 = np.sqrt(n2[:, :, 0]**2 + n2[:, :, 1]**2 + n2[:, :, 2]**2)
            n2 = n2 / np.maximum(len_n2[:, :, np.newaxis], 1e-8)

            bx = n1[:, :, 0] * n2[:, :, 2] + n2[:, :, 0] * n1[:, :, 2]
            by = n1[:, :, 1] * n2[:, :, 2] + n2[:, :, 1] * n1[:, :, 2]
            bz = n1[:, :, 2] * n2[:, :, 2]
            length = np.sqrt(bx**2 + by**2 + bz**2)
            length = np.maximum(length, 1e-8)

            blended = np.stack([
                bx / length * 0.5 + 0.5,
                by / length * 0.5 + 0.5,
                bz / length * 0.5 + 0.5,
            ], axis=-1).astype(np.float32)

            out_path = get_intermediate_path(
                record.filepath,
                "06_postprocess",
                self.config.intermediate_dir,
                suffix="_normal_detail",
                ext=".png",
            )
            save_image(blended, out_path, bits=16)
            result["normal_detailed"] = out_path
        except Exception as exc:
            logger.error("Detail overlay failed for %s: %s", record.filename, exc, exc_info=True)
            result["error"] = str(exc)

        return result

