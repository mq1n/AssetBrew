"""Validate generated textures and produce QA artifacts.

This phase runs tiling, normal-map, quality, regression, and preview-sphere
checks, then emits per-asset and summary reports.
"""

import logging
import os
import json

import numpy as np
import cv2
from PIL import Image

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord, load_image, save_image, ensure_rgb, luminance_bt709, srgb_to_linear
)

logger = logging.getLogger("asset_pipeline.validator")


class Validator:
    """Validate processed textures and generate comparisons."""

    _METAL_CATEGORIES = {"metal", "gold", "silver", "iron", "copper"}
    _NON_METAL_CATEGORIES = {
        "wood",
        "stone",
        "brick",
        "fabric",
        "plastic",
        "glass",
        "skin",
        "leather",
        "concrete",
        "default",
        "unknown",
    }

    def __init__(self, config: PipelineConfig):
        """Initialize validator state and cached sphere geometry."""
        self.config = config
        self.cfg = config.validation
        self.tiling_cfg = config.tiling_quality
        # Pre-compute sphere geometry once
        self._sphere_normals = None
        self._sphere_uvs = None
        self._sphere_mask = None

    @staticmethod
    def _append_unique(target: list, items: list):
        """Append entries that are not already present in the target list."""
        for item in items:
            if item not in target:
                target.append(item)

    def validate_asset(self, record: AssetRecord, processed_paths: dict) -> dict:
        """Run configured validation checks for one processed asset."""
        report = {
            "filename": record.filename,
            "passed": True,
            "warnings": [],
            "errors": [],
            "checks": {}
        }

        # 1. Tiling check
        if self.cfg.check_tiling and record.is_tileable:
            upscaled = processed_paths.get("upscaled")
            if upscaled and os.path.exists(upscaled):
                tiling_ok, score = self._check_tiling(upscaled)
                quality_score = self._tiling_quality_score(score)
                report["checks"]["tiling"] = {
                    "passed": tiling_ok,
                    "seam_score": round(score, 4),
                    "quality_score": round(quality_score, 4),
                }
                if not tiling_ok:
                    report["errors"].append(f"Tiling seam detected (score={score:.4f})")
                if quality_score < self.tiling_cfg.fail_score:
                    report["warnings"].append(
                        f"Tiling quality is poor "
                        f"(score={quality_score:.3f}); manual review required."
                    )
                elif quality_score < self.tiling_cfg.warn_score:
                    report["warnings"].append(
                        f"Tiling quality is below preferred target (score={quality_score:.3f})."
                    )
            elif not upscaled:
                logger.debug(
                    "Skipping tiling check for %s: upscaled output not available.",
                    record.filename
                )
            elif not os.path.exists(upscaled):
                logger.warning(
                    "Skipping tiling check for %s: upscaled output missing (%s).",
                    record.filename, upscaled
                )

        # 2. Normal map validation
        if self.cfg.check_normals and processed_paths.get("normal"):
            path = processed_paths["normal"]
            if os.path.exists(path):
                ok, issues = self._check_normal_map(path)
                report["checks"]["normal_map"] = {"passed": ok, "issues": issues}
                if not ok:
                    report["errors"].extend(issues)
            else:
                logger.warning(
                    "Skipping normal validation for %s: normal map missing (%s).",
                    record.filename, path
                )
        elif self.cfg.check_normals:
            logger.debug("Skipping normal validation for %s: path not provided.", record.filename)

        # 3. Quality checks on all outputs
        for key in [
            "upscaled",
            "albedo",
            "roughness",
            "metalness",
            "ao",
            "gloss",
            "normal",
            "height",
            "orm",
            "emissive",
            "env_mask",
            "zone_mask",
        ]:
            path = processed_paths.get(key)
            if path and os.path.exists(path):
                ok, errors, warnings = self._check_quality(path)
                report["checks"][f"{key}_quality"] = {
                    "passed": ok, "issues": errors, "warnings": warnings,
                }
                if not ok:
                    report["errors"].extend(errors)
                if warnings:
                    self._append_unique(report["warnings"], warnings)
            elif path and not os.path.exists(path):
                logger.warning(
                    "Skipping quality check for %s (%s): file missing (%s).",
                    record.filename, key, path
                )
            elif not path:
                logger.debug(
                    "Skipping quality check for %s (%s): output not provided.",
                    record.filename, key
                )

        # 4. Material-semantic plausibility checks
        if self.cfg.enforce_material_semantics:
            semantic = self._check_material_semantics(record, processed_paths)
            report["checks"]["material_semantics"] = semantic
            self._append_unique(report["warnings"], semantic["warnings"])
            self._append_unique(report["errors"], semantic["errors"])

        # 5. Physically plausible albedo checks
        if self.cfg.enforce_plausible_albedo:
            albedo_check = self._check_physical_albedo(record, processed_paths)
            report["checks"]["albedo_plausibility"] = albedo_check
            self._append_unique(report["warnings"], albedo_check["warnings"])
            self._append_unique(report["errors"], albedo_check["errors"])

        # 6. Heuristic map limitations
        heuristic = self._check_heuristic_map_limitations(record, processed_paths)
        if heuristic["warnings"] or heuristic["errors"]:
            report["checks"]["heuristic_limitations"] = heuristic
            self._append_unique(report["warnings"], heuristic["warnings"])
            self._append_unique(report["errors"], heuristic["errors"])

        # 7. Sphere render test
        if self.cfg.render_test_sphere:
            sphere_path = self._render_test_sphere(record, processed_paths)
            if sphere_path:
                report["checks"]["sphere_render"] = {"path": sphere_path}
            else:
                logger.debug(
                    "Skipping sphere render for %s: prerequisites not met.",
                    record.filename
                )

        # 8. Regression diff against previous output
        if self.cfg.regression_diff:
            upscaled = processed_paths.get("upscaled")
            if upscaled and os.path.exists(upscaled):
                prev_path = os.path.join(
                    self.config.output_dir,
                    os.path.splitext(record.filepath)[0] + ".png"
                )
                if os.path.exists(prev_path):
                    regressed, diff_score, diff_path = self._regression_diff(
                        prev_path, upscaled, record
                    )
                    report["checks"]["regression"] = {
                        "passed": not regressed,
                        "diff_score": round(diff_score, 4),
                    }
                    if diff_path:
                        report["checks"]["regression"]["diff_image"] = diff_path
                    if regressed:
                        report["warnings"].append(
                            f"Regression detected (diff={diff_score:.4f})"
                        )
                else:
                    logger.debug(
                        "Skipping regression check for %s: previous output not found (%s).",
                        record.filename, prev_path
                    )
            else:
                logger.debug(
                    "Skipping regression check for %s: upscaled output missing.",
                    record.filename
                )

        # 9. Before/after comparison
        if self.cfg.output_comparison:
            original_path = os.path.join(self.config.input_dir, record.filepath)
            upscaled = processed_paths.get("upscaled")
            if upscaled and os.path.exists(upscaled) and os.path.exists(original_path):
                comp = self._generate_comparison(original_path, upscaled, record)
                report["comparison_path"] = comp
            elif not upscaled:
                logger.debug(
                    "Skipping output comparison for %s: upscaled output not provided.",
                    record.filename
                )
            elif not os.path.exists(original_path):
                logger.warning(
                    "Skipping output comparison for %s: original input missing (%s).",
                    record.filename, original_path
                )
            elif not os.path.exists(upscaled):
                logger.warning(
                    "Skipping output comparison for %s: processed output missing (%s).",
                    record.filename, upscaled
                )

        report["passed"] = len(report["errors"]) == 0
        return report

    def _check_tiling(self, image_path: str, border_width: int = None) -> tuple:
        """Detect tiling seams using gradient continuity at wrap boundaries.

        Compares the gradient (derivative) across the left/right and top/bottom
        seam lines. A seamless texture has smooth gradient transitions at the
        boundary; a seam produces a sudden gradient discontinuity.

        Combines pixel-level difference with gradient-level difference for
        more robust seam detection than raw pixel comparison alone.
        """
        if border_width is None:
            border_width = self.tiling_cfg.border_width
        img = load_image(image_path, max_pixels=self.config.max_image_pixels)
        rgb = ensure_rgb(img)

        # 1. Pixel-level border difference (original method)
        left = rgb[:, :border_width, :]
        right = rgb[:, -border_width:, :]
        lr_pixel = np.mean(np.abs(left - right))

        top = rgb[:border_width, :, :]
        bottom = rgb[-border_width:, :, :]
        tb_pixel = np.mean(np.abs(top - bottom))

        # 2. Gradient continuity: compute horizontal/vertical gradient at
        #    the wrap seam and compare with internal gradient magnitude.
        gray = luminance_bt709(rgb, assume_srgb=False)

        # Horizontal seam gradient (left edge col 0 vs right edge last col)
        grad_seam_lr = np.abs(gray[:, 0] - gray[:, -1])
        grad_internal_lr = np.mean(np.abs(np.diff(gray, axis=1)), axis=1)
        lr_grad = np.mean(np.maximum(grad_seam_lr - grad_internal_lr, 0))

        # Vertical seam gradient (top row 0 vs bottom last row)
        grad_seam_tb = np.abs(gray[0, :] - gray[-1, :])
        grad_internal_tb = np.mean(np.abs(np.diff(gray, axis=0)), axis=0)
        tb_grad = np.mean(np.maximum(grad_seam_tb - grad_internal_tb, 0))

        # Combined score: weight gradient continuity more heavily
        pixel_score = (lr_pixel + tb_pixel) / 2.0
        grad_score = (lr_grad + tb_grad) / 2.0
        score = 0.4 * pixel_score + 0.6 * grad_score

        return score < self.cfg.max_allowed_diff, float(score)

    def _tiling_quality_score(self, seam_score: float) -> float:
        """Convert seam error into normalized quality score in [0, 1]."""
        denom = max(float(self.cfg.max_allowed_diff) * 2.0, 1e-6)
        return float(np.clip(1.0 - (float(seam_score) / denom), 0.0, 1.0))

    def _check_normal_map(self, normal_path: str) -> tuple:
        img = load_image(normal_path, max_pixels=self.config.max_image_pixels)
        rgb = ensure_rgb(img)
        issues = []

        decoded = rgb * 2.0 - 1.0

        if np.any(~np.isfinite(decoded)):
            issues.append(f"Found {np.sum(~np.isfinite(decoded))} NaN/Inf values")

        lengths = np.sqrt(np.sum(decoded**2, axis=-1))
        tolerance = self.cfg.normal_unit_tolerance
        non_unit = np.abs(lengths - 1.0) > tolerance
        ratio = np.mean(non_unit)
        if ratio > 0.01:
            issues.append(f"{ratio*100:.1f}% of normals are non-unit length")

        neg_z = decoded[:, :, 2] < 0
        neg_ratio = np.mean(neg_z)
        if neg_ratio > 0.01:
            issues.append(f"{neg_ratio*100:.1f}% of normals have negative Z")

        return len(issues) == 0, issues

    def _check_quality(self, image_path: str) -> tuple:
        """Check basic quality metrics.

        Returns (passed: bool, errors: list, warnings: list).
        """
        errors = []
        warnings = []
        try:
            with Image.open(image_path) as img:
                w, h = img.size

                if w > 0 and h > 0:
                    if (w & (w - 1)) != 0 or (h & (h - 1)) != 0:
                        # NPOT is a warning -- many modern engines support it
                        warnings.append(f"Non-power-of-two dimensions: {w}x{h}")

                if w < 64 or h < 64:
                    errors.append(f"Very small resolution: {w}x{h}")

                arr_raw = np.array(img)
                original_dtype = arr_raw.dtype
                arr = arr_raw.astype(np.float32)
                # Normalize to [0,1] so threshold works for both 8-bit and 16-bit
                if arr.max() > 1.0:
                    if np.issubdtype(original_dtype, np.integer):
                        # Use dtype range for integer arrays
                        arr = arr / float(np.iinfo(original_dtype).max)
                    elif arr.max() > 255.0:
                        arr = arr / 65535.0
                    else:
                        arr = arr / 255.0
                if arr.ndim == 2:
                    std = np.std(arr)
                elif arr.ndim == 3 and arr.shape[-1] >= 3:
                    std = np.std(arr[:, :, :3])
                else:
                    std = np.std(arr)
                # Type-aware threshold: metalness/AO/height/mask maps
                # are legitimately near-uniform for many materials
                # (e.g. metalness=0 for non-metals, AO=1 for flat
                # surfaces). For these types, a constant value is
                # only a warning, not an error.
                solid_threshold = 0.004
                _basename = os.path.basename(image_path).lower()
                _uniform_ok_tags = (
                    "_metalness", "_ao", "_height", "_mask",
                    "_zones", "_envmask", "_emissive_mask",
                )
                _is_uniform_ok_type = any(
                    tag in _basename for tag in _uniform_ok_tags
                )
                if _is_uniform_ok_type:
                    solid_threshold = 0.001
                if std < solid_threshold:
                    msg = (
                        f"Image appears nearly solid color "
                        f"(std={std:.4f}, threshold={solid_threshold})"
                    )
                    if _is_uniform_ok_type:
                        # Uniform scalar maps are physically valid
                        # (e.g. metalness=0 for non-metals).
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        except Exception as e:
            errors.append(f"Failed to validate: {e}")

        return len(errors) == 0, errors, warnings

    @staticmethod
    def _to_scalar_map(arr: np.ndarray) -> np.ndarray:
        """Convert image array to a single-channel scalar map in [0, 1]."""
        if arr.ndim == 2:
            raw = arr.astype(np.float32, copy=False)
        else:
            rgb = ensure_rgb(arr).astype(np.float32, copy=False)
            raw = luminance_bt709(rgb, assume_srgb=False)

        # Pre-clip range check: warn if >1% of pixels are significantly out of [0, 1]
        out_of_range_ratio = float(np.mean((raw < -0.01) | (raw > 1.01)))
        if out_of_range_ratio > 0.01:
            logger.warning(
                "PBR scalar map has %.1f%% of pixels outside [0, 1] range "
                "before clipping (ratio=%.4f).",
                out_of_range_ratio * 100.0,
                out_of_range_ratio,
            )

        return np.clip(raw, 0.0, 1.0)

    @staticmethod
    def _map_stats(arr: np.ndarray) -> dict:
        """Compute robust scalar-map statistics."""
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        if flat.size == 0:
            return {
                "mean": 0.0,
                "stddev": 0.0,
                "p05": 0.0,
                "p95": 0.0,
                "midband_ratio": 0.0,
            }
        return {
            "mean": float(np.mean(flat)),
            "stddev": float(np.std(flat)),
            "p05": float(np.percentile(flat, 5)),
            "p95": float(np.percentile(flat, 95)),
            "midband_ratio": float(np.mean((flat > 0.2) & (flat < 0.8))),
        }

    def _check_material_semantics(self, record: AssetRecord, processed_paths: dict) -> dict:
        """Check roughness/metalness/AO plausibility against material category."""
        checks = {"passed": True, "warnings": [], "errors": [], "metrics": {}}
        strict = bool(self.cfg.strict_material_semantics)
        category = (record.material_category or "default").strip().lower() or "default"

        def _issue(msg: str):
            if strict:
                checks["errors"].append(msg)
            else:
                checks["warnings"].append(msg)

        roughness_path = processed_paths.get("roughness")
        if roughness_path and os.path.exists(roughness_path):
            roughness = self._to_scalar_map(
                load_image(roughness_path, max_pixels=self.config.max_image_pixels)
            )
            rough_stats = self._map_stats(roughness)
            checks["metrics"]["roughness"] = rough_stats
            if rough_stats["stddev"] < self.cfg.roughness_min_stddev:
                _issue(
                    "Roughness variation is very low "
                    f"(std={rough_stats['stddev']:.4f} < "
                    f"{self.cfg.roughness_min_stddev:.4f})."
                )

        ao_path = processed_paths.get("ao")
        if ao_path and os.path.exists(ao_path):
            ao = self._to_scalar_map(load_image(ao_path, max_pixels=self.config.max_image_pixels))
            ao_stats = self._map_stats(ao)
            checks["metrics"]["ao"] = ao_stats
            if ao_stats["stddev"] < self.cfg.ao_min_stddev:
                _issue(
                    "AO variation is very low "
                    f"(std={ao_stats['stddev']:.4f} < {self.cfg.ao_min_stddev:.4f})."
                )

        metalness_path = processed_paths.get("metalness")
        if metalness_path and os.path.exists(metalness_path):
            metalness = self._to_scalar_map(
                load_image(metalness_path, max_pixels=self.config.max_image_pixels)
            )
            metal_stats = self._map_stats(metalness)
            checks["metrics"]["metalness"] = metal_stats

            mean = metal_stats["mean"]
            mid = metal_stats["midband_ratio"]

            if (
                category in self._NON_METAL_CATEGORIES
                and mean > self.cfg.metalness_nonmetal_max_mean
            ):
                _issue(
                    f"Non-metal category '{category}' has high metalness mean "
                    f"({mean:.3f} > {self.cfg.metalness_nonmetal_max_mean:.3f})."
                )
            if (
                category in self._METAL_CATEGORIES
                and mean < self.cfg.metalness_metal_min_mean
            ):
                _issue(
                    f"Metal category '{category}' has low metalness mean "
                    f"({mean:.3f} < {self.cfg.metalness_metal_min_mean:.3f})."
                )
            if mid > self.cfg.metalness_midband_max_ratio:
                _issue(
                    "Metalness is heavily mid-range (non-binary) "
                    f"(ratio={mid:.3f} > {self.cfg.metalness_midband_max_ratio:.3f})."
                )

        checks["passed"] = len(checks["errors"]) == 0
        return checks

    def _check_physical_albedo(self, record: AssetRecord, processed_paths: dict) -> dict:
        """Validate albedo reflectance against physically plausible ranges."""
        checks = {"passed": True, "warnings": [], "errors": [], "metrics": {}}
        strict = bool(self.cfg.strict_material_semantics)
        albedo_path = processed_paths.get("albedo") or processed_paths.get("upscaled")
        if not albedo_path or not os.path.exists(albedo_path):
            checks["warnings"].append("Albedo plausibility check skipped (albedo missing).")
            return checks

        try:
            albedo = load_image(albedo_path, max_pixels=self.config.max_image_pixels)
            rgb = ensure_rgb(albedo).astype(np.float32, copy=False)
            linear = srgb_to_linear(np.clip(rgb, 0, 1))
            lum = luminance_bt709(linear, assume_srgb=False)

            min_l = float(np.min(lum))
            max_l = float(np.max(lum))
            black_ratio = float(np.mean(lum < self.cfg.albedo_linear_min))
            white_ratio = float(np.mean(lum > self.cfg.albedo_linear_max))

            checks["metrics"] = {
                "luminance_min": min_l,
                "luminance_max": max_l,
                "black_ratio": black_ratio,
                "white_ratio": white_ratio,
                "limits": {
                    "min": float(self.cfg.albedo_linear_min),
                    "max": float(self.cfg.albedo_linear_max),
                    "black_ratio_max": float(self.cfg.albedo_black_ratio_max),
                    "white_ratio_max": float(self.cfg.albedo_white_ratio_max),
                },
            }

            issues = []
            if black_ratio > self.cfg.albedo_black_ratio_max:
                issues.append(
                    "Albedo has too many near-black pixels "
                    f"({black_ratio:.2%} > {self.cfg.albedo_black_ratio_max:.2%})."
                )
            if white_ratio > self.cfg.albedo_white_ratio_max:
                issues.append(
                    "Albedo has too many near-white pixels "
                    f"({white_ratio:.2%} > {self.cfg.albedo_white_ratio_max:.2%})."
                )
            if min_l <= 0.001:
                issues.append("Albedo contains near-zero reflectance values.")
            if max_l >= 0.999:
                issues.append("Albedo contains near-one reflectance values.")

            if issues:
                if strict:
                    checks["errors"].extend(issues)
                else:
                    checks["warnings"].extend(issues)
        except Exception as exc:
            checks["warnings"].append(f"Albedo plausibility check failed: {exc}")

        checks["passed"] = len(checks["errors"]) == 0
        return checks

    def _check_heuristic_map_limitations(
        self, record: AssetRecord, processed_paths: dict
    ) -> dict:
        """Report risk when maps are synthesized heuristically from diffuse input."""
        result = {"warnings": [], "errors": []}
        if not (self.cfg.warn_on_heuristic_maps or self.cfg.fail_on_heuristic_maps):
            return result

        tex_type = TextureType.UNKNOWN
        try:
            tex_type = TextureType(record.texture_type)
        except Exception:
            pass

        if tex_type not in {TextureType.DIFFUSE, TextureType.ALBEDO, TextureType.UNKNOWN}:
            return result

        has_pbr = any(
            processed_paths.get(key) and os.path.exists(processed_paths[key])
            for key in ("roughness", "metalness", "ao")
        )
        has_normal = any(
            processed_paths.get(key) and os.path.exists(processed_paths[key])
            for key in ("normal", "height")
        )

        issues = []
        if has_pbr:
            issues.append(
                "PBR maps were synthesized from diffuse/albedo using heuristics; "
                "material accuracy is not guaranteed without artist review."
            )
        if has_normal:
            issues.append(
                "Normal/height maps were synthesized from diffuse/albedo; "
                "treat as draft quality unless validated against authored ground truth."
            )

        if not issues:
            return result

        if self.cfg.fail_on_heuristic_maps:
            result["errors"].extend(issues)
        else:
            result["warnings"].extend(issues)
        return result

    # ──────────────────────────────────────
    # Sphere Render Test (IMPLEMENTED)
    # ──────────────────────────────────────

    def _ensure_sphere_geometry(self, resolution: int):
        """Pre-compute sphere normals and UVs for rendering."""
        if self._sphere_normals is not None and self._sphere_normals.shape[0] == resolution:
            return

        res = resolution
        # Create sphere as a 2D grid of normals
        y_coords = np.linspace(-1, 1, res)
        x_coords = np.linspace(-1, 1, res)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Sphere mask: points inside unit circle
        r_sq = xx**2 + yy**2
        mask = r_sq <= 1.0

        # Normal = (x, y, sqrt(1 - x^2 - y^2))
        nz = np.zeros_like(xx)
        nz[mask] = np.sqrt(np.maximum(1.0 - r_sq[mask], 0.0))

        normals = np.stack([xx, -yy, nz], axis=-1)  # flip Y for screen space
        length = np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
        length = np.maximum(length, 1e-8)
        normals = normals / length

        # UV mapping -- proper spherical projection (equirectangular)
        uvs = np.zeros((res, res, 2), dtype=np.float32)
        # Use atan2/asin for correct spherical mapping instead of
        # orthographic (xx+1)*0.5 which distorts at edges.
        theta = np.arctan2(xx, np.maximum(nz, 1e-8))  # azimuth
        phi = np.arcsin(np.clip(-yy, -1.0, 1.0))       # elevation
        uvs[:, :, 0] = theta / (2.0 * np.pi) + 0.5    # U: [0, 1]
        uvs[:, :, 1] = phi / np.pi + 0.5               # V: [0, 1]

        self._sphere_normals = normals.astype(np.float32)
        self._sphere_uvs = uvs
        self._sphere_mask = mask

    def _render_test_sphere(self, record: AssetRecord,
                            processed_paths: dict) -> str:
        """Render the texture on a sphere with Cook-Torrance GGX PBR shading.

        Uses GGX microfacet NDF, Schlick-GGX geometry, Schlick Fresnel,
        and energy conservation (kS + kD <= 1).
        """
        res = self.cfg.sphere_resolution
        self._ensure_sphere_geometry(res)

        mask = self._sphere_mask
        normals = self._sphere_normals
        uvs = self._sphere_uvs

        # Load textures
        albedo_path = processed_paths.get("albedo") or processed_paths.get("upscaled")
        normal_path = processed_paths.get("normal")
        roughness_path = processed_paths.get("roughness")

        if not albedo_path or not os.path.exists(albedo_path):
            return None

        albedo_tex = load_image(albedo_path, max_pixels=self.config.max_image_pixels)
        albedo_rgb = ensure_rgb(albedo_tex)
        # Downsample to avoid loading full-res textures for small preview
        _max_preview = res * 2
        if albedo_rgb.shape[0] > _max_preview or albedo_rgb.shape[1] > _max_preview:
            _scale = _max_preview / max(albedo_rgb.shape[0], albedo_rgb.shape[1])
            _new_h = max(int(albedo_rgb.shape[0] * _scale), 1)
            _new_w = max(int(albedo_rgb.shape[1] * _scale), 1)
            albedo_rgb = cv2.resize(albedo_rgb, (_new_w, _new_h), interpolation=cv2.INTER_AREA)

        # Sample textures at UV coordinates with bilinear interpolation
        def sample_texture(tex, uvs_arr):
            h, w = tex.shape[:2]
            # Use cv2.remap for proper bilinear interpolation
            map_x = (uvs_arr[:, :, 0] * (w - 1)).astype(np.float32)
            map_y = (uvs_arr[:, :, 1] * (h - 1)).astype(np.float32)
            tex_uint8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
            sampled = cv2.remap(
                tex_uint8, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
            return sampled.astype(np.float32) / 255.0

        albedo_sampled = sample_texture(albedo_rgb, uvs)

        # Sample and apply normal map via TBN (tangent-bitangent-normal) transform
        if normal_path and os.path.exists(normal_path):
            normal_tex = load_image(normal_path, max_pixels=self.config.max_image_pixels)
            normal_rgb = ensure_rgb(normal_tex)
            # Downsample to avoid loading full-res textures for small preview
            if normal_rgb.shape[0] > _max_preview or normal_rgb.shape[1] > _max_preview:
                _scale = _max_preview / max(normal_rgb.shape[0], normal_rgb.shape[1])
                _new_h = max(int(normal_rgb.shape[0] * _scale), 1)
                _new_w = max(int(normal_rgb.shape[1] * _scale), 1)
                normal_rgb = cv2.resize(normal_rgb, (_new_w, _new_h), interpolation=cv2.INTER_AREA)
            normal_sampled = sample_texture(normal_rgb, uvs)
            # Decode tangent-space normal from [0,1] to [-1,1]
            n_ts = normal_sampled * 2.0 - 1.0

            # Build per-pixel TBN basis from sphere geometry:
            # N = sphere normal (already computed)
            N = normals  # (res, res, 3)
            # Tangent: dP/du ~ cross(up, N), with up = (0,1,0) for most of
            # the sphere; fall back to (1,0,0) where N is near (0,±1,0).
            up = np.zeros_like(N)
            up[:, :, 1] = 1.0  # (0, 1, 0)
            T = np.cross(up, N)
            t_len = np.sqrt(np.sum(T ** 2, axis=-1, keepdims=True))
            degenerate = (t_len < 1e-6).squeeze(-1)
            # Where T is degenerate, use right vector as up
            up2 = np.zeros_like(N)
            up2[:, :, 0] = 1.0  # (1, 0, 0)
            T[degenerate] = np.cross(up2, N)[degenerate]
            t_len = np.sqrt(np.sum(T ** 2, axis=-1, keepdims=True))
            T = T / np.maximum(t_len, 1e-8)
            # Bitangent: B = cross(N, T)
            B = np.cross(N, T)
            b_len = np.sqrt(np.sum(B ** 2, axis=-1, keepdims=True))
            B = B / np.maximum(b_len, 1e-8)

            # Transform tangent-space normal to world space: world = T*nx + B*ny + N*nz
            world_n = (
                T * n_ts[:, :, 0:1]
                + B * n_ts[:, :, 1:2]
                + N * n_ts[:, :, 2:3]
            )
            length = np.sqrt(np.sum(world_n ** 2, axis=-1, keepdims=True))
            render_normals = world_n / np.maximum(length, 1e-8)
        else:
            render_normals = normals

        # Roughness
        if roughness_path and os.path.exists(roughness_path):
            rough_tex = load_image(roughness_path, max_pixels=self.config.max_image_pixels)
            # Downsample to avoid loading full-res textures for small preview
            if rough_tex.shape[0] > _max_preview or rough_tex.shape[1] > _max_preview:
                _scale = _max_preview / max(rough_tex.shape[0], rough_tex.shape[1])
                _new_h = max(int(rough_tex.shape[0] * _scale), 1)
                _new_w = max(int(rough_tex.shape[1] * _scale), 1)
                rough_tex = cv2.resize(rough_tex, (_new_w, _new_h), interpolation=cv2.INTER_AREA)
            if rough_tex.ndim == 2:
                rough_sampled = sample_texture(
                    np.stack([rough_tex] * 3, axis=-1), uvs
                )[:, :, 0]
            else:
                rough_sampled = np.mean(
                    sample_texture(ensure_rgb(rough_tex), uvs), axis=-1
                )
        else:
            rough_sampled = np.full((res, res), 0.5, dtype=np.float32)

        # Metalness (drives Fresnel F0)
        metalness_path = processed_paths.get("metalness")
        if metalness_path and os.path.exists(metalness_path):
            metal_tex = load_image(metalness_path, max_pixels=self.config.max_image_pixels)
            # Downsample to avoid loading full-res textures for small preview
            if metal_tex.shape[0] > _max_preview or metal_tex.shape[1] > _max_preview:
                _scale = _max_preview / max(metal_tex.shape[0], metal_tex.shape[1])
                _new_h = max(int(metal_tex.shape[0] * _scale), 1)
                _new_w = max(int(metal_tex.shape[1] * _scale), 1)
                metal_tex = cv2.resize(metal_tex, (_new_w, _new_h), interpolation=cv2.INTER_AREA)
            if metal_tex.ndim == 2:
                metal_sampled = sample_texture(
                    np.stack([metal_tex] * 3, axis=-1), uvs
                )[:, :, 0]
            else:
                metal_sampled = np.mean(
                    sample_texture(ensure_rgb(metal_tex), uvs), axis=-1
                )
        else:
            metal_sampled = np.zeros((res, res), dtype=np.float32)

        # PBR F0: dielectrics ~0.04, metals use albedo color
        f0_dielectric = 0.04
        f0 = (
            f0_dielectric * (1.0 - metal_sampled)[:, :, np.newaxis]
            + albedo_sampled * metal_sampled[:, :, np.newaxis]
        )

        # Clamp roughness to avoid division issues in GGX
        alpha = np.maximum(rough_sampled ** 2, 0.001)
        alpha_sq = alpha ** 2

        # Render from multiple light angles using Cook-Torrance GGX
        output_panels = []
        light_dirs = [
            np.array([0.5, 0.5, 0.7]),
            np.array([-0.5, 0.3, 0.7]),
            np.array([0.0, -0.5, 0.8]),
            np.array([0.7, 0.0, 0.5]),
        ][:self.cfg.sphere_light_angles]
        view_dir = np.array([0.0, 0.0, 1.0])

        for light_dir in light_dirs:
            light_dir = light_dir / np.linalg.norm(light_dir)
            half_dir = light_dir + view_dir
            half_dir = half_dir / np.linalg.norm(half_dir)

            ndotl = np.maximum(
                np.sum(render_normals * light_dir[None, None, :], axis=-1), 0.0
            )
            ndotv = np.maximum(render_normals[:, :, 2], 0.0)  # view = (0,0,1)
            ndoth = np.maximum(
                np.sum(render_normals * half_dir[None, None, :], axis=-1), 0.0
            )
            vdoth = np.maximum(
                np.dot(view_dir, half_dir), 0.0
            )

            # GGX/Trowbridge-Reitz NDF
            denom = ndoth ** 2 * (alpha_sq - 1.0) + 1.0
            D = alpha_sq / (np.pi * denom ** 2 + 1e-7)

            # Schlick-GGX geometry (Smith method)
            k = (rough_sampled + 1.0) ** 2 / 8.0
            G1_l = ndotl / (ndotl * (1.0 - k) + k + 1e-7)
            G1_v = ndotv / (ndotv * (1.0 - k) + k + 1e-7)
            G = G1_l * G1_v

            # Schlick Fresnel
            F = f0 + (1.0 - f0) * (1.0 - vdoth) ** 5

            # Cook-Torrance specular
            spec_denom = 4.0 * ndotv * ndotl + 1e-7
            specular = (
                D[:, :, np.newaxis] * G[:, :, np.newaxis] * F / spec_denom[:, :, np.newaxis]
            )

            # Energy conservation: diffuse reduced by Fresnel
            kd = (1.0 - F) * (1.0 - metal_sampled[:, :, np.newaxis])

            # Combine: Lambert diffuse + Cook-Torrance specular
            ambient = 0.03
            diffuse_term = kd * albedo_sampled / np.pi
            color = (diffuse_term + specular) * ndotl[:, :, np.newaxis] + ambient
            color = np.clip(color, 0, 1)

            # Apply sphere mask
            bg = np.full((res, res, 3), 0.15, dtype=np.float32)
            result = np.where(mask[:, :, np.newaxis], color, bg)
            output_panels.append(result)

        # Stitch panels side by side
        divider = np.full((res, 2, 3), 0.3, dtype=np.float32)
        parts = []
        for i, panel in enumerate(output_panels):
            parts.append(panel)
            if i < len(output_panels) - 1:
                parts.append(divider)
        combined = np.hstack(parts)

        # Add label
        combined_uint8 = (combined * 255).astype(np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_uint8, record.filename, (10, 20),
                    font, 0.5, (200, 200, 200), 1)

        out_path = os.path.join(
            self.config.comparison_dir,
            f"sphere_{os.path.splitext(record.filename)[0]}.png"
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_image(combined_uint8.astype(np.float32) / 255.0, out_path)
        logger.debug(f"Sphere render saved: {out_path}")
        return out_path

    def _regression_diff(self, previous_path: str, current_path: str,
                         record: AssetRecord) -> tuple:
        """Compare current output against previous baseline.

        Returns (regressed: bool, diff_score: float, diff_image_path: str|None).
        A regression is flagged when the mean absolute difference exceeds
        max_allowed_diff.
        """
        try:
            prev = load_image(previous_path, max_pixels=self.config.max_image_pixels)
            curr = load_image(current_path, max_pixels=self.config.max_image_pixels)
            prev_rgb = ensure_rgb(prev)
            curr_rgb = ensure_rgb(curr)

            # Resize to match if dimensions differ
            if prev_rgb.shape[:2] != curr_rgb.shape[:2]:
                target_h, target_w = curr_rgb.shape[:2]
                prev_rgb = cv2.resize(
                    prev_rgb.astype(np.float32), (target_w, target_h),
                    interpolation=cv2.INTER_AREA
                )

            diff = np.abs(curr_rgb - prev_rgb)
            diff_score = float(np.mean(diff))
            regressed = diff_score > self.cfg.max_allowed_diff

            # Save diff heatmap
            diff_path = None
            if regressed:
                heatmap = np.mean(diff, axis=-1)
                heatmap = np.clip(heatmap * 5.0, 0, 1)  # Amplify for visibility
                heatmap_color = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                diff_path = os.path.join(
                    self.config.comparison_dir,
                    f"regression_{os.path.splitext(record.filename)[0]}.png"
                )
                os.makedirs(os.path.dirname(diff_path) or ".", exist_ok=True)
                cv2.imwrite(diff_path, heatmap_color)
                logger.warning(
                    f"Regression detected for {record.filename}: "
                    f"diff={diff_score:.4f} > {self.cfg.max_allowed_diff}"
                )

            return regressed, diff_score, diff_path

        except Exception as e:
            logger.error(f"Regression diff failed for {record.filename}: {e}", exc_info=True)
            return False, 0.0, None

    def _generate_comparison(self, original_path: str, processed_path: str,
                             record: AssetRecord) -> str:
        try:
            original = load_image(original_path, max_pixels=self.config.max_image_pixels)
            processed = load_image(processed_path, max_pixels=self.config.max_image_pixels)
            orig_rgb = ensure_rgb(original)
            proc_rgb = ensure_rgb(processed)

            proc_h, proc_w = proc_rgb.shape[:2]
            orig_resized = cv2.resize(
                (orig_rgb * 255).astype(np.uint8),
                (proc_w, proc_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.float32) / 255.0

            divider = np.ones((proc_h, 4, 3), dtype=np.float32) * 0.8
            comparison = np.hstack([orig_resized, divider, proc_rgb])

            comp_uint8 = (comparison * 255).astype(np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comp_uint8, "ORIGINAL", (10, 30), font, 0.8, (255, 255, 0), 2)
            cv2.putText(comp_uint8, "PROCESSED", (proc_w + 14, 30), font, 0.8, (0, 255, 0), 2)
            orig_dims = f"{record.original_width}x{record.original_height}"
            info = f"{record.filename} | {orig_dims} -> {proc_w}x{proc_h}"
            cv2.putText(comp_uint8, info, (10, proc_h - 10), font, 0.5, (200, 200, 200), 1)

            comp_path = os.path.join(
                self.config.comparison_dir,
                f"comparison_{os.path.splitext(record.filename)[0]}.png"
            )
            os.makedirs(os.path.dirname(comp_path) or ".", exist_ok=True)
            save_image(comp_uint8.astype(np.float32) / 255.0, comp_path)
            return comp_path

        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            return None

    def generate_report(self, all_reports: list, output_path: str):
        """Write a human-readable validation summary report to disk."""
        total = len(all_reports)
        passed = sum(1 for r in all_reports if r["passed"])
        failed_count = total - passed
        total_errors = sum(len(r["errors"]) for r in all_reports)

        lines = [
            "=" * 60,
            "ASSET PIPELINE VALIDATION REPORT",
            "=" * 60,
            f"Total assets:    {total}",
            f"Passed:          {passed}/{total}",
            f"Failed:          {failed_count}/{total}",
            f"Total errors:    {total_errors}",
            "=" * 60, "",
        ]

        failed = [r for r in all_reports if not r["passed"]]

        if failed:
            lines.append("FAILED ASSETS:")
            for r in failed:
                lines.append(f"  {r['filename']}:")
                for e in r.get("errors", []):
                    lines.append(f"    ERROR: {e}")
            lines.append("")
        else:
            lines.append("All assets passed validation.")
            lines.append("")

        report_text = "\n".join(lines)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_text)
        logger.info(f"Validation report saved: {output_path}")
        print(report_text)

    def generate_tiling_quality_report(self, all_reports: list, output_path: str):
        """Write per-asset tiling quality scores and review flags to JSON."""
        entries = []
        for report in all_reports:
            tiling = report.get("checks", {}).get("tiling")
            if not isinstance(tiling, dict):
                continue
            seam_score = float(tiling.get("seam_score", 0.0))
            quality_score = float(tiling.get(
                "quality_score", self._tiling_quality_score(seam_score),
            ))
            entries.append({
                "filename": report.get("filename"),
                "passed": bool(tiling.get("passed", False)),
                "seam_score": seam_score,
                "quality_score": quality_score,
            })

        if not entries:
            return

        entries.sort(key=lambda item: item["quality_score"])
        top_n = int(max(self.tiling_cfg.auto_flag_top_n, 0))
        warn_score = float(self.tiling_cfg.warn_score)
        fail_score = float(self.tiling_cfg.fail_score)

        for idx, item in enumerate(entries):
            low_quality = item["quality_score"] < fail_score
            auto_top = idx < top_n and item["quality_score"] < warn_score
            item["flagged_for_review"] = bool(low_quality or auto_top)

        payload = {
            "total_scored": len(entries),
            "warn_score": warn_score,
            "fail_score": fail_score,
            "auto_flag_top_n": top_n,
            "entries": entries,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Tiling quality report saved: %s", output_path)
