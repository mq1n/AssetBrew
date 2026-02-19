"""Generate derived PBR material maps from diffuse/albedo textures.

This phase supports improved de-lighting, material-zone masking, and
heuristic synthesis of roughness, metalness, AO, and optional gloss maps.
"""

import logging
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord,
    ensure_rgb,
    get_intermediate_path,
    linear_to_srgb,
    load_image,
    luminance_bt709,
    save_image,
    srgb_to_linear,
)

logger = logging.getLogger("asset_pipeline.pbr")


def _large_scale_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that handles very large sigma efficiently."""
    if sigma <= 100:
        return gaussian_filter(img, sigma=sigma)
    factor = sigma / 100.0
    h, w = img.shape[:2]
    small_h = max(int(h / factor), 1)
    small_w = max(int(w / factor), 1)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blurred = gaussian_filter(small, sigma=100.0)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


class PBRGenerator:
    """Generate PBR material maps from diffuse/albedo textures.

    **Important**: All generated PBR maps (roughness, metalness, AO) are
    *heuristic approximations* derived from image analysis, not physically
    measured material properties.  They are intended as starting-point
    drafts that will likely require manual artist correction for
    production use.  In particular:

    - Roughness is estimated from local pixel variance, which conflates
      visual texture detail with surface microstructure.
    - AO is estimated from luminance, which conflates dark albedo with
      geometric occlusion (may double-darken at runtime).
    - Metalness uses saturation/hue heuristics and produces soft mid-range
      values rather than the binary 0/1 that PBR theory expects.
    """

    _ZONE_ORDER = ("metal", "cloth", "leather", "skin")

    def __init__(self, config: PipelineConfig):
        """Initialize generator with current pipeline configuration."""
        self.config = config
        self.cfg = config.pbr
        self._warned_heuristic = False

    def process(self, record: AssetRecord, upscaled_path: str = None) -> dict:
        """Create PBR outputs for a single supported texture asset."""
        tex_type = TextureType(record.texture_type)
        result = {
            "albedo": None,
            "roughness": None,
            "metalness": None,
            "ao": None,
            "gloss": None,
            "zone_mask": None,
            "skipped": False,
            "is_heuristic": False,
            "error": None,
        }

        if tex_type not in (TextureType.DIFFUSE, TextureType.ALBEDO, TextureType.UNKNOWN):
            result["skipped"] = True
            logger.debug(
                "Skipping PBR generation for %s: unsupported texture type %s",
                record.filename, tex_type.value
            )
            return result

        if upscaled_path and os.path.exists(upscaled_path):
            source_path = upscaled_path
        else:
            source_path = os.path.join(self.config.input_dir, record.filepath)

        try:
            if not self._warned_heuristic:
                self._warned_heuristic = True
                logger.warning(
                    "PBR maps are heuristic approximations derived from image "
                    "analysis — not physically measured material properties. "
                    "Generated roughness, metalness, and AO maps should be "
                    "treated as drafts that require manual artist review for "
                    "production use."
                )
            img = load_image(source_path, max_pixels=self.config.max_image_pixels)
            rgb_srgb = ensure_rgb(img)

            if self.cfg.delight_diffuse:
                albedo_srgb = self._delight(rgb_srgb)
                out = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_albedo",
                    ext=".png",
                )
                save_image(albedo_srgb, out)
                result["albedo"] = out
            else:
                albedo_srgb = np.clip(rgb_srgb, 0, 1).astype(np.float32, copy=False)

            albedo_linear = srgb_to_linear(albedo_srgb)
            zones = self._detect_material_zones(albedo_srgb, record.material_category)

            if self.cfg.material_zone_masks:
                zone_rgba = np.stack(
                    [zones[name] for name in self._ZONE_ORDER],
                    axis=-1,
                ).astype(np.float32)
                zone_path = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_zones",
                    ext=".png",
                )
                save_image(zone_rgba, zone_path, bits=16)
                result["zone_mask"] = zone_path

            roughness = None
            if self.cfg.generate_roughness:
                roughness = self._generate_roughness(albedo_linear, record.material_category)
                if self.cfg.apply_zone_pbr_adjustments:
                    roughness = self._apply_zone_roughness_adjustments(roughness, zones)
                out = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_roughness",
                    ext=".png",
                )
                save_image(roughness, out, bits=16)
                result["roughness"] = out

            if self.cfg.generate_metalness:
                metalness = self._generate_metalness(
                    albedo_srgb,
                    record.material_category,
                    linear_luminance=luminance_bt709(albedo_linear, assume_srgb=False),
                )
                if self.cfg.apply_zone_pbr_adjustments:
                    metalness = self._apply_zone_metalness_adjustments(metalness, zones)
                out = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_metalness",
                    ext=".png",
                )
                save_image(metalness, out, bits=16)
                result["metalness"] = out

            if self.cfg.generate_ao:
                ao = self._generate_ao(albedo_linear)
                out = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_ao",
                    ext=".png",
                )
                save_image(ao, out, bits=16)
                result["ao"] = out

            if self.cfg.generate_gloss and roughness is not None:
                gloss = self._roughness_to_gloss(roughness)
                out = get_intermediate_path(
                    record.filepath,
                    "02_pbr",
                    self.config.intermediate_dir,
                    suffix="_gloss",
                    ext=".png",
                )
                save_image(gloss, out, bits=16)
                result["gloss"] = out

            result["is_heuristic"] = True
            logger.info(f"PBR maps generated for {record.filename}")

        except Exception as e:
            logger.error(f"PBR generation failed for {record.filename}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    @staticmethod
    def _resolution_scale(img: np.ndarray, base: int = 1024) -> float:
        h, w = img.shape[:2]
        return max(max(h, w) / float(base), 0.25)

    @staticmethod
    def _odd_window(value: float) -> int:
        size = max(int(round(value)), 3)
        if size % 2 == 0:
            size += 1
        return size

    @staticmethod
    def _robust_normalize(arr: np.ndarray, sigma_multiplier: float = 3.0) -> np.ndarray:
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        denom = mean + sigma_multiplier * std
        if denom > 1e-8:
            return np.clip(arr / denom, 0, 1)
        return np.zeros_like(arr, dtype=np.float32)

    def _delight(self, rgb: np.ndarray) -> np.ndarray:
        """Dispatch to configured de-lighting method."""
        method = (self.cfg.delight_method or "multifrequency").strip().lower()
        if method == "gaussian":
            return self._delight_gaussian(rgb)
        return self._delight_multifrequency(rgb)

    def _delight_gaussian(self, rgb: np.ndarray) -> np.ndarray:
        """Legacy LAB Gaussian de-lighting baseline.

        Uses float32 LAB conversion to avoid posterization artefacts that
        the previous uint8 round-trip caused in shadow regions.
        """
        # Convert via uint8 so OpenCV applies the sRGB transfer correctly,
        # then immediately promote to float32 for all computation.
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

        h, w = lab.shape[:2]
        l_channel = lab[:, :, 0]
        scales = [max(h, w) // 8, max(h, w) // 4, max(h, w) // 2]

        illumination_estimates = []
        for scale in scales:
            sigma = max(scale / 3.0, 1.0)
            illumination_estimates.append(_large_scale_blur(l_channel, sigma=sigma))

        weights = [0.2, 0.5, 0.3]
        illumination = np.zeros_like(l_channel)
        for est, wt in zip(illumination_estimates, weights):
            illumination += est * wt

        mean_l = np.mean(illumination)
        if mean_l > 1.0:
            correction = mean_l / np.maximum(illumination, 1.0)
            correction = np.clip(correction, 0.5, 2.0)
            l_corrected = np.clip(l_channel * correction, 0, 255)
        else:
            l_corrected = l_channel

        blend = float(np.clip(self.cfg.delight_strength, 0.0, 1.0))
        lab[:, :, 0] = l_corrected * blend + l_channel * (1.0 - blend)
        # Stay in float32 for the inverse conversion to avoid 8-bit
        # quantization/posterization in shadows.
        lab_clipped = np.clip(lab, 0, 255).astype(np.float32)
        # OpenCV treats float32 Lab input as already-scaled [0,100]/[-127,127],
        # but our Lab values are in uint8 range [0,255].  Roundtrip through
        # uint8 for the inverse conversion to get correct sRGB output.
        result = cv2.cvtColor(lab_clipped.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return (result.astype(np.float32) / 255.0).astype(np.float32, copy=False)

    def _delight_multifrequency(self, rgb: np.ndarray) -> np.ndarray:
        """Multi-frequency de-lighting in linear space with detail preservation."""
        rgb_srgb = np.clip(rgb, 0, 1).astype(np.float32, copy=False)
        rgb_linear = srgb_to_linear(rgb_srgb)
        lum = luminance_bt709(rgb_linear, assume_srgb=False)
        scale = self._resolution_scale(rgb_linear)

        low_sigma = max(self.cfg.delight_low_frequency_sigma * scale, 1.0)
        mid_sigma = max(self.cfg.delight_mid_frequency_sigma * scale, 0.75)
        low = _large_scale_blur(lum, sigma=low_sigma)
        mid = _large_scale_blur(lum, sigma=mid_sigma)
        illum = 0.7 * low + 0.3 * mid

        mean_illum = float(np.mean(illum)) + 1e-6
        illum_norm = illum / mean_illum

        shadow_lift = float(np.clip(self.cfg.delight_shadow_lift, 0.0, 1.0))
        highlight_sup = float(np.clip(self.cfg.delight_highlight_suppress, 0.0, 1.0))
        inv_illum = 1.0 / np.maximum(illum_norm, 1e-3)

        shadow_term = 1.0 + shadow_lift * np.clip(inv_illum - 1.0, 0.0, 2.0)
        highlight_term = 1.0 - highlight_sup * np.clip(illum_norm - 1.0, 0.0, 1.0)
        correction = np.clip(shadow_term * highlight_term, 0.6, 1.6)

        corrected_lum = np.clip(lum * correction, 0.0, 1.0)
        ratio = corrected_lum / np.maximum(lum, 1e-5)
        corrected_linear = np.clip(rgb_linear * ratio[:, :, None], 0.0, 1.0)

        strength = float(np.clip(self.cfg.delight_strength, 0.0, 1.0))
        blended_linear = np.clip(
            corrected_linear * strength + rgb_linear * (1.0 - strength),
            0.0,
            1.0,
        )
        return np.clip(linear_to_srgb(blended_linear), 0.0, 1.0).astype(np.float32)

    def _detect_material_zones(
        self,
        albedo_srgb: np.ndarray,
        material_category: str,
    ) -> dict[str, np.ndarray]:
        """Generate soft material-zone masks for mixed-material textures."""
        rgb = np.clip(ensure_rgb(albedo_srgb), 0, 1).astype(np.float32, copy=False)
        rgb_u8 = (rgb * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[:, :, 0] / 180.0
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0

        lum = luminance_bt709(srgb_to_linear(rgb), assume_srgb=False)
        local_mean = uniform_filter(lum, size=7)
        local_sq_mean = uniform_filter(lum ** 2, size=7)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        metal = (
            (sat < 0.22)
            & (val > 0.22)
            & (lum > 0.16)
            & (local_std < 0.11)
        ).astype(np.float32)
        cloth = (
            (sat > 0.12)
            & (val > 0.10)
            & (val < 0.85)
            & (local_std > 0.04)
        ).astype(np.float32)
        leather = (
            (hue > 0.03)
            & (hue < 0.15)
            & (sat > 0.18)
            & (sat < 0.75)
            & (val > 0.12)
            & (val < 0.80)
        ).astype(np.float32)
        skin = (
            (hue > 0.02)
            & (hue < 0.14)
            & (sat > 0.16)
            & (sat < 0.65)
            & (val > 0.30)
            & (rgb[:, :, 0] > rgb[:, :, 1])
            & (rgb[:, :, 1] > rgb[:, :, 2] * 0.8)
        ).astype(np.float32)

        cat = (material_category or "default").strip().lower()
        if cat in {"metal", "iron", "silver", "gold", "copper"}:
            metal = np.clip(np.maximum(metal, 0.65), 0.0, 1.0)
        elif cat == "leather":
            leather = np.clip(np.maximum(leather, 0.6), 0.0, 1.0)
        elif cat == "skin":
            skin = np.clip(np.maximum(skin, 0.6), 0.0, 1.0)
        elif cat in {"fabric", "cloth"}:
            cloth = np.clip(np.maximum(cloth, 0.6), 0.0, 1.0)

        zone_maps = {
            "metal": metal,
            "cloth": cloth,
            "leather": leather,
            "skin": skin,
        }

        blur = int(max(self.cfg.zone_blur_radius, 0))
        if blur > 0:
            for zone_name in zone_maps:
                zone_maps[zone_name] = cv2.GaussianBlur(
                    zone_maps[zone_name],
                    (0, 0),
                    sigmaX=max(blur / 2.0, 0.5),
                    sigmaY=max(blur / 2.0, 0.5),
                )

        for zone_name in zone_maps:
            zone_maps[zone_name] = np.clip(zone_maps[zone_name], 0, 1).astype(np.float32)

        return zone_maps

    def _apply_zone_roughness_adjustments(
        self,
        roughness: np.ndarray,
        zones: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply zone-specific roughness bias for mixed materials."""
        out = np.clip(roughness.astype(np.float32, copy=False), 0, 1)
        for zone_name, bias in self.cfg.zone_roughness_bias.items():
            mask = zones.get(zone_name)
            if mask is None:
                continue
            out = out + mask * float(bias)
        return np.clip(out, 0.04, 1.0).astype(np.float32)

    def _apply_zone_metalness_adjustments(
        self,
        metalness: np.ndarray,
        zones: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply zone-specific metalness floors for mixed materials."""
        out = np.clip(metalness.astype(np.float32, copy=False), 0, 1)
        for zone_name, floor in self.cfg.zone_metalness_floor.items():
            mask = zones.get(zone_name)
            if mask is None:
                continue
            out = np.maximum(out, np.clip(mask * float(floor), 0, 1))
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _roughness_to_gloss(roughness: np.ndarray) -> np.ndarray:
        """Convert roughness to gloss/smoothness."""
        return np.clip(1.0 - roughness.astype(np.float32, copy=False), 0, 1).astype(np.float32)

    def _generate_roughness(self, albedo: np.ndarray, material_category: str) -> np.ndarray:
        """Roughness from multi-scale variance, Sobel edges, and Laplacian detail."""
        # Input is linear-space albedo. Convert luminance to perceptual (gamma)
        # space so that variance computation gives dark and light regions equal
        # weight — linear-space variance under-represents detail in shadows.
        gray_linear = luminance_bt709(albedo, assume_srgb=False)
        gray = np.power(np.clip(gray_linear, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)
        scale = self._resolution_scale(albedo)

        var_windows = [
            self._odd_window(5 * scale),
            self._odd_window(9 * scale),
            self._odd_window(15 * scale),
        ]
        variance_maps = []
        for window in var_windows:
            local_mean = uniform_filter(gray, size=window)
            local_sq_mean = uniform_filter(gray ** 2, size=window)
            local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
            variance_maps.append(np.sqrt(local_var))

        multi_var = (
            0.4 * variance_maps[0]
            + 0.35 * variance_maps[1]
            + 0.25 * variance_maps[2]
        )
        multi_var = self._robust_normalize(multi_var)

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_map = self._robust_normalize(edge_map)

        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lap_win = self._odd_window(11 * scale)
        laplacian_local = uniform_filter(laplacian, size=lap_win)
        laplacian_local = self._robust_normalize(laplacian_local)

        vw = self.cfg.roughness_variance_weight
        sw = self.cfg.roughness_sobel_weight
        lw = self.cfg.roughness_laplacian_weight
        total_w = vw + sw + lw
        if total_w > 0:
            vw, sw, lw = vw / total_w, sw / total_w, lw / total_w
        else:
            vw, sw, lw = 0.5, 0.3, 0.2

        combined = vw * multi_var + sw * edge_map + lw * laplacian_local
        combined = self._robust_normalize(combined, sigma_multiplier=2.5)

        material_base = self.cfg.material_roughness_defaults.get(
            material_category,
            self.cfg.material_roughness_defaults["default"],
        )
        base = np.clip(material_base + (self.cfg.roughness_base_value - 0.5), 0.0, 1.0)
        roughness = np.clip(base + (combined - 0.5) * 0.8, 0.04, 1.0)
        roughness = gaussian_filter(roughness, sigma=max(0.5, 1.5 * scale)).astype(np.float32)
        return roughness

    def _generate_metalness(
        self,
        albedo: np.ndarray,
        material_category: str,
        linear_luminance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Metalness with Fresnel-aware color analysis and edge-preserving smoothing.

        All thresholds are evaluated in sRGB/perceptual space to avoid
        inconsistent zone boundaries from mixing sRGB ``brightness`` (HSV V)
        with linear ``luminance``.
        """
        h, w = albedo.shape[:2]
        base_metal = self.cfg.material_metalness_defaults.get(
            material_category,
            self.cfg.material_metalness_defaults["default"],
        )

        metalness = np.full((h, w), base_metal, dtype=np.float32)
        rgb_uint8 = (np.clip(albedo, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] /= 180.0
        hsv[:, :, 1:] /= 255.0
        brightness = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        hue = hsv[:, :, 0]

        # Use perceptual (sRGB) luminance for all thresholds so that
        # brightness and luminance comparisons are in the same domain.
        srgb_luminance = luminance_bt709(albedo, assume_srgb=False)

        local_mean = uniform_filter(srgb_luminance, size=7)
        local_sq_mean = uniform_filter(srgb_luminance ** 2, size=7)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        low_texture = local_std < 0.08

        if base_metal > 0.5:
            dark_mask = srgb_luminance < 0.08
            metalness[dark_mask] *= 0.1
            bright_desat = (srgb_luminance > 0.85) & (saturation < 0.05)
            metalness[bright_desat] *= 0.5
            high_sat = saturation > 0.7
            metalness[high_sat] *= 0.4
        else:
            # Only apply pixel-based metal detection for the "default"
            # (unknown) category or categories with some metallic
            # plausibility.  Known non-metallic categories (wood, fabric,
            # skin, etc.) should not have silver/iron/gold boosted based
            # on pixel statistics alone — a desaturated wood texture at
            # mid brightness looks identical to silver to the detector.
            _metal_plausible = material_category in {
                "default", "metal", "iron", "gold", "silver", "copper",
                "glass", "ceramic", "cobblestone", "concrete",
            }
            if _metal_plausible:
                gold_copper = (
                    (hue > 0.06)
                    & (hue < 0.18)
                    & (saturation > 0.25)
                    & (saturation < 0.7)
                    & (brightness > 0.55)
                    & (srgb_luminance > 0.5)
                    & ((albedo[:, :, 0] - albedo[:, :, 2]) > 0.2)
                    & low_texture
                )
                metalness[gold_copper] = np.maximum(metalness[gold_copper], 0.8)

                silver = (
                    (saturation < 0.1)
                    & (brightness > 0.55)
                    & (srgb_luminance > 0.45)
                    & low_texture
                )
                metalness[silver] = np.maximum(metalness[silver], 0.7)

                iron = (
                    (saturation < 0.1)
                    & (brightness > 0.35)
                    & (brightness < 0.65)
                    & (srgb_luminance > 0.18)
                    & (srgb_luminance < 0.55)
                    & low_texture
                )
                metalness[iron] = np.maximum(metalness[iron], 0.55)

        metalness = np.clip(metalness, 0, 1).astype(np.float32)
        metalness = cv2.bilateralFilter(metalness, d=9, sigmaColor=0.2, sigmaSpace=50)

        m = np.clip(metalness, 0, 1)
        metalness = np.where(
            m > 0.5,
            1.0 - 0.5 * np.power(2.0 * (1.0 - m), 2.0),
            0.5 * np.power(2.0 * m, 2.0),
        )
        metalness = np.clip(metalness, 0, 1)
        if self.cfg.metalness_binarize:
            metalness = np.where(
                metalness > self.cfg.metalness_threshold, 1.0, 0.0
            )
        return metalness.astype(np.float32)

    def _generate_ao(self, albedo: np.ndarray) -> np.ndarray:
        """AO via multi-scale *relative* cavity detection.

        Uses the ratio (blurred / local) rather than a raw difference so that
        dark albedo regions are not misinterpreted as geometric occlusion.
        This avoids the double-darkening artefact at runtime when the engine
        multiplies AO on top of already-dark albedo.
        """
        gray = luminance_bt709(albedo, assume_srgb=False)
        scale = self._resolution_scale(albedo)

        base_radius = max(int(self.cfg.ao_radius), 1)
        base_scales = [
            max(3, base_radius // 2),
            max(5, base_radius),
            max(7, base_radius * 2),
            max(9, base_radius * 3),
        ]
        scales = [self._odd_window(s * scale) for s in base_scales]
        cavity_maps = []
        for s in scales:
            sigma = max(min(s / 2.0, 32.0 * scale), 0.5)
            blurred = gaussian_filter(gray, sigma=sigma)
            # Relative cavity: how much darker is the pixel compared to its
            # neighbourhood *proportionally*.  Values near 1.0 mean "same as
            # surroundings"; values < 1.0 indicate a cavity.  Using a ratio
            # decouples the result from absolute albedo brightness.
            ratio = gray / np.maximum(blurred, 1e-6)
            cavity = np.maximum(1.0 - ratio, 0)
            cavity = self._robust_normalize(cavity, sigma_multiplier=2.5)
            cavity_maps.append(cavity)

        weights = [0.35, 0.30, 0.20, 0.15]
        cavity_combined = np.zeros_like(gray)
        for cav, wt in zip(cavity_maps, weights):
            cavity_combined += cav * wt

        cavity_combined = self._robust_normalize(cavity_combined, sigma_multiplier=2.0)
        ao = 1.0 - cavity_combined * self.cfg.ao_strength
        ao = np.clip(ao, 0, 1)
        ao = gaussian_filter(ao, sigma=max(0.5, 1.5 * scale)).astype(np.float32)
        return ao
