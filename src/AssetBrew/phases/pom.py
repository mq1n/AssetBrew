"""Refine height maps for parallax occlusion mapping workflows.

This phase denoises height maps, preserves tiling continuity, and exports
reference POM shader snippets.
"""

import logging
import os

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from ..config import PipelineConfig
from ..core import (
    AssetRecord, load_image, save_image,
    get_intermediate_path, pad_for_tiling, crop_from_padded, luminance_bt709
)

logger = logging.getLogger("asset_pipeline.pom")


class POMProcessor:
    """Refine height maps for POM and generate shader snippets."""

    def __init__(self, config: PipelineConfig):
        """Initialize POM processor with runtime settings."""
        self.config = config
        self.cfg = config.pom

    def process(self, record: AssetRecord, height_path: str) -> dict:
        """Refine one height map and return updated POM parameters."""
        result = {
            "height_refined": None, "pom_params": {},
            "skipped": False, "error": None
        }

        if not height_path or not os.path.exists(height_path):
            result["skipped"] = True
            if height_path:
                logger.warning("Skipping POM for %s: height source not found (%s)",
                               record.filename, height_path)
            else:
                logger.debug("Skipping POM for %s: no height source provided.", record.filename)
            return result

        try:
            height = load_image(height_path, max_pixels=self.config.max_image_pixels)
            if height.ndim == 3:
                height = luminance_bt709(height, assume_srgb=False)

            h_min, h_max = np.min(height), np.max(height)
            if h_max - h_min < 1e-8:
                # Flat surface -- center at 0.5
                height = np.full_like(height, 0.5)
            elif h_min < 0 or h_max > 1:
                # Out-of-range values -- clamp to [0,1] without rescaling
                height = np.clip(height, 0, 1).astype(np.float32)

            # Pad for tiling if the texture is tileable
            pad_info = None
            if record.is_tileable:
                height, pad_info = pad_for_tiling(
                    height, pad_fraction=0.25
                )

            # Enhance depth detail BEFORE bilateral filtering so that the
            # detail extraction sees the original high-frequency content.
            # Bilateral after enhancement denoises while preserving the
            # amplified edges (avoids the double-smoothing problem).
            height = self._enhance_depth(height)
            filtered = cv2.bilateralFilter(
                height.astype(np.float32),
                d=self.cfg.bilateral_filter_d,
                sigmaColor=self.cfg.bilateral_sigma_color,
                sigmaSpace=self.cfg.bilateral_sigma_space
            )
            height = np.clip(filtered, 0, 1).astype(np.float32)

            # Crop back to original dimensions (scale=1, no upscaling)
            if pad_info is not None:
                height = crop_from_padded(height, pad_info, scale=1)

            out_path = get_intermediate_path(
                record.filepath, "04_pom",
                self.config.intermediate_dir, suffix="_height_pom", ext=".png"
            )
            save_image(height, out_path, bits=16)
            result["height_refined"] = out_path

            height_scale = self.cfg.height_scale_defaults.get(
                record.material_category,
                self.cfg.height_scale_defaults["default"]
            )
            result["pom_params"] = {
                "height_scale": height_scale,
                "min_samples": self.cfg.min_samples,
                "max_samples": self.cfg.max_samples,
                "lod_fade_start": self.cfg.lod_fade_start,
                "material_category": record.material_category
            }
            logger.info(f"POM height refined for {record.filename} (scale={height_scale})")

        except Exception as e:
            logger.error(f"POM processing failed for {record.filename}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _enhance_depth(self, height: np.ndarray) -> np.ndarray:
        h, w = height.shape[:2]
        sigma = max(1.0, 5.0 * (max(h, w) / 1024.0))
        blurred = gaussian_filter(height, sigma=sigma)
        detail = height - blurred
        enhanced = blurred + detail * self.cfg.depth_detail_multiplier
        return np.clip(enhanced, 0, 1).astype(np.float32)

    @staticmethod
    def generate_pom_shader_glsl() -> str:
        """Return reference GLSL code for parallax occlusion mapping."""
        return '''// POM - GLSL Reference Shader
// Integrate into your fragment shader.

uniform sampler2D u_heightMap;
uniform float u_heightScale;       // e.g. 0.05
uniform int u_minSamples;          // e.g. 8
uniform int u_maxSamples;          // e.g. 32

vec2 parallaxOcclusionMapping(vec2 texCoords, vec3 viewDirTangent) {
    const int MAX_POM_STEPS = 128;
    float numSamples = mix(
        float(u_maxSamples), float(u_minSamples),
        clamp(abs(dot(vec3(0.0, 0.0, 1.0), viewDirTangent)), 0.0, 1.0)
    );
    float layerDepth = 1.0 / numSamples;
    float currentLayerDepth = 0.0;
    vec2 P = viewDirTangent.xy / viewDirTangent.z * u_heightScale;
    vec2 deltaUV = P / numSamples;
    vec2 currentUV = texCoords;
    float currentHeight = 1.0 - texture(u_heightMap, currentUV).r;
    int stepCount = 0;

    while (currentLayerDepth < currentHeight) {
        currentUV -= deltaUV;
        currentHeight = 1.0 - texture(u_heightMap, currentUV).r;
        currentLayerDepth += layerDepth;
        stepCount += 1;
        if (stepCount >= MAX_POM_STEPS) {
            break;
        }
    }

    vec2 prevUV = currentUV + deltaUV;
    float afterDepth = currentHeight - currentLayerDepth;
    float beforeDepth = (1.0 - texture(u_heightMap, prevUV).r)
                        - currentLayerDepth + layerDepth;
    float denom = afterDepth - beforeDepth;
    float weight = (abs(denom) > 1e-6) ? afterDepth / denom : 0.5;
    return mix(currentUV, prevUV, weight);
}

float pomSoftShadow(vec2 texCoords, vec3 lightDirTangent) {
    const int MAX_POM_STEPS = 128;
    float initialHeight = 1.0 - texture(u_heightMap, texCoords).r;
    float numSamples = mix(
        float(u_maxSamples), float(u_minSamples),
        clamp(abs(dot(vec3(0.0, 0.0, 1.0), lightDirTangent)), 0.0, 1.0)
    );
    float layerDepth = initialHeight / numSamples;
    vec2 deltaUV = lightDirTangent.xy / lightDirTangent.z * u_heightScale / numSamples;
    vec2 currentUV = texCoords + deltaUV;
    float currentLayerDepth = initialHeight - layerDepth;
    float currentHeight = 1.0 - texture(u_heightMap, currentUV).r;
    float shadow = 0.0;
    int stepCount = 0;

    while (currentLayerDepth > 0.0) {
        if (currentHeight > currentLayerDepth) {
            float newShadow = (currentHeight - currentLayerDepth)
                              * (1.0 - float(stepCount) / numSamples);
            shadow = max(shadow, newShadow);
        }
        currentUV += deltaUV;
        currentLayerDepth -= layerDepth;
        currentHeight = 1.0 - texture(u_heightMap, currentUV).r;
        stepCount += 1;
        if (stepCount >= MAX_POM_STEPS) {
            break;
        }
    }
    return 1.0 - clamp(shadow * 3.0, 0.0, 1.0);
}
'''

    @staticmethod
    def generate_pom_shader_hlsl() -> str:
        """Return reference HLSL code for parallax occlusion mapping."""
        return '''// POM - HLSL Reference Shader

Texture2D HeightMap : register(t4);
SamplerState HeightSampler : register(s4);

cbuffer POMParams : register(b3) {
    float HeightScale;
    int MinSamples;
    int MaxSamples;
    float LODFadeStart;
};

float2 ParallaxOcclusionMapping(float2 texCoords, float3 viewDirTangent) {
    const int MAX_POM_STEPS = 128;
    float cosAngle = abs(dot(float3(0, 0, 1), viewDirTangent));
    float numSamples = lerp((float)MaxSamples, (float)MinSamples, saturate(cosAngle));
    float layerDepth = 1.0 / numSamples;
    float currentLayerDepth = 0.0;
    float2 P = viewDirTangent.xy / viewDirTangent.z * HeightScale;
    float2 deltaUV = P / numSamples;
    float2 currentUV = texCoords;
    float currentHeight = 1.0 - HeightMap.Sample(HeightSampler, currentUV).r;

    [loop]
    for (int i = 0; i < MAX_POM_STEPS && currentLayerDepth < currentHeight; ++i) {
        currentUV -= deltaUV;
        currentHeight = 1.0 - HeightMap.Sample(HeightSampler, currentUV).r;
        currentLayerDepth += layerDepth;
    }

    float2 prevUV = currentUV + deltaUV;
    float afterDepth = currentHeight - currentLayerDepth;
    float beforeDepth = (1.0 - HeightMap.Sample(HeightSampler, prevUV).r)
                        - currentLayerDepth + layerDepth;
    float denom = afterDepth - beforeDepth;
    float weight = (abs(denom) > 1e-6) ? afterDepth / denom : 0.5;
    return lerp(currentUV, prevUV, weight);
}
'''

    def export_shaders(self, output_dir: str):
        """Write GLSL and HLSL reference shaders to an output directory."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "pom_reference.glsl"), "w") as f:
            f.write(self.generate_pom_shader_glsl())
        with open(os.path.join(output_dir, "pom_reference.hlsl"), "w") as f:
            f.write(self.generate_pom_shader_hlsl())
        logger.info(f"POM shaders exported to {output_dir}")
