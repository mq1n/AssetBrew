"""Define typed configuration models for all pipeline phases.

Use `PipelineConfig` to load, validate, and persist runtime settings.
"""

import os
import logging
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger("asset_pipeline.config")


class TextureType(Enum):
    """Enumerate supported texture semantic types."""

    DIFFUSE = "diffuse"
    ALBEDO = "albedo"
    NORMAL = "normal"
    HEIGHT = "height"
    ROUGHNESS = "roughness"
    METALNESS = "metalness"
    AO = "ao"
    SPECULAR = "specular"
    EMISSIVE = "emissive"
    OPACITY = "opacity"
    MASK = "mask"
    UI = "ui"
    ORM = "orm"
    UNKNOWN = "unknown"


class CompressionFormat(Enum):
    """Enumerate supported texture compression targets."""

    BC1 = "bc1"
    BC3 = "bc3"
    BC4 = "bc4"
    BC5 = "bc5"
    BC6H = "bc6h"
    BC7 = "bc7"
    ASTC = "astc"
    ETC2 = "etc2"
    NONE = "none"


TEXTURE_PATTERNS: Dict[TextureType, List[str]] = {
    TextureType.DIFFUSE:   [
        "_diff", "_diffuse", "_color", "_col", "_c", "_d", "_basecolor", "_base",
        "_bc",
    ],
    TextureType.ALBEDO:    ["_albedo", "_alb"],
    TextureType.NORMAL:    ["_norm", "_normal", "_nrm", "_n"],
    TextureType.HEIGHT:    ["_height", "_h", "_disp", "_displacement", "_parallax", "_bump"],
    TextureType.ROUGHNESS: ["_rough", "_roughness", "_r", "_gloss", "_glossiness"],
    TextureType.METALNESS: ["_metal", "_metalness", "_metallic", "_met"],
    TextureType.AO:        ["_ao", "_ambient", "_occlusion", "_ambientocclusion"],
    TextureType.SPECULAR:  ["_spec", "_specular"],
    TextureType.EMISSIVE:  ["_emissive", "_emit", "_glow"],
    TextureType.OPACITY:   ["_opacity", "_alpha", "_transparency"],
    TextureType.MASK:      ["_mask", "_msk"],
    TextureType.UI:        ["_ui", "_hud", "_gui", "_icon"],
    TextureType.ORM:       ["_orm", "_rma", "_arm"],
}

# Patterns that represent glossiness (inverse of roughness)
GLOSS_PATTERNS: List[str] = ["_gloss", "_glossiness"]


@dataclass
class UpscaleConfig:
    """Store settings for the AI upscaling phase."""

    enabled: bool = True
    model_name: str = "RealESRGAN_x4plus"
    scale_factor: int = 4
    target_resolution: int = 2048
    hero_resolution: int = 4096
    enforce_power_of_two: bool = True
    hero_asset_patterns: List[str] = field(default_factory=lambda: [
        "character", "face", "hero", "main", "player", "weapon"
    ])
    tile_size: int = 512
    tile_pad: int = 32
    half_precision: bool = True
    preserve_tiling: bool = True
    tiling_asset_patterns: List[str] = field(default_factory=lambda: [
        "floor", "wall", "ground", "brick", "tile", "wood", "stone",
        "concrete", "metal", "fabric", "grass", "dirt", "sand", "rock"
    ])
    skip_types: List[str] = field(default_factory=lambda: ["ui", "mask"])
    nearest_neighbor_types: List[str] = field(default_factory=lambda: ["mask", "opacity"])
    model_dir: str = "./models"
    model_sha256: Dict[str, str] = field(default_factory=dict)
    allow_unverified_model_download: bool = False
    require_ai: bool = False


@dataclass
class PBRConfig:
    """Store settings for roughness/metalness/AO generation."""

    enabled: bool = True
    generate_roughness: bool = True
    generate_metalness: bool = True
    generate_ao: bool = True
    delight_diffuse: bool = True
    delight_method: str = "multifrequency"  # "gaussian" | "multifrequency"
    delight_strength: float = 1.0
    delight_low_frequency_sigma: float = 48.0
    delight_mid_frequency_sigma: float = 12.0
    delight_shadow_lift: float = 0.12
    delight_highlight_suppress: float = 0.25
    roughness_base_value: float = 0.5
    roughness_variance_weight: float = 0.5
    roughness_sobel_weight: float = 0.3
    roughness_laplacian_weight: float = 0.2
    generate_gloss: bool = False
    material_zone_masks: bool = True
    apply_zone_pbr_adjustments: bool = True
    zone_blur_radius: int = 5
    zone_roughness_bias: Dict[str, float] = field(default_factory=lambda: {
        "metal": -0.22, "cloth": 0.18, "leather": 0.08, "skin": 0.02,
    })
    zone_metalness_floor: Dict[str, float] = field(default_factory=lambda: {
        "metal": 0.75, "cloth": 0.0, "leather": 0.0, "skin": 0.0,
    })
    material_roughness_defaults: Dict[str, float] = field(default_factory=lambda: {
        "metal": 0.25, "wood": 0.65, "stone": 0.7, "brick": 0.75,
        "fabric": 0.85, "plastic": 0.4, "glass": 0.1, "skin": 0.55,
        "leather": 0.6, "concrete": 0.8, "rubber": 0.9, "paint": 0.35,
        "ceramic": 0.3, "default": 0.5
    })
    material_metalness_defaults: Dict[str, float] = field(default_factory=lambda: {
        "metal": 0.95, "gold": 1.0, "silver": 1.0, "iron": 0.9,
        "copper": 0.95, "rubber": 0.0, "paint": 0.0, "ceramic": 0.0,
        "default": 0.0
    })
    ao_strength: float = 1.0
    ao_radius: int = 8
    metalness_binarize: bool = True
    metalness_threshold: float = 0.5


@dataclass
class NormalConfig:
    """Store settings for normal and height map generation."""

    enabled: bool = True
    method: str = "hybrid"
    strength: float = 1.0
    nz_scale: float = 2.0  # Z-component scale for Sobel normals (higher = flatter normals)
    sobel_weight: float = 0.6
    sobel_blur_radius: int = 1
    invert_y: bool = False
    generate_height: bool = True
    height_high_pass_radius: int = 5
    height_contrast: float = 1.5
    height_normalize: bool = True
    validate_normals: bool = True
    deepbump_overlap: str = "LARGE"        # "SMALL" | "MEDIUM" | "LARGE"
    deepbump_seamless_height: bool = True   # Seamless Frankot-Chellappa height


@dataclass
class POMConfig:
    """Store settings for parallax occlusion map refinement."""

    enabled: bool = True
    height_scale_defaults: Dict[str, float] = field(default_factory=lambda: {
        "brick": 0.05, "cobblestone": 0.08, "stone": 0.06,
        "wood": 0.03, "fabric": 0.02, "metal": 0.01,
        "tile": 0.04, "concrete": 0.03, "default": 0.04
    })
    min_samples: int = 8
    max_samples: int = 32
    lod_fade_start: int = 2
    bilateral_filter_d: int = 5
    # Color sigma in [0,1] float32 range (NOT uint8 0-255)
    bilateral_sigma_color: float = 0.3
    bilateral_sigma_space: float = 10.0
    depth_detail_multiplier: float = 1.3  # High-frequency detail boost for depth enhancement


@dataclass
class MipmapConfig:
    """Store settings for mipmap generation and filtering."""

    enabled: bool = True
    filter_method: str = "lanczos"
    srgb_downsampling: bool = True
    renormalize_normals: bool = True
    increase_roughness_per_mip: bool = True
    roughness_mip_increase: float = 0.04
    sharpen_mips: bool = True
    sharpen_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    sharpen_strength: float = 0.3
    sharpen_radius: int = 1
    min_resident_mips: int = 3


@dataclass
class CompressionConfig:
    """Store settings for DDS/KTX2/TGA output compression."""

    enabled: bool = True
    tool: str = "compressonator"
    tool_path: str = ""
    tool_timeout_seconds: int = 60
    compressonator_encode_with: str = "hpc"
    compressonator_num_threads: int = 0  # 0 = auto
    compressonator_performance: Optional[float] = None
    compressonator_no_progress: bool = True
    format_map: Dict[str, str] = field(default_factory=lambda: {
        "diffuse": "bc7", "albedo": "bc7", "normal": "bc5",
        "height": "bc4", "roughness": "bc4", "metalness": "bc4",
        "ao": "bc4", "specular": "bc7", "emissive": "bc7",
        "opacity": "bc4", "mask": "bc4", "ui": "bc7",
        "orm": "bc7", "unknown": "bc7",
        # Optional alpha-specific overrides (used for base RGBA textures)
        "diffuse_alpha": "bc3", "albedo_alpha": "bc3",
        "specular_alpha": "bc3", "emissive_alpha": "bc3",
        "ui_alpha": "bc3", "unknown_alpha": "bc3",
    })
    quality: float = 0.85
    generate_dds: bool = True
    generate_ktx2: bool = False  # Requires toktx or basisu on PATH
    generate_tga: bool = False
    fail_on_fallback: bool = False
    srgb_texture_types: List[str] = field(default_factory=lambda: [
        "diffuse", "albedo", "specular", "emissive", "ui", "unknown",
    ])


@dataclass
class ValidationConfig:
    """Store settings for quality checks and validation outputs."""

    enabled: bool = True
    check_tiling: bool = True
    check_normals: bool = True
    render_test_sphere: bool = True
    regression_diff: bool = False  # Compare against previous output
    max_allowed_diff: float = 0.15
    output_comparison: bool = True
    sphere_resolution: int = 512
    sphere_light_angles: int = 4
    enforce_material_semantics: bool = True
    strict_material_semantics: bool = True
    warn_on_heuristic_maps: bool = True
    fail_on_heuristic_maps: bool = False
    require_full_mipchain: bool = True
    metalness_nonmetal_max_mean: float = 0.25
    metalness_metal_min_mean: float = 0.55
    metalness_midband_max_ratio: float = 0.25
    roughness_min_stddev: float = 0.01
    ao_min_stddev: float = 0.005
    enforce_plausible_albedo: bool = True
    albedo_linear_min: float = 0.02
    albedo_linear_max: float = 0.90
    albedo_black_ratio_max: float = 0.02
    albedo_white_ratio_max: float = 0.02
    normal_unit_tolerance: float = 0.02


@dataclass
class ORMPackingConfig:
    """Flexible engine-specific channel packing for material maps."""

    enabled: bool = True
    preset: str = "unreal_orm"  # unreal_orm | unity_mas | source_phong | idtech_rma | custom
    r_channel: str = "ao"
    g_channel: str = "roughness"
    b_channel: str = "metalness"
    alpha_channel: str = "none"
    custom_layout: Dict[str, str] = field(default_factory=dict)
    output_suffix: str = "_orm"
    generate_gloss_in_diffuse_alpha: bool = False
    gloss_source: str = "roughness"  # roughness | gloss
    overwrite_existing_alpha: bool = True


@dataclass
class ColorConsistencyConfig:
    """Color consistency pass across assets of the same material group."""

    enabled: bool = True
    reference_image: str = ""  # NOT IMPLEMENTED — accepted but has no effect
    correction_strength: float = 0.35
    group_by_material: bool = True


@dataclass
class ColorGradingConfig:
    """Color/gamma pipeline controls for albedo-like textures."""

    enabled: bool = True
    process_in_linear: bool = True
    apply_to_texture_types: List[str] = field(default_factory=lambda: [
        "diffuse", "albedo", "specular", "emissive", "ui", "unknown",
    ])
    white_balance_shift: float = 0.0  # [-1, 1], negative=cool, positive=warm
    white_balance_per_material: Dict[str, float] = field(default_factory=dict)
    exposure_ev: float = 0.0
    exposure_per_material: Dict[str, float] = field(default_factory=dict)
    midtone_gamma: float = 1.0
    midtone_gamma_per_material: Dict[str, float] = field(default_factory=dict)
    saturation: float = 1.0
    saturation_per_material: Dict[str, float] = field(default_factory=dict)
    lut_path: str = ""  # .cube path
    lut_strength: float = 0.0
    denoise_strength: float = 0.0
    sharpen_strength: float = 0.0
    sharpen_radius: int = 1
    auto_plausible_albedo_clamp: bool = True
    plausible_albedo_min: float = 0.02
    plausible_albedo_max: float = 0.90


@dataclass
class EmissiveConfig:
    """Emissive map detection/generation from albedo-like inputs."""

    enabled: bool = True
    luminance_threshold: float = 0.80
    saturation_threshold: float = 0.50
    value_threshold: float = 0.90
    min_region_ratio: float = 0.0005
    boost: float = 1.0


@dataclass
class ReflectionMaskConfig:
    """Generate environment/reflection masks from roughness/metalness cues."""

    enabled: bool = True
    metalness_weight: float = 0.65
    gloss_weight: float = 0.35
    bias: float = 0.0


@dataclass
class SeamRepairConfig:
    """Post-upscale seam repair configuration."""

    enabled: bool = True
    only_tileable: bool = True
    repair_border_width: int = 12
    detect_threshold: float = 0.08
    blend_strength: float = 0.85
    inpaint_radius: int = 2


@dataclass
class TilingQualityConfig:
    """Per-asset tiling quality scoring and auto-flag thresholds."""

    enabled: bool = True
    border_width: int = 8
    warn_score: float = 0.75
    fail_score: float = 0.55
    auto_flag_top_n: int = 20


@dataclass
class SpecularAAConfig:
    """Geometric specular anti-aliasing via normal variance -> roughness boost."""

    enabled: bool = True
    variance_threshold: float = 0.001
    max_roughness_increase: float = 0.15
    kernel_size: int = 3


@dataclass
class DetailMapConfig:
    """Detail normal map overlay for close-up micro-detail."""

    enabled: bool = False
    detail_normal_path: str = ""
    detail_uv_scale: float = 8.0
    detail_strength: float = 0.3
    detail_fade_distance: float = 5.0  # NOT IMPLEMENTED — accepted but has no effect
    apply_to_categories: List[str] = field(default_factory=lambda: [
        "stone", "concrete", "brick", "wood", "metal", "fabric"
    ])


@dataclass
class GPUConfig:
    """GPU memory monitoring and adaptive tile sizing."""

    enabled: bool = True
    max_vram_mb: int = 0
    min_tile_size: int = 128
    log_vram: bool = True
    fallback_to_cpu: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint / resume for crash recovery and incremental processing."""

    enabled: bool = True
    checkpoint_path: str = "./assets/checkpoint.json"
    save_interval: int = 10
    skip_completed: bool = True


_SUPPORTED_CONFIG_VERSION = 1


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""

    config_version: int = 1
    input_dir: str = "./assets/source"
    output_dir: str = "./assets/output"
    intermediate_dir: str = "./assets/intermediate"
    comparison_dir: str = "./assets/comparisons"
    manifest_path: str = "./assets/manifest.csv"

    supported_formats: List[str] = field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".tif", ".dds"
    ])
    max_workers: int = 4
    phase_timeout_seconds: int = 600
    phase_failure_abort_ratio: float = 0.50
    phase_failure_abort_min_processed: int = 10
    device: str = "auto"
    log_level: str = "INFO"
    dry_run: bool = False
    cleanup_intermediates: bool = False
    max_image_pixels: int = 67108864  # 8192x8192
    min_texture_dim: int = 4

    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    pbr: PBRConfig = field(default_factory=PBRConfig)
    normal: NormalConfig = field(default_factory=NormalConfig)
    pom: POMConfig = field(default_factory=POMConfig)
    mipmap: MipmapConfig = field(default_factory=MipmapConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    orm_packing: ORMPackingConfig = field(default_factory=ORMPackingConfig)
    color_consistency: ColorConsistencyConfig = field(default_factory=ColorConsistencyConfig)
    color_grading: ColorGradingConfig = field(default_factory=ColorGradingConfig)
    emissive: EmissiveConfig = field(default_factory=EmissiveConfig)
    reflection_mask: ReflectionMaskConfig = field(default_factory=ReflectionMaskConfig)
    seam_repair: SeamRepairConfig = field(default_factory=SeamRepairConfig)
    tiling_quality: TilingQualityConfig = field(default_factory=TilingQualityConfig)
    specular_aa: SpecularAAConfig = field(default_factory=SpecularAAConfig)
    detail_map: DetailMapConfig = field(default_factory=DetailMapConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load pipeline configuration from YAML or return defaults."""
        if not os.path.exists(path):
            logger.info("Config file '%s' not found. Using defaults.", path)
            config = cls()
            config.validate()
            return config
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Failed to parse YAML config '{path}': {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Config file '{path}' must contain a YAML mapping, "
                f"got {type(data).__name__}"
            )
        # Warn if the YAML specifies a higher config version than we support.
        yaml_version = data.get("config_version", 1)
        if isinstance(yaml_version, int) and yaml_version > _SUPPORTED_CONFIG_VERSION:
            logger.warning(
                "Config file '%s' has config_version=%d, but this build only "
                "supports up to version %d. Some settings may be ignored.",
                path, yaml_version, _SUPPORTED_CONFIG_VERSION,
            )
        config = cls()
        _merge_dict_to_dataclass(config, data)
        try:
            config.validate()
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        return config

    def to_yaml(self, path: str):
        """Write pipeline configuration to a YAML file."""
        import dataclasses
        import threading as _th
        data = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1]
        tmp_path = f"{path}.tmp.{os.getpid()}.{_th.get_ident()}{ext}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # Class-level cache so torch is only imported once across all instances.
    _resolved_auto_device: str | None = None

    def resolve_device(self) -> str:
        """Resolve runtime device from config (`auto`, `cuda`, or `cpu`)."""
        if self.device != "auto":
            logger.debug("Using configured device setting: %s", self.device)
            return self.device

        # Return cached result to avoid re-importing torch (which can crash
        # on some Python/platform combinations due to DLL incompatibilities).
        if PipelineConfig._resolved_auto_device is not None:
            return PipelineConfig._resolved_auto_device

        result = self._probe_cuda()
        PipelineConfig._resolved_auto_device = result
        return result

    @staticmethod
    def _probe_cuda() -> str:
        """Detect CUDA availability via torch, with subprocess fallback.

        On some Python/platform combinations the torch DLL loader can crash
        the interpreter process with an access violation. To keep auto-device
        detection crash-safe, probing prefers a subprocess import unless torch
        is already loaded in-process.
        """
        import json
        import subprocess
        import sys
        # Fast path: torch already loaded in this process (no DLL risk).
        if "torch" in sys.modules:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        probe_script = (
            "import json\n"
            "try:\n"
            "    import torch\n"
            "    print(json.dumps({'status': 'ok', 'cuda': bool(torch.cuda.is_available())}))\n"
            "except ImportError:\n"
            "    print(json.dumps({'status': 'missing'}))\n"
            "except Exception as exc:\n"
            "    err = f'{exc.__class__.__name__}: {exc}'\n"
            "    print(json.dumps({'status': 'error', 'error': err}))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", probe_script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "Torch CUDA probe subprocess failed (%s: %s). Falling back to CPU.",
                exc.__class__.__name__,
                exc,
            )
            return "cpu"
        if proc.returncode != 0:
            logger.warning(
                "Torch CUDA probe subprocess exited with code %s. Falling back to CPU.",
                proc.returncode,
            )
            return "cpu"
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            logger.warning("Torch CUDA probe subprocess produced no output. Falling back to CPU.")
            return "cpu"
        try:
            payload = json.loads(lines[-1])
        except Exception as exc:
            logger.warning(
                "Torch CUDA probe returned invalid payload (%s: %s). Falling back to CPU.",
                exc.__class__.__name__,
                exc,
            )
            return "cpu"
        status = payload.get("status")
        if status == "ok":
            return "cuda" if payload.get("cuda") else "cpu"
        if status == "missing":
            logger.debug("Torch not installed; using CPU device fallback.")
            return "cpu"
        error_text = payload.get("error")
        if error_text:
            logger.warning(
                "Torch CUDA probe failed (%s). Falling back to CPU.",
                error_text,
            )
        return "cpu"

    def validate(self):
        """Validate configuration values. Raises ValueError on invalid config."""
        errors = []

        valid_devices = {"auto", "cuda", "cpu"}
        device_str = self.device.strip() if isinstance(self.device, str) else self.device
        device_base = device_str.split(":", 1)[0] if ":" in device_str else device_str
        if device_base not in valid_devices:
            errors.append(
                f"device must be one of {sorted(valid_devices)} (or cuda:N), "
                f"got '{self.device}'"
            )
        elif device_base == "cuda" and ":" in device_str:
            idx_part = device_str.split(":", 1)[1]
            if not idx_part.isdigit():
                errors.append(
                    "device cuda index must be a non-negative integer "
                    f"(e.g. 'cuda:0'), got '{self.device}'"
                )

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            errors.append(
                f"log_level must be one of {sorted(valid_log_levels)}, "
                f"got '{self.log_level}'"
            )

        if self.max_workers < 1:
            errors.append("max_workers must be >= 1")
        if self.max_workers > 128:
            errors.append("max_workers must be <= 128")
        if self.phase_timeout_seconds < 1:
            errors.append("phase_timeout_seconds must be >= 1")
        if not (0.0 < self.phase_failure_abort_ratio <= 1.0):
            errors.append("phase_failure_abort_ratio must be in (0, 1]")
        if self.phase_failure_abort_min_processed < 1:
            errors.append("phase_failure_abort_min_processed must be >= 1")
        if self.max_image_pixels < 0:
            errors.append("max_image_pixels must be >= 0 (0 = unlimited)")
        if self.min_texture_dim < 1:
            errors.append("min_texture_dim must be >= 1")
        if not self.supported_formats:
            errors.append(
                "supported_formats must not be empty — no files would be processed"
            )

        resolved_device = self.resolve_device()

        # Upscale
        if self.upscale.scale_factor < 1:
            errors.append("upscale.scale_factor must be >= 1")
        if self.upscale.scale_factor != 4:
            import warnings
            warnings.warn(
                "upscale.scale_factor is deprecated and has no effect. "
                "The upscale phase always uses the model's native scale "
                "factor (4x for RealESRGAN). Use upscale.target_resolution "
                "to control final output size.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.upscale.tile_size < 64:
            errors.append("upscale.tile_size must be >= 64")
        if self.upscale.target_resolution < 1:
            errors.append("upscale.target_resolution must be >= 1")
        if self.upscale.hero_resolution < self.upscale.target_resolution:
            errors.append("upscale.hero_resolution must be >= target_resolution")
        if self.upscale.tile_pad < 0:
            errors.append("upscale.tile_pad must be >= 0")
        if self.upscale.tile_pad >= self.upscale.tile_size // 2:
            errors.append(
                f"upscale.tile_pad ({self.upscale.tile_pad}) must be < "
                f"tile_size / 2 ({self.upscale.tile_size // 2}) to avoid "
                f"Real-ESRGAN assertion failures"
            )
        if (self.upscale.enabled and self.upscale.half_precision
                and resolved_device == "cpu"):
            logger.warning(
                "upscale.half_precision is enabled but resolved device is CPU. "
                "Runtime fixups should disable half precision before execution."
            )

        # PBR
        w_sum = (self.pbr.roughness_variance_weight +
                 self.pbr.roughness_sobel_weight +
                 self.pbr.roughness_laplacian_weight)
        if w_sum <= 0:
            errors.append("PBR roughness weights must sum to > 0")
        for _wname in ("roughness_variance_weight", "roughness_sobel_weight",
                       "roughness_laplacian_weight"):
            if getattr(self.pbr, _wname) < 0:
                errors.append(f"pbr.{_wname} must be >= 0")
        if not (0.0 <= self.pbr.roughness_base_value <= 1.0):
            errors.append("pbr.roughness_base_value must be in [0, 1]")
        if not (0.0 <= self.pbr.metalness_threshold <= 1.0):
            errors.append("pbr.metalness_threshold must be in [0, 1]")
        if not (0 <= self.pbr.ao_strength <= 2.0):
            errors.append("pbr.ao_strength should be in [0, 2.0]")
        if self.pbr.ao_radius < 1:
            errors.append("pbr.ao_radius must be >= 1")
        valid_delight_methods = {"gaussian", "multifrequency"}
        if self.pbr.delight_method not in valid_delight_methods:
            errors.append(
                f"pbr.delight_method must be one of {sorted(valid_delight_methods)}, "
                f"got '{self.pbr.delight_method}'"
            )
        if not (0.0 <= self.pbr.delight_strength <= 1.0):
            errors.append("pbr.delight_strength must be in [0, 1]")
        if self.pbr.delight_low_frequency_sigma <= 0:
            errors.append("pbr.delight_low_frequency_sigma must be > 0")
        if self.pbr.delight_mid_frequency_sigma <= 0:
            errors.append("pbr.delight_mid_frequency_sigma must be > 0")
        if (self.pbr.delight_low_frequency_sigma > 0
                and self.pbr.delight_mid_frequency_sigma > 0
                and self.pbr.delight_low_frequency_sigma
                <= self.pbr.delight_mid_frequency_sigma):
            logger.warning(
                "pbr.delight_low_frequency_sigma (%.1f) <= "
                "delight_mid_frequency_sigma (%.1f). "
                "Typically the low-frequency sigma should be larger.",
                self.pbr.delight_low_frequency_sigma,
                self.pbr.delight_mid_frequency_sigma,
            )
        if not (0.0 <= self.pbr.delight_shadow_lift <= 1.0):
            errors.append("pbr.delight_shadow_lift must be in [0, 1]")
        if not (0.0 <= self.pbr.delight_highlight_suppress <= 1.0):
            errors.append("pbr.delight_highlight_suppress must be in [0, 1]")
        if self.pbr.zone_blur_radius < 0:
            errors.append("pbr.zone_blur_radius must be >= 0")

        # Normal
        valid_methods = {"sobel", "hybrid", "deepbump"}
        if self.normal.method not in valid_methods:
            errors.append(
                f"normal.method must be one of {sorted(valid_methods)}, "
                f"got '{self.normal.method}'"
            )
        if self.normal.strength <= 0:
            errors.append("normal.strength must be > 0")
        if self.normal.strength > 100.0:
            errors.append("normal.strength must be <= 100.0")
        if self.normal.nz_scale <= 0:
            errors.append("normal.nz_scale must be > 0")
        if not (0 <= self.normal.sobel_weight <= 1):
            errors.append("normal.sobel_weight must be in [0, 1]")
        if self.normal.height_contrast <= 0:
            errors.append("normal.height_contrast must be > 0")
        valid_overlaps = {"SMALL", "MEDIUM", "LARGE"}
        if self.normal.deepbump_overlap not in valid_overlaps:
            errors.append(
                f"normal.deepbump_overlap must be one of {sorted(valid_overlaps)}, "
                f"got '{self.normal.deepbump_overlap}'"
            )

        # POM
        if self.pom.min_samples < 1:
            errors.append("pom.min_samples must be >= 1")
        if self.pom.max_samples < self.pom.min_samples:
            errors.append("pom.max_samples must be >= min_samples")
        if not (0.0 <= self.pom.bilateral_sigma_color <= 1.0):
            errors.append("pom.bilateral_sigma_color must be in [0, 1] (float32 range)")
        if not (0.0 < self.pom.bilateral_sigma_space <= 100.0):
            errors.append("pom.bilateral_sigma_space must be in (0, 100]")
        if self.pom.bilateral_filter_d < 0:
            errors.append("pom.bilateral_filter_d must be >= 0")
        if self.pom.depth_detail_multiplier < 0:
            errors.append("pom.depth_detail_multiplier must be >= 0")

        # Mipmap
        if self.mipmap.roughness_mip_increase < 0:
            errors.append("mipmap.roughness_mip_increase must be >= 0")
        if self.mipmap.sharpen_strength < 0:
            errors.append("mipmap.sharpen_strength must be >= 0")
        if self.mipmap.sharpen_radius < 0:
            errors.append("mipmap.sharpen_radius must be >= 0")
        if self.mipmap.min_resident_mips < 1:
            errors.append("mipmap.min_resident_mips must be >= 1")

        # Compression
        if not (0 < self.compression.quality <= 1.0):
            errors.append("compression.quality must be in (0, 1.0]")
        if self.compression.tool_timeout_seconds < 1:
            errors.append("compression.tool_timeout_seconds must be >= 1")

        valid_tools = {"compressonator", "texconv"}
        if self.compression.tool not in valid_tools:
            errors.append(
                f"compression.tool must be one of {sorted(valid_tools)}, "
                f"got '{self.compression.tool}'"
            )

        valid_encode_with = {"", "cpu", "hpc", "ocl", "dxc", "gpu"}
        encode_with = self.compression.compressonator_encode_with.strip().lower()
        if encode_with not in valid_encode_with:
            errors.append(
                "compression.compressonator_encode_with must be one of "
                f"{sorted(valid_encode_with)}, got "
                f"'{self.compression.compressonator_encode_with}'"
            )

        if not (0 <= self.compression.compressonator_num_threads <= 128):
            errors.append(
                "compression.compressonator_num_threads must be in [0, 128] "
                "(0 = auto)"
            )

        _perf_sentinel = object()
        _pending_perf_normalized = _perf_sentinel
        perf = self.compression.compressonator_performance
        normalized_perf = perf
        if isinstance(perf, str):
            stripped = perf.strip()
            lowered = stripped.lower()
            if stripped == "" or lowered in {"none", "null", "~"}:
                normalized_perf = None
                _pending_perf_normalized = None
            else:
                try:
                    normalized_perf = float(stripped)
                    _pending_perf_normalized = float(normalized_perf)
                except ValueError:
                    errors.append(
                        "compression.compressonator_performance must be null or "
                        "a number in [0, 1]"
                    )

        if normalized_perf is not None:
            if not isinstance(normalized_perf, (int, float)):
                errors.append(
                    "compression.compressonator_performance must be null or "
                    "a number in [0, 1]"
                )
            elif not (0 <= float(normalized_perf) <= 1):
                errors.append(
                    "compression.compressonator_performance must be in [0, 1]"
                )

        valid_compression_formats = {fmt.value for fmt in CompressionFormat}
        _pending_format_normalizations = {}
        for tex_type, fmt in self.compression.format_map.items():
            if not isinstance(fmt, str):
                errors.append(
                    "compression.format_map"
                    f"['{tex_type}'] must be a string, got {type(fmt).__name__}"
                )
                continue
            normalized = fmt.lower()
            if normalized not in valid_compression_formats:
                errors.append(
                    "compression.format_map"
                    f"['{tex_type}'] must be one of "
                    f"{sorted(valid_compression_formats)}, got '{fmt}'"
                )
            else:
                # Defer normalization until validation succeeds to avoid
                # partial mutation on ValidationError.
                _pending_format_normalizations[tex_type] = normalized

        # sRGB texture types
        valid_tex_type_values = {t.value for t in TextureType}
        for stype in self.compression.srgb_texture_types:
            if stype not in valid_tex_type_values:
                errors.append(
                    f"compression.srgb_texture_types: unknown type '{stype}', "
                    f"must be one of {sorted(valid_tex_type_values)}"
                )

        # Validation
        if self.validation.max_allowed_diff <= 0:
            errors.append("validation.max_allowed_diff must be > 0")
        if self.validation.sphere_resolution < 32:
            errors.append("validation.sphere_resolution must be >= 32")
        if self.validation.sphere_light_angles < 1:
            errors.append("validation.sphere_light_angles must be >= 1")
        if not (0 <= self.validation.metalness_nonmetal_max_mean <= 1):
            errors.append("validation.metalness_nonmetal_max_mean must be in [0, 1]")
        if not (0 <= self.validation.metalness_metal_min_mean <= 1):
            errors.append("validation.metalness_metal_min_mean must be in [0, 1]")
        if not (0 <= self.validation.metalness_midband_max_ratio <= 1):
            errors.append("validation.metalness_midband_max_ratio must be in [0, 1]")
        if not (0 <= self.validation.roughness_min_stddev <= 1):
            errors.append("validation.roughness_min_stddev must be in [0, 1]")
        if not (0 <= self.validation.ao_min_stddev <= 1):
            errors.append("validation.ao_min_stddev must be in [0, 1]")
        if not (0 <= self.validation.albedo_linear_min <= 1):
            errors.append("validation.albedo_linear_min must be in [0, 1]")
        if not (0 <= self.validation.albedo_linear_max <= 1):
            errors.append("validation.albedo_linear_max must be in [0, 1]")
        if not (0 <= self.validation.albedo_black_ratio_max <= 1):
            errors.append("validation.albedo_black_ratio_max must be in [0, 1]")
        if not (0 <= self.validation.albedo_white_ratio_max <= 1):
            errors.append("validation.albedo_white_ratio_max must be in [0, 1]")
        if not (0 < self.validation.normal_unit_tolerance <= 1):
            errors.append("validation.normal_unit_tolerance must be in (0, 1]")
        if self.validation.albedo_linear_min >= self.validation.albedo_linear_max:
            errors.append("validation.albedo_linear_min must be < albedo_linear_max")
        if (
            self.validation.metalness_metal_min_mean
            < self.validation.metalness_nonmetal_max_mean
        ):
            errors.append(
                "validation.metalness_metal_min_mean should be >= "
                "validation.metalness_nonmetal_max_mean"
            )

        # Specular AA
        if self.specular_aa.kernel_size < 1 or self.specular_aa.kernel_size % 2 == 0:
            errors.append("specular_aa.kernel_size must be odd and >= 1")
        if self.specular_aa.max_roughness_increase < 0:
            errors.append("specular_aa.max_roughness_increase must be >= 0")

        # Color consistency
        if not (0 <= self.color_consistency.correction_strength <= 1.0):
            errors.append("color_consistency.correction_strength must be in [0, 1.0]")

        # Color grading
        if not (-1.0 <= self.color_grading.white_balance_shift <= 1.0):
            errors.append("color_grading.white_balance_shift must be in [-1, 1]")
        if self.color_grading.midtone_gamma <= 0:
            errors.append("color_grading.midtone_gamma must be > 0")
        if not (0.0 <= self.color_grading.lut_strength <= 1.0):
            errors.append("color_grading.lut_strength must be in [0, 1]")
        if self.color_grading.saturation < 0:
            errors.append("color_grading.saturation must be >= 0")
        if self.color_grading.denoise_strength < 0:
            errors.append("color_grading.denoise_strength must be >= 0")
        if self.color_grading.sharpen_strength < 0:
            errors.append("color_grading.sharpen_strength must be >= 0")
        if self.color_grading.sharpen_radius < 0:
            errors.append("color_grading.sharpen_radius must be >= 0")
        if not (0 <= self.color_grading.plausible_albedo_min <= 1):
            errors.append("color_grading.plausible_albedo_min must be in [0, 1]")
        if not (0 <= self.color_grading.plausible_albedo_max <= 1):
            errors.append("color_grading.plausible_albedo_max must be in [0, 1]")
        if self.color_grading.plausible_albedo_min >= self.color_grading.plausible_albedo_max:
            errors.append(
                "color_grading.plausible_albedo_min must be < plausible_albedo_max"
            )

        # Emissive detection
        if not (0 <= self.emissive.luminance_threshold <= 1):
            errors.append("emissive.luminance_threshold must be in [0, 1]")
        if not (0 <= self.emissive.saturation_threshold <= 1):
            errors.append("emissive.saturation_threshold must be in [0, 1]")
        if not (0 <= self.emissive.value_threshold <= 1):
            errors.append("emissive.value_threshold must be in [0, 1]")
        if not (0 <= self.emissive.min_region_ratio <= 1):
            errors.append("emissive.min_region_ratio must be in [0, 1]")
        if self.emissive.boost < 0:
            errors.append("emissive.boost must be >= 0")

        # Reflection mask
        if self.reflection_mask.metalness_weight < 0:
            errors.append("reflection_mask.metalness_weight must be >= 0")
        if self.reflection_mask.gloss_weight < 0:
            errors.append("reflection_mask.gloss_weight must be >= 0")
        if (
            self.reflection_mask.metalness_weight == 0
            and self.reflection_mask.gloss_weight == 0
        ):
            errors.append("reflection_mask weights cannot both be 0")

        # Seam repair
        if self.seam_repair.repair_border_width < 1:
            errors.append("seam_repair.repair_border_width must be >= 1")
        if self.seam_repair.detect_threshold <= 0:
            errors.append("seam_repair.detect_threshold must be > 0")
        if not (0 <= self.seam_repair.blend_strength <= 1):
            errors.append("seam_repair.blend_strength must be in [0, 1]")
        if self.seam_repair.inpaint_radius < 0:
            errors.append("seam_repair.inpaint_radius must be >= 0")

        # Tiling quality scoring
        if self.tiling_quality.border_width < 1:
            errors.append("tiling_quality.border_width must be >= 1")
        if not (0 <= self.tiling_quality.warn_score <= 1):
            errors.append("tiling_quality.warn_score must be in [0, 1]")
        if not (0 <= self.tiling_quality.fail_score <= 1):
            errors.append("tiling_quality.fail_score must be in [0, 1]")
        if self.tiling_quality.fail_score > self.tiling_quality.warn_score:
            errors.append("tiling_quality.fail_score must be <= warn_score")
        if self.tiling_quality.auto_flag_top_n < 0:
            errors.append("tiling_quality.auto_flag_top_n must be >= 0")

        # GPU
        if self.gpu.max_vram_mb < 0:
            errors.append("gpu.max_vram_mb must be >= 0 (0 = auto-detect)")
        if self.gpu.min_tile_size < 32:
            errors.append("gpu.min_tile_size must be >= 32")

        # Checkpoint
        if self.checkpoint.save_interval < 1:
            errors.append("checkpoint.save_interval must be >= 1")

        # ORM packing
        valid_presets = {
            "unreal_orm", "unity_mas", "source_phong", "idtech_rma", "custom"
        }
        if self.orm_packing.preset not in valid_presets:
            errors.append(
                f"orm_packing.preset must be one of {sorted(valid_presets)}"
            )
        valid_channels = {
            "ao", "roughness", "metalness", "gloss", "smoothness", "one", "zero", "none"
        }
        orm_channels = []
        for ch_name in ["r_channel", "g_channel", "b_channel"]:
            val = getattr(self.orm_packing, ch_name)
            if val not in valid_channels:
                errors.append(f"orm_packing.{ch_name} must be one of {valid_channels}")
            orm_channels.append(val)
        if self.orm_packing.alpha_channel not in valid_channels:
            errors.append(f"orm_packing.alpha_channel must be one of {valid_channels}")
        _constant_channels = {"zero", "one", "none"}
        data_channels = [ch for ch in orm_channels if ch not in _constant_channels]
        if len(set(data_channels)) != len(data_channels):
            errors.append("orm_packing data channels must be unique (no duplicates)")
        if not self.orm_packing.output_suffix or not self.orm_packing.output_suffix.startswith("_"):
            errors.append("orm_packing.output_suffix must be non-empty and start with '_'")
        if self.orm_packing.gloss_source not in {"roughness", "gloss"}:
            errors.append("orm_packing.gloss_source must be 'roughness' or 'gloss'")

        # Mipmap filter method
        valid_filter_methods = {
            "lanczos", "area", "bilinear", "linear", "nearest", "box", "cubic"
        }
        if self.mipmap.filter_method not in valid_filter_methods:
            errors.append(
                f"mipmap.filter_method must be one of {sorted(valid_filter_methods)}, "
                f"got '{self.mipmap.filter_method}'"
            )

        # Color grading exposure range
        if not (-5.0 <= self.color_grading.exposure_ev <= 5.0):
            errors.append("color_grading.exposure_ev must be in [-5.0, 5.0]")

        # Detail map
        if self.detail_map.detail_uv_scale <= 0:
            errors.append("detail_map.detail_uv_scale must be > 0")
        if not (0.0 <= self.detail_map.detail_strength <= 2.0):
            errors.append("detail_map.detail_strength must be in [0.0, 2.0]")

        # --- Warn on unimplemented fields that users might configure ---
        if self.color_consistency.reference_image:
            if self.validation.strict_material_semantics:
                errors.append(
                    "color_consistency.reference_image is set but NOT YET "
                    "IMPLEMENTED, and strict_material_semantics is enabled. "
                    "Remove the value or disable strict mode."
                )
            else:
                logger.warning(
                    "color_consistency.reference_image is set but NOT YET IMPLEMENTED. "
                    "The value '%s' will have no effect.",
                    self.color_consistency.reference_image,
                )
        if self.detail_map.detail_fade_distance != 5.0:
            if self.validation.strict_material_semantics:
                errors.append(
                    "detail_map.detail_fade_distance is set to "
                    f"{self.detail_map.detail_fade_distance:.2f} but NOT YET "
                    "IMPLEMENTED, and strict_material_semantics is enabled. "
                    "Reset to default (5.0) or disable strict mode."
                )
            else:
                logger.warning(
                    "detail_map.detail_fade_distance is set to %.2f but NOT YET IMPLEMENTED. "
                    "The value will have no effect.",
                    self.detail_map.detail_fade_distance,
                )

        # --- Cross-field validation warnings (non-fatal) ---
        if self.orm_packing.enabled and not self.pbr.enabled:
            logger.warning(
                "ORM packing is enabled but PBR generation is disabled. "
                "ORM maps require roughness/metalness/AO from the PBR phase."
            )
        if self.specular_aa.enabled and not self.normal.enabled:
            logger.warning(
                "Specular AA is enabled but normal map generation is disabled. "
                "Specular AA requires normal map variance data."
            )
        if self.normal.method == "deepbump":
            from . import BIN_DIR
            deepbump_model_dir = BIN_DIR / "DeepBump-8"
            deepbump_onnx = deepbump_model_dir / "deepbump256.onnx"
            if not deepbump_onnx.is_file():
                logger.warning(
                    "Normal method is 'deepbump' but ONNX model not found at %s. "
                    "DeepBump will fall back to hybrid method at runtime.",
                    deepbump_onnx,
                )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        # Apply deferred format_map normalizations now that validation passed.
        for tex_type, normalized in _pending_format_normalizations.items():
            self.compression.format_map[tex_type] = normalized
        if _pending_perf_normalized is not _perf_sentinel:
            self.compression.compressonator_performance = _pending_perf_normalized

    def apply_runtime_fixups(self):
        """Apply runtime-safe config normalization that intentionally mutates state."""
        resolved_device = self.resolve_device()
        if (
            self.upscale.enabled
            and self.upscale.half_precision
            and resolved_device == "cpu"
        ):
            logger.warning(
                "upscale.half_precision is enabled but resolved device is CPU. "
                "Disabling half precision for runtime safety."
            )
            self.upscale.half_precision = False


def _merge_dict_to_dataclass(obj, data: dict, _path: str = ""):
    import dataclasses
    for key, value in data.items():
        if hasattr(obj, key):
            field_val = getattr(obj, key)
            if dataclasses.is_dataclass(field_val) and isinstance(value, dict):
                _merge_dict_to_dataclass(field_val, value, f"{_path}{key}.")
            else:
                full_key = f"{_path}{key}"
                # Reject None for fields with non-None defaults
                if value is None and field_val is not None:
                    logger.warning(
                        f"Config key '{full_key}' is null but field default is "
                        f"{type(field_val).__name__}. Using default value."
                    )
                    continue
                expected_type = type(field_val)
                # Check type compatibility (allow int->float and float->int promotion)
                if (field_val is not None
                        and not isinstance(value, expected_type)
                        and not (expected_type is float
                                 and isinstance(value, int))
                        and not (expected_type is int
                                 and isinstance(value, float)
                                 and value == int(value))):
                    logger.warning(
                        f"Config type mismatch for '{full_key}': "
                        f"expected {expected_type.__name__}, "
                        f"got {type(value).__name__} ({value!r}). "
                        f"Using default value."
                    )
                    continue
                # Promote exact-integer floats to int (e.g. YAML 4.0 -> 4)
                if (expected_type is int and isinstance(value, float)
                        and value == int(value)):
                    value = int(value)
                # Merge dicts instead of replacing (preserves defaults)
                if isinstance(field_val, dict) and isinstance(value, dict):
                    field_val.update(value)
                else:
                    setattr(obj, key, value)
        else:
            full_key = f"{_path}{key}"
            logger.warning(f"Unknown config key ignored: '{full_key}'")
