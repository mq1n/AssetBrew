"""Generate normal and height maps from diffuse or albedo textures.

This phase supports Sobel, hybrid, and DeepBump-backed normal generation with
automatic fallback when optional AI dependencies are unavailable.
"""

import logging
import os
import sys

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from ..config import PipelineConfig, TextureType
from ..core import (
    AssetRecord, load_image, save_image, ensure_rgb,
    get_intermediate_path, luminance_bt709
)

logger = logging.getLogger("asset_pipeline.normal_gen")


class NormalMapGenerator:
    """Generate normal maps and height maps from texture data."""

    def __init__(self, config: PipelineConfig):
        """Initialize generator state and lazy DeepBump handles."""
        self.config = config
        self.cfg = config.normal
        self._deepbump_modules = None  # Lazy-loaded: (utils_inference, normals_to_height) or False
        self._ort_session = None  # Cached ONNX InferenceSession for DeepBump

    def cleanup(self):
        """Release ONNX session and cached module references."""
        self._ort_session = None
        self._deepbump_modules = None

    def __enter__(self):
        """Context manager entry for deterministic cleanup."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit cleanup hook."""
        self.cleanup()
        return False

    def process(self, record: AssetRecord, upscaled_path: str = None,
                albedo_path: str = None) -> dict:
        """Generate normal and optional height outputs for one asset."""
        tex_type = TextureType(record.texture_type)
        result = {
            "normal": None, "height": None,
            "skipped": False, "error": None
        }

        if tex_type not in (TextureType.DIFFUSE, TextureType.ALBEDO, TextureType.UNKNOWN):
            result["skipped"] = True
            logger.debug(
                "Skipping normal generation for %s: unsupported texture type %s",
                record.filename, tex_type.value
            )
            return result

        source_path = None
        for candidate in [albedo_path, upscaled_path]:
            if candidate and os.path.exists(candidate):
                source_path = candidate
                break
        if source_path is None:
            source_path = os.path.join(self.config.input_dir, record.filepath)

        try:
            img = load_image(source_path, max_pixels=self.config.max_image_pixels)
            rgb = ensure_rgb(img)

            # 2. Normal map (done before height so deepbump can derive height from normals)
            normal_map = None
            used_deepbump = False
            height_map_for_sobel = None
            if self.cfg.method == "deepbump":
                normal_map = self._deepbump_normal(rgb)
                if normal_map is not None:
                    used_deepbump = True
                else:
                    logger.warning("DeepBump unavailable, falling back to hybrid method")

            if normal_map is None:
                # Sobel / hybrid path (also serves as deepbump fallback)
                if self.cfg.generate_height:
                    height_map_for_sobel = self._generate_height_map(rgb)
                else:
                    height_map_for_sobel = self._quick_heightmap(rgb)
                is_hybrid = (
                    self.cfg.method == "hybrid"
                    or (self.cfg.method == "deepbump" and not used_deepbump)
                )
                if is_hybrid:
                    sobel_n = self._sobel_normal(height_map_for_sobel)
                    detail_n = self._detail_normal(rgb)
                    sw = self.cfg.sobel_weight
                    normal_map = self._blend_normals(sobel_n, detail_n, sw, 1.0 - sw)
                else:
                    normal_map = self._sobel_normal(height_map_for_sobel)

            # 1. Height map
            if self.cfg.generate_height:
                if used_deepbump:
                    height_map = self._deepbump_height(normal_map)
                else:
                    height_map = height_map_for_sobel
                out_h = get_intermediate_path(
                    record.filepath, "03_normal",
                    self.config.intermediate_dir, suffix="_height", ext=".png"
                )
                save_image(height_map, out_h, bits=16)
                result["height"] = out_h

            # Apply strength
            if self.cfg.strength != 1.0:
                normal_map = self._adjust_normal_strength(normal_map, self.cfg.strength)

            # Validate
            if self.cfg.validate_normals:
                normal_map = self._validate_and_fix(normal_map)

            out_n = get_intermediate_path(
                record.filepath, "03_normal",
                self.config.intermediate_dir, suffix="_normal", ext=".png"
            )
            save_image(normal_map, out_n, bits=16)
            result["normal"] = out_n

            logger.info(f"Normal/height maps generated for {record.filename}")

        except Exception as e:
            logger.error(f"Normal generation failed for {record.filename}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _generate_height_map(self, rgb: np.ndarray) -> np.ndarray:
        """Extract height map: luminance + high-pass + contrast."""
        gray = luminance_bt709(rgb, assume_srgb=True)

        low_freq = gaussian_filter(gray, sigma=self.cfg.height_high_pass_radius)
        high_pass = gray - low_freq + 0.5
        # Favour low-frequency shape (base luminance) over high-pass detail
        # so that POM parallax displaces actual surface depth rather than
        # producing an embossed look from texture detail.
        height = 0.7 * gray + 0.3 * high_pass

        height = self._adjust_contrast(height, self.cfg.height_contrast)

        if self.cfg.height_normalize:
            h_min, h_max = np.min(height), np.max(height)
            if h_max - h_min > 1e-8:
                height = (height - h_min) / (h_max - h_min)
            else:
                height = np.full_like(height, 0.5)

        return height.astype(np.float32)

    def _quick_heightmap(self, rgb: np.ndarray) -> np.ndarray:
        gray = luminance_bt709(rgb, assume_srgb=True)
        return gray.astype(np.float32)

    @staticmethod
    def _adaptive_sobel_ksize(h: int, w: int) -> int:
        """Select Sobel kernel size by resolution for better large-texture gradients."""
        max_dim = max(int(h), int(w))
        if max_dim >= 2048:
            return 7
        if max_dim >= 1024:
            return 5
        return 3

    def _sobel_normal(self, height: np.ndarray) -> np.ndarray:
        """Generate a normal map from a height map via Sobel gradients."""
        if self.cfg.sobel_blur_radius > 0:
            height = gaussian_filter(height, sigma=self.cfg.sobel_blur_radius)

        h, w = height.shape[:2]
        ksize = self._adaptive_sobel_ksize(h, w)
        dx = cv2.Sobel(height.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ksize)
        dy = cv2.Sobel(height.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ksize)

        if self.cfg.invert_y:
            dy = -dy

        # nz_scale controls normal map "flatness". Higher values produce
        # flatter normals (subtle detail); lower values produce steeper angles.
        nz_scale = max(getattr(self.cfg, 'nz_scale', 2.0), 0.01)
        nx, ny, nz = -dx, -dy, np.full_like(dx, nz_scale)
        length = np.sqrt(nx**2 + ny**2 + nz**2)
        length = np.maximum(length, 1e-8)

        normal_map = np.stack([
            nx / length * 0.5 + 0.5,
            ny / length * 0.5 + 0.5,
            nz / length * 0.5 + 0.5
        ], axis=-1)
        return normal_map.astype(np.float32)

    def _detail_normal(self, rgb: np.ndarray) -> np.ndarray:
        """Detail normal map from per-channel RGB gradients."""
        h, w = rgb.shape[:2]
        ksize = self._adaptive_sobel_ksize(h, w)
        grads = []
        for c in range(3):
            channel = rgb[:, :, c]
            dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=ksize)
            dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=ksize)
            grads.append((dx, dy))

        dx_avg = np.mean([g[0] for g in grads], axis=0)
        dy_avg = np.mean([g[1] for g in grads], axis=0)
        del grads

        if self.cfg.invert_y:
            dy_avg = -dy_avg

        nx, ny = -dx_avg, -dy_avg
        nz_scale = max(getattr(self.cfg, 'nz_scale', 2.0), 0.01)
        nz = np.full_like(nx, nz_scale)

        length = np.sqrt(nx**2 + ny**2 + nz**2)
        length = np.maximum(length, 1e-8)

        normal_map = np.stack([
            nx / length * 0.5 + 0.5,
            ny / length * 0.5 + 0.5,
            nz / length * 0.5 + 0.5
        ], axis=-1)
        return normal_map.astype(np.float32)

    def _blend_normals(self, base: np.ndarray, detail: np.ndarray,
                       base_weight: float, detail_weight: float) -> np.ndarray:
        """Blend normals using Whiteout method.

        Whiteout blending (Barré-Brisebois & Hill) preserves both base
        curvature and detail by cross-multiplying Z components:
          result.xy = n1.xy * n2.z + n2.xy * n1.z
          result.z  = n1.z * n2.z

        Both base_weight and detail_weight are applied by interpolating
        toward flat (0,0,1) *before* the blend so the unit-vector
        property is preserved.
        """
        n1 = base * 2.0 - 1.0
        n2 = detail * 2.0 - 1.0

        flat = np.zeros_like(n1)
        flat[:, :, 2] = 1.0

        # Weight base by lerping toward flat normal (0,0,1).
        eff_base = float(np.clip(base_weight, 0.0, 1.0))
        if eff_base < 1.0:
            n1 = n1 * eff_base + flat * (1.0 - eff_base)
            length_n1 = np.sqrt(np.sum(n1 ** 2, axis=-1, keepdims=True))
            n1 = n1 / np.maximum(length_n1, 1e-8)

        # Weight detail by lerping toward flat normal (0,0,1).
        eff_weight = float(np.clip(detail_weight, 0.0, 1.0))
        if eff_weight < 1.0:
            n2 = n2 * eff_weight + flat * (1.0 - eff_weight)
            length_n2 = np.sqrt(np.sum(n2 ** 2, axis=-1, keepdims=True))
            n2 = n2 / np.maximum(length_n2, 1e-8)

        # Whiteout blend
        bx = n1[:, :, 0] * n2[:, :, 2] + n2[:, :, 0] * n1[:, :, 2]
        by = n1[:, :, 1] * n2[:, :, 2] + n2[:, :, 1] * n1[:, :, 2]
        bz = n1[:, :, 2] * n2[:, :, 2]

        length = np.sqrt(bx**2 + by**2 + bz**2)
        length = np.maximum(length, 1e-8)

        result = np.stack([
            bx / length * 0.5 + 0.5,
            by / length * 0.5 + 0.5,
            bz / length * 0.5 + 0.5
        ], axis=-1)
        return result.astype(np.float32)

    def _adjust_normal_strength(self, normal: np.ndarray, strength: float) -> np.ndarray:
        decoded = normal * 2.0 - 1.0
        decoded[:, :, 0] *= strength
        decoded[:, :, 1] *= strength
        # Clamp XY magnitude to prevent degenerate normals at high strength
        xy_sq_sum = decoded[:, :, 0] ** 2 + decoded[:, :, 1] ** 2
        max_xy_sq = 1.0 - 1e-4
        excess = xy_sq_sum > max_xy_sq
        if np.any(excess):
            scale_down = np.where(excess, np.sqrt(max_xy_sq / np.maximum(xy_sq_sum, 1e-8)), 1.0)
            decoded[:, :, 0] *= scale_down
            decoded[:, :, 1] *= scale_down
        z_sq = 1.0 - decoded[:, :, 0]**2 - decoded[:, :, 1]**2
        # Clamp to a small positive floor to prevent degenerate zero-Z
        # normals when strength > ~2.0 pushes XY beyond the unit sphere.
        decoded[:, :, 2] = np.sqrt(np.maximum(z_sq, 1e-6))
        # Re-normalize to ensure unit length after clamping
        length = np.sqrt(np.sum(decoded**2, axis=-1, keepdims=True))
        decoded = decoded / np.maximum(length, 1e-8)
        return (decoded * 0.5 + 0.5).astype(np.float32)

    def _validate_and_fix(self, normal: np.ndarray) -> np.ndarray:
        decoded = normal * 2.0 - 1.0

        nan_mask = ~np.isfinite(decoded)
        if np.any(nan_mask):
            logger.warning("Found NaN/Inf in normal map, fixing...")
            decoded[nan_mask] = 0.0
            zero_mask = np.all(decoded == 0, axis=-1)
            decoded[zero_mask, 2] = 1.0

        length = np.sqrt(np.sum(decoded**2, axis=-1, keepdims=True))
        length = np.maximum(length, 1e-8)
        decoded = decoded / length
        decoded[:, :, 2] = np.abs(decoded[:, :, 2])

        return (decoded * 0.5 + 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    # DeepBump AI normal / height generation
    # ------------------------------------------------------------------

    def _load_deepbump(self):
        """Lazy-load DeepBump modules. Returns (utils_inference, normals_to_height) or None."""
        if self._deepbump_modules is not None:
            return self._deepbump_modules if self._deepbump_modules is not False else None
        try:
            import onnxruntime  # noqa: F401 -- ensure available before loading DeepBump
        except ImportError:
            logger.warning("onnxruntime not installed -- DeepBump disabled")
            self._deepbump_modules = False
            return None

        from .. import BIN_DIR
        deepbump_dir = str(BIN_DIR / "DeepBump-8")
        onnx_path = os.path.join(deepbump_dir, "deepbump256.onnx")
        if not os.path.isfile(onnx_path):
            logger.warning(f"DeepBump model not found at {onnx_path} -- DeepBump disabled")
            self._deepbump_modules = False
            return None

        _path_added = False
        if deepbump_dir not in sys.path:
            sys.path.insert(0, deepbump_dir)
            _path_added = True
        try:
            import utils_inference as db_utils_inf
            import module_normals_to_height as db_n2h
            self._deepbump_modules = (db_utils_inf, db_n2h, onnx_path)
            logger.info("DeepBump modules loaded successfully")
            return self._deepbump_modules
        except Exception as e:
            logger.warning(f"Failed to load DeepBump modules: {e}")
            self._deepbump_modules = False
            return None
        finally:
            if _path_added and deepbump_dir in sys.path:
                sys.path.remove(deepbump_dir)

    def _deepbump_normal(self, rgb: np.ndarray) -> np.ndarray | None:
        """Generate normal map using DeepBump AI model. Returns H,W,C or None on failure."""
        mods = self._load_deepbump()
        if mods is None:
            return None

        db_utils_inf, _, onnx_path = mods
        try:
            import onnxruntime as ort

            # DeepBump was trained on perceptual (sRGB) luminance, not linear.
            # Use assume_srgb=False to pass perceptual values directly.
            img_gray = luminance_bt709(rgb, assume_srgb=False)[None, :, :].astype(np.float32)

            # Tile the image
            tile_size = 256
            overlaps = {"SMALL": tile_size // 6, "MEDIUM": tile_size // 4, "LARGE": tile_size // 2}
            overlap = self.cfg.deepbump_overlap
            if overlap not in overlaps:
                logger.warning(
                    f"Invalid deepbump_overlap '{overlap}', "
                    f"must be one of {sorted(overlaps)}. Defaulting to LARGE."
                )
                overlap = "LARGE"
            stride_size = tile_size - overlaps[overlap]

            tiles, paddings = db_utils_inf.tiles_split(
                img_gray, (tile_size, tile_size), (stride_size, stride_size)
            )

            # Reuse cached ONNX session (expensive to create per asset)
            if self._ort_session is None:
                providers = []
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
                self._ort_session = ort.InferenceSession(
                    onnx_path, providers=providers
                )
            ort_session = self._ort_session

            # Predict normals per tile
            pred_tiles = db_utils_inf.tiles_infer(tiles, ort_session)

            # Merge tiles
            pred_img = db_utils_inf.tiles_merge(
                pred_tiles,
                (stride_size, stride_size),
                (3, img_gray.shape[1], img_gray.shape[2]),
                paddings,
            )

            # Normalize to unit vectors
            pred_img = db_utils_inf.normalize(pred_img)

            # Convert C,H,W -> H,W,C and remap [-1,1] -> [0,1]
            normal_map = np.transpose(pred_img, (1, 2, 0)).astype(np.float32)
            normal_map = normal_map * 0.5 + 0.5

            logger.info("DeepBump AI normal map generated")
            return normal_map

        except Exception as e:
            logger.warning(f"DeepBump normal generation failed: {e}")
            return None

    def _height_from_normals_fallback(self, normal_map: np.ndarray) -> np.ndarray:
        """Derive a height map from a normal map using Frankot-Chellappa FFT integration.

        Unlike _generate_height_map (designed for diffuse/albedo input), this
        method treats the input as linear tangent-space normals: it integrates
        the XY gradient encoded in the normal map.

        Uses Frankot-Chellappa (1988) FFT-based integration to avoid the
        cumulative ramp artifacts of row/column cumulative-sum approaches.
        """
        rgb = ensure_rgb(normal_map)
        decoded = rgb * 2.0 - 1.0
        nx = decoded[:, :, 0]
        ny = decoded[:, :, 1]
        nz = np.maximum(decoded[:, :, 2], 1e-8)

        # Surface gradients from normal map
        h, w = nx.shape

        # Memory guard: FFT uses float64 complex128 arrays.
        # Each complex128 array is 16 bytes/pixel; we need ~5 such arrays
        # (p, q, P, Q, Z) so budget ~80 bytes/pixel.  Cap at ~2 GB.
        _fft_bytes = h * w * 80
        _MAX_FFT_BYTES = 2 * 1024 ** 3  # 2 GB
        if _fft_bytes > _MAX_FFT_BYTES:
            logger.warning(
                "Image %dx%d too large for FFT height integration "
                "(estimated %.1f GB); skipping",
                w, h, _fft_bytes / (1024 ** 3),
            )
            return np.full((h, w), 0.5, dtype=np.float32)

        p = (-nx / nz).astype(np.float64)  # dz/dx
        q = (-ny / nz).astype(np.float64)  # dz/dy

        # For non-tileable content, mirror-pad before FFT to eliminate
        # boundary discontinuities without forcing height to zero (which
        # causes the bowl artifact from the old Hann window approach).
        seamless = getattr(self.cfg, 'deepbump_seamless_height', True)
        pad_h = pad_w = 0
        if not seamless:
            pad_h = h // 4
            pad_w = w // 4
            p = np.pad(p, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
            q = np.pad(q, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        ph, pw = p.shape

        # Frequency coordinates
        u = np.fft.fftfreq(pw) * 2.0 * np.pi
        v = np.fft.fftfreq(ph) * 2.0 * np.pi
        u, v = np.meshgrid(u, v)

        # FFT of gradients
        P = np.fft.fft2(p)
        Q = np.fft.fft2(q)

        # Frankot-Chellappa: Z = F^-1[ (-j*u*P - j*v*Q) / (u^2 + v^2) ]
        denom = u ** 2 + v ** 2
        denom[0, 0] = 1.0  # avoid division by zero at DC
        Z = (-1j * u * P - 1j * v * Q) / denom
        Z[0, 0] = 0.0  # DC component (mean height) = 0

        height = np.real(np.fft.ifft2(Z)).astype(np.float32)
        del p, q, P, Q, Z, u, v, denom

        # Crop back to original dimensions if mirror-padded
        if pad_h > 0 or pad_w > 0:
            height = height[pad_h:pad_h + h, pad_w:pad_w + w]

        # Normalize to [0, 1]
        h_min, h_max = float(height.min()), float(height.max())
        if h_max - h_min > 1e-8:
            height = (height - h_min) / (h_max - h_min)
        else:
            height = np.full_like(height, 0.5)
        return height.astype(np.float32)

    def _deepbump_height(self, normal_map: np.ndarray) -> np.ndarray:
        """Generate height map from normals using Frankot-Chellappa. Input/output H,W,C / H,W."""
        mods = self._load_deepbump()
        if mods is None:
            # Fallback: gradient integration from normal map (no sRGB decode)
            logger.warning(
                "DeepBump unavailable -- falling back to gradient-based height "
                "from normal map. Quality may be reduced."
            )
            return self._height_from_normals_fallback(normal_map)

        _, db_n2h, _ = mods
        try:
            # Convert H,W,C -> C,H,W
            normals_chw = np.transpose(normal_map, (2, 0, 1)).astype(np.float32)

            # apply() returns C,H,W (3-channel, all identical)
            seamless = self.cfg.deepbump_seamless_height
            height_chw = db_n2h.apply(
                normals_chw, seamless, progress_callback=None
            )

            # Extract single channel -> H,W
            height_map = height_chw[0].astype(np.float32)

            logger.info("DeepBump Frankot-Chellappa height map generated")
            return height_map

        except Exception as e:
            logger.warning(f"DeepBump height generation failed: {e}, using gradient fallback")
            return self._height_from_normals_fallback(normal_map)

    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(img)
        return np.clip(mean + (img - mean) * factor, 0, 1)
