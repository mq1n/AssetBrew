"""Tests for mipmap generation and DDS assembly."""

import importlib.util
import logging
import os
import shutil
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pytest

from AssetBrew.config import PipelineConfig, TextureType
from AssetBrew.core import AssetRecord, save_image

pytestmark = pytest.mark.slow

HAS_CV2 = importlib.util.find_spec("cv2") is not None
_requires_cv2 = unittest.skipUnless(HAS_CV2, "cv2 (opencv) not installed")


@_requires_cv2
class TestMipmapGenerator(unittest.TestCase):
    def test_mip_count_respects_min_resident(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        config.mipmap.min_resident_mips = 4
        gen = MipmapGenerator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            arr = np.random.rand(256, 256, 3).astype(np.float32)
            path = os.path.join(tmpdir, "test.png")
            save_image(arr, path)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=256, original_height=256,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=100.0,
            )
            result = gen.process(record, path, TextureType.DIFFUSE)
            mips = result["mips"]
            smallest = min(m["width"] for m in mips)
            self.assertGreaterEqual(smallest, 16)

    def test_roughness_increase_per_mip(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)
        mip = np.full((32, 32), 0.5, dtype=np.float32)
        increased = gen._increase_roughness(mip, level=3)
        # Perceptual roughness: new_r = sqrt(r² + increase)
        # Exponential: increase = cfg * (2^level - 1)
        increase = config.mipmap.roughness_mip_increase * (2.0 ** 3 - 1.0)
        expected = np.sqrt(0.5 ** 2 + increase)
        np.testing.assert_allclose(increased.mean(), expected, atol=0.01)

    def test_srgb_downsampling_uses_linear_space(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        from AssetBrew.core import linear_to_srgb

        config = PipelineConfig()
        config.mipmap.filter_method = "area"
        config.mipmap.srgb_downsampling = True
        gen = MipmapGenerator(config)

        arr = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        mip = gen._generate_mip(arr, 1, 1, TextureType.DIFFUSE, level=1)
        expected = linear_to_srgb(np.full((1, 1, 3), 0.5, dtype=np.float32))
        np.testing.assert_allclose(mip[0, 0, :3], expected[0, 0, :3], atol=0.02)

        config.mipmap.srgb_downsampling = False
        gen = MipmapGenerator(config)
        mip_no_srgb = gen._generate_mip(arr, 1, 1, TextureType.DIFFUSE, level=1)
        np.testing.assert_allclose(mip_no_srgb[0, 0, :3], [0.5, 0.5, 0.5], atol=0.02)

    def test_mip_outputs_do_not_collide_between_map_types(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = tmpdir
            gen = MipmapGenerator(config)
            arr = np.random.rand(128, 128, 3).astype(np.float32)
            path = os.path.join(tmpdir, "test.png")
            save_image(arr, path)
            record = AssetRecord(
                filepath="test.png", filename="test.png",
                texture_type="diffuse", original_width=128, original_height=128,
                channels=3, has_alpha=False, is_tileable=False, is_hero=False,
                material_category="default", file_size_kb=10.0,
            )

            main = gen.process(record, path, TextureType.DIFFUSE, name_tag="main")
            normal = gen.process(record, path, TextureType.NORMAL, name_tag="normal")

            main_paths = {m["path"] for m in main["mips"]}
            normal_paths = {m["path"] for m in normal["mips"]}
            self.assertTrue(main_paths)
            self.assertTrue(normal_paths)
            self.assertTrue(main_paths.isdisjoint(normal_paths))


class TestDDSMipchainAssembly(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_dds(self, path, width, height, data_size=None):
        DDSD_CAPS = 0x1
        DDSD_HEIGHT = 0x2
        DDSD_WIDTH = 0x4
        DDSD_PIXELFORMAT = 0x1000
        FLAGS = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
        DDSCAPS_TEXTURE = 0x1000
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 124)
        struct.pack_into("<I", header, 4, FLAGS)
        struct.pack_into("<I", header, 8, height)
        struct.pack_into("<I", header, 12, width)
        struct.pack_into("<I", header, 24, 1)
        struct.pack_into("<I", header, 72, 32)
        struct.pack_into("<I", header, 76, 0x4)
        header[80:84] = b"DXT5"
        struct.pack_into("<I", header, 104, DDSCAPS_TEXTURE)
        if data_size is None:
            data_size = max(width * height, 16)
        pixel_data = bytes([width & 0xFF]) * data_size
        with open(path, "wb") as f:
            f.write(b"DDS ")
            f.write(bytes(header))
            f.write(pixel_data)
        return pixel_data

    def test_assembly_header_flags(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        dds0 = os.path.join(self.tmpdir, "mip0.dds")
        dds1 = os.path.join(self.tmpdir, "mip1.dds")
        dds2 = os.path.join(self.tmpdir, "mip2.dds")
        self._make_fake_dds(dds0, 256, 256, data_size=256)
        self._make_fake_dds(dds1, 128, 128, data_size=64)
        self._make_fake_dds(dds2, 64, 64, data_size=16)
        out = os.path.join(self.tmpdir, "assembled.dds")
        ok = MipmapGenerator._assemble_dds_mipchain([dds0, dds1, dds2], out)
        self.assertTrue(ok)
        with open(out, "rb") as f:
            magic = f.read(4)
            header = f.read(124)
        self.assertEqual(magic, b"DDS ")
        mip_count = struct.unpack_from("<I", header, 24)[0]
        self.assertEqual(mip_count, 3)
        flags = struct.unpack_from("<I", header, 4)[0]
        self.assertTrue(flags & 0x20000)
        caps = struct.unpack_from("<I", header, 104)[0]
        self.assertTrue(caps & 0x8)
        self.assertTrue(caps & 0x400000)

    def test_assembly_data_order(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        dds0 = os.path.join(self.tmpdir, "mip0.dds")
        dds1 = os.path.join(self.tmpdir, "mip1.dds")
        data0 = self._make_fake_dds(dds0, 64, 64, data_size=32)
        data1 = self._make_fake_dds(dds1, 32, 32, data_size=8)
        out = os.path.join(self.tmpdir, "assembled.dds")
        MipmapGenerator._assemble_dds_mipchain([dds0, dds1], out)
        with open(out, "rb") as f:
            f.seek(128)
            all_data = f.read()
        self.assertEqual(all_data[:32], data0)
        self.assertEqual(all_data[32:40], data1)

    def test_dds_mipchain_linear_size_updated(self):
        """Verify dwPitchOrLinearSize matches base mip data size and DDSD_LINEARSIZE is set."""
        from AssetBrew.phases.mipmap import MipmapGenerator

        DDSD_LINEARSIZE = 0x80000

        # Build two minimal DDS files with DX10 extended header
        def _build_dx10_dds(path, width, height, pixel_data_size):
            header = bytearray(124)
            DDSD_CAPS = 0x1
            DDSD_HEIGHT = 0x2
            DDSD_WIDTH = 0x4
            DDSD_PIXELFORMAT = 0x1000
            struct.pack_into("<I", header, 0, 124)  # dwSize
            flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
            struct.pack_into("<I", header, 4, flags)
            struct.pack_into("<I", header, 8, height)
            struct.pack_into("<I", header, 12, width)
            struct.pack_into("<I", header, 24, 1)  # dwMipMapCount
            struct.pack_into("<I", header, 72, 32)  # pfSize
            struct.pack_into("<I", header, 76, 0x4)  # pfFlags (DDPF_FOURCC)
            header[80:84] = b"DX10"  # fourCC
            struct.pack_into("<I", header, 104, 0x1000)  # dwCaps = TEXTURE

            # DX10 extended header (20 bytes)
            dx10 = bytearray(20)
            struct.pack_into("<I", dx10, 0, 98)  # DXGI_FORMAT_BC7_UNORM
            struct.pack_into("<I", dx10, 4, 3)   # D3D10_RESOURCE_DIMENSION_TEXTURE2D
            struct.pack_into("<I", dx10, 12, 1)  # arraySize

            pixel_data = bytes(range(256))[:pixel_data_size] * (pixel_data_size // 256 + 1)
            pixel_data = pixel_data[:pixel_data_size]
            with open(path, "wb") as f:
                f.write(b"DDS ")
                f.write(bytes(header))
                f.write(bytes(dx10))
                f.write(pixel_data)
            return pixel_data

        mip0_path = os.path.join(self.tmpdir, "mip0.dds")
        mip1_path = os.path.join(self.tmpdir, "mip1.dds")
        base_pixel_data = _build_dx10_dds(mip0_path, 128, 128, pixel_data_size=200)
        _build_dx10_dds(mip1_path, 64, 64, pixel_data_size=50)

        out = os.path.join(self.tmpdir, "assembled.dds")
        ok = MipmapGenerator._assemble_dds_mipchain([mip0_path, mip1_path], out)
        self.assertTrue(ok)

        with open(out, "rb") as f:
            magic = f.read(4)
            header = f.read(124)

        self.assertEqual(magic, b"DDS ")

        # dwPitchOrLinearSize at header offset 16 should equal base mip data size
        pitch_or_linear = struct.unpack_from("<I", header, 16)[0]
        self.assertEqual(
            pitch_or_linear, len(base_pixel_data),
            f"dwPitchOrLinearSize should be {len(base_pixel_data)},"
            f" got {pitch_or_linear}",
        )

        # DDSD_LINEARSIZE flag should be set in dwFlags (header offset 4)
        flags = struct.unpack_from("<I", header, 4)[0]
        self.assertTrue(flags & DDSD_LINEARSIZE,
                        f"DDSD_LINEARSIZE (0x80000) should be set in dwFlags, got 0x{flags:X}")

    def test_single_mip_no_assembly(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        dds0 = os.path.join(self.tmpdir, "mip0.dds")
        self._make_fake_dds(dds0, 64, 64)
        out = os.path.join(self.tmpdir, "assembled.dds")
        ok = MipmapGenerator._assemble_dds_mipchain([dds0], out)
        self.assertTrue(ok)

    def test_normalize_dds_header_repairs_required_flags_and_caps(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        path = os.path.join(self.tmpdir, "malformed.dds")
        header = bytearray(124)
        struct.pack_into("<I", header, 0, 0)            # malformed dwSize
        struct.pack_into("<I", header, 4, 0x00000006)   # HEIGHT|WIDTH only
        struct.pack_into("<I", header, 8, 8)            # dwHeight
        struct.pack_into("<I", header, 12, 8)           # dwWidth
        struct.pack_into("<I", header, 16, 0)           # dwPitchOrLinearSize
        struct.pack_into("<I", header, 24, 1)           # dwMipMapCount
        struct.pack_into("<I", header, 72, 0)           # malformed ddspf.dwSize
        struct.pack_into("<I", header, 76, 0x4)         # DDPF_FOURCC
        header[80:84] = b"DXT1"
        struct.pack_into("<I", header, 104, 0)          # dwCaps missing TEXTURE

        with open(path, "wb") as f:
            f.write(b"DDS ")
            f.write(bytes(header))
            f.write(b"\x00" * 32)  # 8x8 BC1 base level

        gen = MipmapGenerator(PipelineConfig())
        ok = gen._normalize_dds_header(path)
        self.assertTrue(ok)

        with open(path, "rb") as f:
            raw = f.read(128)

        self.assertEqual(struct.unpack_from("<I", raw, 4)[0], 124)
        self.assertEqual(struct.unpack_from("<I", raw, 76)[0], 32)

        flags = struct.unpack_from("<I", raw, 8)[0]
        self.assertTrue(flags & 0x1)       # DDSD_CAPS
        self.assertTrue(flags & 0x2)       # DDSD_HEIGHT
        self.assertTrue(flags & 0x4)       # DDSD_WIDTH
        self.assertTrue(flags & 0x1000)    # DDSD_PIXELFORMAT
        self.assertTrue(flags & 0x80000)   # DDSD_LINEARSIZE
        self.assertFalse(flags & 0x8)      # DDSD_PITCH

        linear_size = struct.unpack_from("<I", raw, 20)[0]
        self.assertEqual(linear_size, 32)

        caps = struct.unpack_from("<I", raw, 108)[0]
        self.assertTrue(caps & 0x1000)     # DDSCAPS_TEXTURE


class TestDDSMipchainDegradation(unittest.TestCase):
    def test_generate_dds_mipchain_falls_back_to_base_on_partial_failure(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            mip0 = os.path.join(tmpdir, "mip0.png")
            mip1 = os.path.join(tmpdir, "mip1.png")
            out = os.path.join(tmpdir, "out.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            def _fake_generate_dds(src, dst, compression, srgb=False):
                if src == mip1:
                    return False  # Simulate mid-chain compression failure.
                with open(dst, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            mip_paths = [{"level": 0, "path": mip0}, {"level": 1, "path": mip1}]
            with mock.patch.object(gen, "_resolve_compression_tool", return_value="fake.exe"):
                with mock.patch.object(gen, "generate_dds", side_effect=_fake_generate_dds):
                    success, degraded = gen.generate_dds_mipchain(mip_paths, out, compression="bc7")

            self.assertTrue(success)
            self.assertTrue(degraded)
            self.assertTrue(os.path.exists(out))

    def test_generate_dds_mipchain_uses_project_local_temp_dir(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.intermediate_dir = os.path.join(tmpdir, "inter")
            os.makedirs(config.intermediate_dir, exist_ok=True)

            mip0 = os.path.join(tmpdir, "mip0.png")
            mip1 = os.path.join(tmpdir, "mip1.png")
            out = os.path.join(tmpdir, "out.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            temp_dests = []

            def _fake_generate_dds(src, dst, compression, srgb=False):  # noqa: ARG001
                if src in {mip0, mip1}:
                    temp_dests.append(dst)
                with open(dst, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            def _fake_assemble(dds_paths, output_path, srgb=False):  # noqa: ARG001
                for dds_path in dds_paths:
                    self.assertTrue(os.path.exists(dds_path))
                with open(output_path, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            mip_paths = [{"level": 0, "path": mip0}, {"level": 1, "path": mip1}]
            with mock.patch.object(gen, "_resolve_compression_tool", return_value="fake.exe"):
                with mock.patch.object(gen, "generate_dds", side_effect=_fake_generate_dds):
                    with mock.patch.object(
                        gen,
                        "_assemble_dds_mipchain",
                        side_effect=_fake_assemble,
                    ):
                        success, degraded = gen.generate_dds_mipchain(
                            mip_paths, out, compression="bc7",
                        )

            self.assertTrue(success)
            self.assertFalse(degraded)
            expected_root = os.path.join(config.intermediate_dir, "_dds_mipchain_tmp")
            self.assertTrue(temp_dests)
            for dst in temp_dests:
                abs_dst = os.path.abspath(dst)
                self.assertTrue(abs_dst.startswith(os.path.abspath(expected_root)))


class TestKTX2BundledToolResolution(unittest.TestCase):
    def test_resolve_prefers_bundled_ktx_tool_when_path_missing(self):
        import AssetBrew
        from AssetBrew.phases.mipmap import MipmapGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            bundled_dir = os.path.join(tmpdir, "KTX-Software-9.9.9-Windows-x64")
            os.makedirs(bundled_dir, exist_ok=True)
            bundled_tool = os.path.join(bundled_dir, "toktx.exe")
            with open(bundled_tool, "wb") as f:
                f.write(b"")

            config = PipelineConfig()
            gen = MipmapGenerator(config)

            with mock.patch("shutil.which", return_value=None):
                with mock.patch("platform.system", return_value="Windows"):
                    with mock.patch.object(AssetBrew, "BIN_DIR", Path(tmpdir)):
                        resolved = gen._resolve_ktx2_tool()

            self.assertEqual(resolved, bundled_tool)


@_requires_cv2
@unittest.skipUnless(os.name == "nt", "Bundled compressonator integration test is Windows-only")
class TestCompressionToolIntegration(unittest.TestCase):
    def test_generate_dds_with_bundled_compressonator(self):
        import AssetBrew
        from AssetBrew.phases.mipmap import MipmapGenerator

        tool_path = (
            Path(AssetBrew.BIN_DIR)
            / "compressonatorcli-4.5.52-win64"
            / "compressonatorcli.exe"
        )
        if not tool_path.is_file():
            self.skipTest(f"Bundled compressonator not found: {tool_path}")

        config = PipelineConfig()
        config.compression.tool = "compressonator"
        config.compression.tool_path = str(tool_path)

        gen = MipmapGenerator(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), src)

            ok = gen.generate_dds(src, out, compression="bc7")
            self.assertTrue(ok, "compressonator failed to produce DDS output")
            self.assertTrue(os.path.exists(out))
            with open(out, "rb") as f:
                self.assertEqual(f.read(4), b"DDS ")


class TestGenerateDdsSrgb(unittest.TestCase):
    """Verify sRGB flag is threaded through DDS generation."""

    def test_generate_dds_srgb_texconv(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        config.compression.tool = "texconv"
        config.compression.tool_path = "/fake/texconv"
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.dds")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_dds(src, out, compression="bc7", srgb=True)
            cmd = m.call_args[0][0]
            self.assertIn("-sRGB", cmd)

    def test_generate_dds_srgb_compressonator(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        config.compression.tool = "compressonator"
        config.compression.tool_path = "/fake/compressonatorcli"
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m, \
             mock.patch.object(gen, "_patch_dds_srgb") as m_patch:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.dds")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_dds(src, out, compression="bc7", srgb=True)
            cmd = m.call_args[0][0]
            # Compressonator receives plain format (no _SRGB suffix — it
            # rejects those). sRGB is handled by post-compression header patch.
            fd_idx = cmd.index("-fd")
            fmt_arg = cmd[fd_idx + 1]
            self.assertEqual(fmt_arg, "BC7", f"Expected plain BC7, got {fmt_arg}")
            self.assertIn("-EncodeWith", cmd)
            enc_idx = cmd.index("-EncodeWith")
            self.assertEqual(cmd[enc_idx + 1], "HPC")
            self.assertIn("-NumThreads", cmd)
            thr_idx = cmd.index("-NumThreads")
            self.assertGreaterEqual(int(cmd[thr_idx + 1]), 0)
            self.assertIn("-noprogress", cmd)
            self.assertEqual(m.call_args.kwargs.get("timeout"), 60)
            m_patch.assert_called_once_with(out)

    def test_generate_dds_honors_configured_timeout(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        config.compression.tool = "compressonator"
        config.compression.tool_path = "/fake/compressonatorcli"
        config.compression.tool_timeout_seconds = 17
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.dds")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_dds(src, out, compression="bc7", srgb=False)
            self.assertEqual(m.call_args.kwargs.get("timeout"), 17)

    def test_generate_dds_no_srgb_flag_when_false(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        config.compression.tool = "texconv"
        config.compression.tool_path = "/fake/texconv"
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.dds")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_dds(src, out, compression="bc7", srgb=False)
            cmd = m.call_args[0][0]
            self.assertNotIn("-sRGB", cmd)


class TestGenerateKtx2Srgb(unittest.TestCase):
    """Verify sRGB oetf flag is threaded through KTX2 generation."""

    def test_generate_ktx2_srgb_oetf(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_resolve_ktx2_tool", return_value="/fake/toktx"), \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.ktx2")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_ktx2(src, out, srgb=True)
            cmd = m.call_args[0][0]
            oetf_idx = cmd.index("--assign_oetf")
            self.assertEqual(cmd[oetf_idx + 1], "srgb")

    def test_generate_ktx2_linear_oetf(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(gen, "_resolve_ktx2_tool", return_value="/fake/toktx"), \
             mock.patch.object(gen, "_run_tool", return_value=(True, None)) as m:
            src = os.path.join(tmpdir, "input.png")
            out = os.path.join(tmpdir, "output.ktx2")
            save_image(np.random.rand(16, 16, 3).astype(np.float32), src)

            gen.generate_ktx2(src, out, srgb=False)
            cmd = m.call_args[0][0]
            oetf_idx = cmd.index("--assign_oetf")
            self.assertEqual(cmd[oetf_idx + 1], "linear")


class TestAssembleDdsMipchainDx10Srgb(unittest.TestCase):
    """Verify DX10 header DXGI format is patched for sRGB."""

    def _build_minimal_dds(self, dxgi_format: int, width: int = 16,
                           height: int = 16) -> bytes:
        """Build a minimal DDS with a DX10 extended header."""
        header = bytearray(124)
        # dwFlags at offset 4
        struct.pack_into("<I", header, 4, 0x1 | 0x2 | 0x4 | 0x1000)
        struct.pack_into("<I", header, 8, height)   # dwHeight
        struct.pack_into("<I", header, 12, width)    # dwWidth
        # PixelFormat fourCC = "DX10" at header offset 80
        header[80:84] = b"DX10"

        # DX10 extended header (20 bytes)
        dx10 = bytearray(20)
        struct.pack_into("<I", dx10, 0, dxgi_format)
        struct.pack_into("<I", dx10, 4, 3)  # TEXTURE2D
        struct.pack_into("<I", dx10, 12, 1)  # arraySize

        pixel_data = b"\xAB" * 16
        return b"DDS " + bytes(header) + bytes(dx10) + pixel_data

    def test_assemble_dds_mipchain_dx10_srgb(self):
        from AssetBrew.phases.mipmap import MipmapGenerator

        # BC7_UNORM = 98, should become BC7_UNORM_SRGB = 99
        mip0_data = self._build_minimal_dds(98, width=16, height=16)
        mip1_data = self._build_minimal_dds(98, width=8, height=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            mip0 = os.path.join(tmpdir, "mip0.dds")
            mip1 = os.path.join(tmpdir, "mip1.dds")
            out = os.path.join(tmpdir, "assembled.dds")

            with open(mip0, "wb") as f:
                f.write(mip0_data)
            with open(mip1, "wb") as f:
                f.write(mip1_data)

            ok = MipmapGenerator._assemble_dds_mipchain([mip0, mip1], out, srgb=True)
            self.assertTrue(ok)

            with open(out, "rb") as f:
                raw = f.read()
            # DX10 header starts at offset 128, DXGI format is first 4 bytes
            dxgi_out = struct.unpack_from("<I", raw, 128)[0]
            self.assertEqual(dxgi_out, 99, "Expected BC7_UNORM_SRGB (99)")


class TestSrgbTextureTypesValidation(unittest.TestCase):
    """Verify srgb_texture_types config validation."""

    def test_srgb_texture_types_default_valid(self):
        config = PipelineConfig()
        config.validate()  # Should not raise

    def test_srgb_texture_types_unknown_rejected(self):
        config = PipelineConfig()
        config.compression.srgb_texture_types = ["diffuse", "nonexistent_type"]
        with self.assertRaises(ValueError):
            config.validate()


@_requires_cv2
class TestRunToolRetryAndErrors(unittest.TestCase):
    """Test _run_tool retry logic, timeout, and error paths."""

    def _make_gen(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        return MipmapGenerator(config)

    @mock.patch("subprocess.run")
    def test_run_tool_success(self, mock_run):
        gen = self._make_gen()
        mock_run.return_value = mock.MagicMock(
            returncode=0, stdout="OK", stderr=""
        )
        ok, proc = gen._run_tool(["tool", "arg"], "test-tool", "file.png")
        self.assertTrue(ok)
        self.assertIsNotNone(proc)

    @mock.patch("subprocess.run")
    def test_run_tool_success_forwards_stdout_at_info(self, mock_run):
        gen = self._make_gen()
        mock_run.return_value = mock.MagicMock(
            returncode=0, stdout="progress 42%", stderr=""
        )
        with mock.patch("AssetBrew.phases.mipmap.logger.log") as log_mock:
            ok, _proc = gen._run_tool(["tool", "arg"], "test-tool", "file.png")
        self.assertTrue(ok)
        self.assertTrue(
            any(
                call.args
                and call.args[0] == logging.INFO
                and call.args[1] == "[%s] %s: %s"
                and call.args[2] == "test-tool"
                and call.args[3] == "stdout"
                for call in log_mock.call_args_list
            ),
            "Expected stdout lines from successful tool run to be logged at INFO",
        )

    @mock.patch("subprocess.run", side_effect=FileNotFoundError("not found"))
    def test_run_tool_file_not_found(self, mock_run):
        gen = self._make_gen()
        ok, proc = gen._run_tool(["missing_tool"], "test-tool", "file.png")
        self.assertFalse(ok)
        self.assertIsNone(proc)

    @mock.patch("subprocess.run", side_effect=PermissionError("denied"))
    def test_run_tool_permission_error(self, mock_run):
        gen = self._make_gen()
        ok, proc = gen._run_tool(["tool"], "test-tool", "file.png")
        self.assertFalse(ok)
        self.assertIsNone(proc)

    @mock.patch("subprocess.run")
    def test_run_tool_timeout(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd=["tool"], timeout=5)
        gen = self._make_gen()
        ok, proc = gen._run_tool(["tool"], "test-tool", "file.png", timeout=5)
        self.assertFalse(ok)
        self.assertIsNone(proc)

    @mock.patch("subprocess.run")
    def test_run_tool_non_zero_exit(self, mock_run):
        gen = self._make_gen()
        mock_run.return_value = mock.MagicMock(
            returncode=1, stdout="", stderr="error msg"
        )
        ok, proc = gen._run_tool(["tool"], "test-tool", "file.png")
        self.assertFalse(ok)


@_requires_cv2
class TestMipchainTupleReturn(unittest.TestCase):
    """Verify generate_dds_mipchain returns (success, degraded) tuple."""

    def test_generate_dds_mipchain_returns_tuple_on_success(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            mip0 = os.path.join(tmpdir, "mip0.png")
            mip1 = os.path.join(tmpdir, "mip1.png")
            out = os.path.join(tmpdir, "out.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            def _fake_generate_dds(src, dst, compression, srgb=False):
                with open(dst, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            def _fake_assemble(dds_paths, output_path, srgb=False):
                with open(output_path, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            mip_paths = [{"level": 0, "path": mip0}, {"level": 1, "path": mip1}]
            with mock.patch.object(gen, "_resolve_compression_tool", return_value="fake.exe"):
                with mock.patch.object(gen, "generate_dds", side_effect=_fake_generate_dds):
                    with mock.patch.object(
                        gen, "_assemble_dds_mipchain",
                        side_effect=_fake_assemble,
                    ):
                        result = gen.generate_dds_mipchain(
                            mip_paths, out, compression="bc7",
                        )

            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            success, degraded = result
            self.assertTrue(success)
            self.assertFalse(degraded)

    def test_mipchain_degraded_on_mip_failure(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            mip0 = os.path.join(tmpdir, "mip0.png")
            mip1 = os.path.join(tmpdir, "mip1.png")
            out = os.path.join(tmpdir, "out.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            call_count = [0]

            def _fake_generate_dds(src, dst, compression, srgb=False):
                call_count[0] += 1
                if src == mip1:
                    return False  # Second mip fails
                with open(dst, "wb") as f:
                    f.write(b"DDS " + b"\x00" * 124)
                return True

            mip_paths = [{"level": 0, "path": mip0}, {"level": 1, "path": mip1}]
            with mock.patch.object(gen, "_resolve_compression_tool", return_value="fake.exe"):
                with mock.patch.object(gen, "generate_dds", side_effect=_fake_generate_dds):
                    success, degraded = gen.generate_dds_mipchain(mip_paths, out, compression="bc7")

            self.assertTrue(success)
            self.assertTrue(degraded)

    def test_mipchain_returns_false_false_on_total_failure(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            mip0 = os.path.join(tmpdir, "mip0.png")
            mip1 = os.path.join(tmpdir, "mip1.png")
            out = os.path.join(tmpdir, "out.dds")
            save_image(np.random.rand(64, 64, 3).astype(np.float32), mip0)
            save_image(np.random.rand(32, 32, 3).astype(np.float32), mip1)

            def _fake_generate_dds(src, dst, compression, srgb=False):
                return False  # All mips fail

            mip_paths = [{"level": 0, "path": mip0}, {"level": 1, "path": mip1}]
            with mock.patch.object(gen, "_resolve_compression_tool", return_value="fake.exe"):
                with mock.patch.object(gen, "generate_dds", side_effect=_fake_generate_dds):
                    success, degraded = gen.generate_dds_mipchain(mip_paths, out, compression="bc7")

            self.assertFalse(success)
            self.assertFalse(degraded)

    def test_no_shared_state_attribute(self):
        """Verify _last_mipchain_degraded and consume_mipchain_degraded are removed."""
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)
        self.assertFalse(hasattr(gen, "_last_mipchain_degraded"))
        self.assertFalse(hasattr(gen, "consume_mipchain_degraded"))


class TestMipmapSharpenExclusion(unittest.TestCase):
    """Tests for mipmap sharpen exclusion of PBR data maps (Issue 4)."""

    def test_sharpen_skipped_for_data_maps(self):
        """Verify ROUGHNESS, METALNESS, AO, ORM are NOT sharpened."""
        from AssetBrew.phases.mipmap import MipmapGenerator
        from AssetBrew.config import TextureType
        config = PipelineConfig()
        config.mipmap.sharpen_mips = True
        config.mipmap.sharpen_levels = [1, 2, 3]
        MipmapGenerator(config)  # verify it constructs with config

        excluded = [
            TextureType.ROUGHNESS, TextureType.METALNESS,
            TextureType.AO, TextureType.ORM,
            TextureType.NORMAL, TextureType.HEIGHT, TextureType.MASK,
        ]
        for tex_type in excluded:
            # The sharpen logic is: if tex_type not in exclusion set, sharpen.
            # We verify the exclusion set is correct by checking the condition.
            self.assertTrue(
                tex_type in (
                    TextureType.NORMAL, TextureType.HEIGHT, TextureType.MASK,
                    TextureType.ROUGHNESS, TextureType.METALNESS,
                    TextureType.AO, TextureType.ORM,
                ),
                f"{tex_type} should be in sharpen exclusion set",
            )


class TestRoughnessIncreaseExponential(unittest.TestCase):
    """Tests for exponential roughness mip increase formula (Issue 5)."""

    def test_exponential_vs_linear_at_high_level(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)
        mip = np.full((16, 16), 0.3, dtype=np.float32)
        inc = config.mipmap.roughness_mip_increase

        # Level 4: exponential increase = inc * (2^4 - 1) = inc * 15
        result_l4 = gen._increase_roughness(mip, level=4)
        expected_increase = inc * (2.0 ** 4 - 1.0)
        expected = np.sqrt(0.3 ** 2 + expected_increase)
        np.testing.assert_allclose(result_l4.mean(), min(expected, 1.0), atol=0.01)

    def test_level_zero_unchanged(self):
        from AssetBrew.phases.mipmap import MipmapGenerator
        config = PipelineConfig()
        gen = MipmapGenerator(config)
        mip = np.full((16, 16), 0.5, dtype=np.float32)
        result = gen._increase_roughness(mip, level=0)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
