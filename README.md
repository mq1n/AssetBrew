# AssetBrew - Game Asset Visual Upgrade Pipeline

Automated pipeline to upgrade low-resolution game textures into PBR-ready outputs at scale.
Built for large batch runs with checkpoint resume, timeout handling, compression tool integration,
and GPU memory safeguards.

## Features

| Phase | What it does |
|---|---|
| **0. Scan** | Classify textures by type/material and write a manifest with file hashes |
| **1. AI Upscale** | Real-ESRGAN upscale with tiling preservation, OOM recovery, adaptive tile sizing |
| **2. PBR** | Generate roughness, metalness, AO, optional gloss, de-lit albedo, and zone masks |
| **3. Normal** | Generate normal + height maps (Sobel, hybrid, optional DeepBump) |
| **4. POM** | Refine height for parallax mapping and export reference shaders |
| **5. Post-Process** | Seam repair, color consistency/grading/LUT, specular AA, detail overlay, emissive/env masks, engine presets |
| **6. Mipmap** | Per-map-type mip generation and normal renormalization |
| **7. Validate** | Structural checks, semantic plausibility/albedo checks, tiling scores, and preview renders |

## Production Notes

- PBR and normal/height synthesis from diffuse/albedo are still heuristic. Treat them as generated drafts unless validated against authored ground truth.
- Validation now supports strict production gates:
- `validation.strict_material_semantics: true` to turn semantic plausibility warnings into errors.
- `validation.fail_on_heuristic_maps: true` to fail runs when heuristic maps are generated.
- `validation.enforce_material_semantics: true` to check roughness/metalness/AO plausibility.
- `validation.enforce_plausible_albedo: true` to gate out physically implausible albedo values.
- `tiling_quality.*` writes `tiling_quality_report.json` with per-asset quality scores and review flags.
- Model auto-download is checksum-hardened by default. Unverified downloads require explicit opt-in:
- `upscale.allow_unverified_model_download: true`

---

## Prerequisites

- **Python 3.10 or newer** (3.10-3.12 recommended for GPU upscaling)
- **pip** (ships with Python)
- **Git** (to clone the repo)

> GPU upscaling (Real-ESRGAN / BasicSR) is incompatible with Python 3.13+. Use 3.10-3.12 if you need GPU acceleration.

### Verify your Python installation

```bash
python --version        # should print 3.10.x or higher
pip --version           # should print pip 23+ or higher
```

On some systems, use `python3` and `pip3` instead of `python` and `pip`.

---

## Installation (Virtual Environment)

Always install into a virtual environment to avoid polluting your system Python.

### 1. Clone and enter the repository

```bash
git clone <repo-url> AssetBrew
cd AssetBrew
```

### 2. Create and activate a virtual environment

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)` to confirm the environment is active.

### 3. Upgrade pip and install setuptools

```bash
pip install --upgrade pip setuptools
```

> **Python 3.13+:** `setuptools` is no longer bundled with the standard library. If you skip this step, editable installs will fail with `Cannot import 'setuptools.build_meta'`.

### 4. Install the package

Pick **one** of the profiles below depending on your hardware and use case.

#### CPU-only (all phases except AI upscale)

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

This gives you PBR generation, normal/height maps (Sobel & hybrid), mipmaps, validation, and compression. AI upscale is skipped at runtime when torch is absent.

#### CPU + DeepBump AI normals

```bash
pip install -e ".[cpu]" --no-build-isolation
```

Adds `onnxruntime` for the DeepBump ONNX model (AI-based normal map generation on CPU).

#### GPU (full pipeline with AI upscale)

```bash
pip install -e ".[gpu]" --no-build-isolation
```

Installs PyTorch, Real-ESRGAN, BasicSR, and `onnxruntime-gpu`. Requires an NVIDIA GPU with CUDA support. **Do not** mix `[gpu]` and `[cpu]` extras in the same environment -- the ONNX runtimes conflict.

> If PyTorch does not detect your GPU after install, you may need to install a CUDA-specific PyTorch build. See https://pytorch.org/get-started/locally/ for the correct command for your CUDA version.

#### Desktop UI

```bash
pip install -e ".[ui]" --no-build-isolation
```

Or layer it on top of GPU:

```bash
pip install -e ".[gpu,ui]" --no-build-isolation
```

#### Development (linting + tests)

```bash
pip install -e ".[dev]" --no-build-isolation
```

Or combine all extras:

```bash
pip install -e ".[gpu,ui,dev]" --no-build-isolation
```

### 5. Verify the installation

```bash
python -m AssetBrew --help
```

You should see the full CLI options. If you installed the UI extra:

```bash
python -m AssetBrew --ui
```

---

## Quick Start

```bash
# Generate a default config file
python -m AssetBrew --generate-config

# Run the full pipeline
python -m AssetBrew --input ./assets/source --output ./assets/output

# Run specific phases only
python -m AssetBrew -i ./assets/source --phases upscale,pbr,normal

# Dry run (preview what would happen)
python -m AssetBrew -i ./assets/source --dry-run

# Force full reprocess (clear checkpoint)
python -m AssetBrew -i ./assets/source --reset-checkpoint

# CPU-only run (skip GPU even if available)
python -m AssetBrew -i ./assets/source --device cpu

# Use a custom config
python -m AssetBrew --config config.yaml
```

---

## External Tools (optional)

Texture compression depends on external CLI tools that are **not** Python packages. Place them in the `bin/` directory or set their paths in `config.yaml`.

| Tool | Purpose | Source |
|---|---|---|
| `compressonatorcli` | BC1-BC7 GPU texture compression | [GPUOpen Compressonator](https://github.com/GPUOpen-Tools/compressonator) |
| `texconv.exe` | DirectXTex DDS conversion (Windows) | [DirectXTex](https://github.com/microsoft/DirectXTex) |
| `toktx` | KTX2 format support | [KTX-Software](https://github.com/KhronosGroup/KTX-Software) |
| `DeepBump-8/` | ONNX model for AI normal maps | [DeepBumb](https://github.com/HugoTini/DeepBump) |

---

## Desktop UI (PyQt)

```bash
pip install -e ".[ui]" --no-build-isolation
python -m AssetBrew --ui
```

Or via the console script:

```bash
AssetBrew-ui
```

UI capabilities:

- Runtime config editing and validation
- Phase selection and run control
- Asset browsing and filtering
- Interactive preview with split/before/after
- Map mode selector for base, albedo, roughness, metalness, AO, normal, height, ORM

---

## CLI Reference

```text
python -m AssetBrew [OPTIONS]

Options:
  --input, -i DIR       Input assets directory
  --output, -o DIR      Output directory
  --config, -c FILE     Path to config YAML
  --phases, -p PHASES   Comma-separated: upscale,pbr,normal,pom,mipmap,postprocess,validate
  --device DEVICE       auto | cuda | cuda:N | cpu
  --workers N           Max parallel workers
  --dry-run             Preview without writing files
  --reset-checkpoint    Clear checkpoint, reprocess everything
  --generate-config     Write default config.yaml and exit
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR
  --ui                  Launch the desktop UI
```

---

## Output Structure

```text
assets/output/
|-- brick_wall_diff.png
|-- brick_wall_diff_albedo.png
|-- brick_wall_diff_normal.png
|-- brick_wall_diff_roughness.png
|-- brick_wall_diff_metalness.png
|-- brick_wall_diff_ao.png
|-- brick_wall_diff_gloss.png
|-- brick_wall_diff_height.png
|-- brick_wall_diff_orm.png
|-- brick_wall_diff_emissive.png
|-- brick_wall_diff_envmask.png
|-- brick_wall_diff_zones.png
|-- _shaders/
|   |-- pom_reference.glsl
|   \-- pom_reference.hlsl
|-- validation_report.txt
|-- tiling_quality_report.json
|-- pipeline_results.json
\-- pipeline.log
```

---

## Key Config Knobs

Upscaling:

- `upscale.target_resolution`
- `upscale.hero_resolution`
- `upscale.tile_size`
- `upscale.enforce_power_of_two`
- `upscale.model_sha256.<model_name>`
- `upscale.allow_unverified_model_download`

PBR and normal generation:

- `pbr.material_roughness_defaults`
- `pbr.material_metalness_defaults`
- `pbr.delight_method`
- `pbr.material_zone_masks`
- `pbr.apply_zone_pbr_adjustments`
- `pbr.generate_gloss`
- `normal.method`
- `normal.invert_y`

Color grading and postprocess:

- `color_grading.white_balance_shift`
- `color_grading.exposure_ev`
- `color_grading.midtone_gamma`
- `color_grading.saturation`
- `color_grading.lut_path`
- `seam_repair.detect_threshold`
- `orm_packing.preset`
- `emissive.enabled`
- `reflection_mask.enabled`

Validation and quality gates:

- `validation.enforce_material_semantics`
- `validation.strict_material_semantics`
- `validation.fail_on_heuristic_maps`
- `validation.enforce_plausible_albedo`
- `validation.require_full_mipchain`
- `validation.metalness_nonmetal_max_mean`
- `validation.metalness_metal_min_mean`
- `validation.roughness_min_stddev`
- `validation.ao_min_stddev`
- `tiling_quality.warn_score`
- `tiling_quality.fail_score`

Pipeline safety:

- `phase_failure_abort_ratio`
- `phase_failure_abort_min_processed`

Mipmap generation:

- `mipmap.srgb_downsampling`

Compression:

- `compression.tool`
- `compression.tool_path`
- `compression.tool_timeout_seconds`
- `compression.compressonator_encode_with`
- `compression.compressonator_num_threads`
- `compression.compressonator_performance`
- `compression.compressonator_no_progress`
- `compression.generate_dds`
- `compression.generate_ktx2`
- `compression.generate_tga`

---

## Running Tests

```bash
pip install -e ".[dev]" --no-build-isolation
python -m pytest tests/ -v
```

To include slow compression tests:

```bash
python -m pytest tests/ -v -m slow
```

Lint:

```bash
ruff check src/ tests/
```

---

## Deactivating / Cleaning Up

When you are done, deactivate the virtual environment:

```bash
deactivate
```

To completely remove the environment and start fresh:

**Windows:**
```cmd
rmdir /s /q .venv
```

**Linux / macOS:**
```bash
rm -rf .venv
```

Then repeat from [step 2](#2-create-and-activate-a-virtual-environment) to recreate it.

---

## Extending

Each phase exposes a `process(record, ...) -> dict` style interface.
To add a phase:

1. Add config dataclass in `src/AssetBrew/config.py`.
2. Implement the processor in `src/AssetBrew/phases/`.
3. Wire it through `AssetPipeline` in `src/AssetBrew/pipeline.py`.
4. Add behavioral tests in `tests/`.

---

## Requirements and Notes

- Python 3.10+ required. 3.10-3.12 recommended for GPU support.
- GPU upscaling stack (Real-ESRGAN / BasicSR) is broken on Python 3.13+.
- CPU-only mode works for all phases; upscale is just slower.
- `[gpu]` and `[cpu]` extras must not be installed together (ONNX runtime conflict).
- Compression integration depends on external tools (`compressonator`, `texconv`, `toktx`).

## License

MIT
