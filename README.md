# DTF Print-Prep Pipeline

Local, end-to-end prep for DTF (direct-to-film) print artwork:

**background removal (BiRefNet) → edge color decontamination → upscaling (Real-ESRGAN) → 300 DPI print-ready RGBA PNG.**

Built for logos and graphics that get dropped into a third-party gang-sheet
builder. It does **not** do gang-sheet / nesting / layout — that's handled
elsewhere. It produces one clean, correctly-sized, transparent file per input.

> **Why the decontamination step matters for DTF:** the film prints *exactly*
> what you submit, so any background color left in the soft edge pixels prints as
> a visible halo/outline around the art on the garment. This pipeline recovers
> clean foreground color on every semi-transparent edge pixel so the rim carries
> zero background tint. Clean, halo-free edges are the #1 quality bar.

---

## What it does (3 stages)

1. **Background removal** — BiRefNet via [`rembg`](https://github.com/danielgatis/rembg)
   with **alpha matting enabled**. The model infers at a fixed working resolution
   (~1024 px); only the *mask* makes that round trip — it's computed on a
   downscaled copy and upsampled back to native resolution with LANCZOS, so your
   full-resolution RGB is never destructively downscaled. Edge handling is
   swappable: **hard** (near-binary, default — best for logos) or **matte** (soft
   alpha for photographic subjects). Then **color decontamination** (pymatting
   `estimate_foreground_ml`) cleans the rim. Output of this stage is a
   full-resolution RGBA at native dimensions.

2. **Upscaling** — Real-ESRGAN. Because Real-ESRGAN doesn't handle RGBA natively,
   the channels are split: RGB is upscaled by the model, the alpha channel is
   upscaled separately (LANCZOS by default — cleanest edge on graphics — or by
   the model), then recombined. Decontamination runs *before* upscaling so the
   RGB is clean everywhere under the alpha.

3. **300 DPI export** — DPI is just metadata mapping pixels to inches
   (pixels = inches × 300). Give a target physical size with `--target-inches`
   and the artwork is sized (aspect preserved) to fit that box at 300 DPI and the
   file is tagged at 300 DPI (PNG pHYs chunk). Transparency is preserved. TIFF
   output is also available.

---

## Requirements

- **Python 3.10 or 3.11** (recommended; `basicsr`/`realesrgan` can be fiddly on 3.12).
- A **GPU is strongly recommended** — BiRefNet and Real-ESRGAN are slow on CPU
  (tens of seconds to minutes per image). The code auto-detects CUDA / Apple MPS
  and falls back to CPU with a warning.
- ~2–3 GB free disk for model weights, downloaded automatically on first run
  (needs internet that one time).

---

## Install

```bash
# 1) create + activate a virtual environment
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 2) install PyTorch for YOUR hardware (pick one)

#  Windows / Linux + NVIDIA GPU (RTX 3070), CUDA 12.1:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

#  Apple Silicon Mac (M1/M2/M3/M4) — MPS acceleration:
pip install torch==2.3.1 torchvision==0.18.1

#  CPU-only:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# 3) install the rest
pip install -r requirements.txt
```

### GPU acceleration for background removal (optional but recommended)

`rembg` runs BiRefNet through **onnxruntime**. The default `onnxruntime` package
uses CPU (and CoreML on Apple Silicon). For NVIDIA GPU segmentation, swap it:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.18.1
```

On your **RTX 3070** this gives GPU-accelerated segmentation *and* upscaling. On
**Apple Silicon**, keep the default `onnxruntime` (CoreML) and Real-ESRGAN will
use MPS automatically.

> First run downloads BiRefNet weights (to the rembg cache) and the Real-ESRGAN
> weights (to `~/.cache/dtf_pipeline/weights`). Subsequent runs are offline.

---

## Usage

```bash
# Single image -> exact output path, sized to 10x12 inches at 300 DPI
python pipeline.py input.png -o output.png --target-inches "10,12"

# Batch a folder -> output folder
python pipeline.py ./inputs/ -o ./outputs/ --target-inches "10,12"

# Photographic subject (soft edges) instead of a logo
python pipeline.py photo.png -o photo_print.png --edge-mode matte --upscale-model realesr-general-x4v3 --target-inches "8,10"

# No target size: just upscale 4x and tag 300 DPI
python pipeline.py logo.png -o logo_print.png --scale 4

# Verify edge quality on magenta (writes extra *_debug_*_magenta.png files)
python pipeline.py logo.png -o logo_print.png --target-inches "10,12" --debug

# List upscale models
python pipeline.py --list-models
```

### Try it on the included synthetic tests

```bash
python make_test_images.py                       # writes 4 images into ./inputs/
python pipeline.py ./inputs/ -o ./outputs/ --target-inches "10,12" --debug
```

---

## All flags

| Flag | Default | What it does |
|------|---------|--------------|
| `input` | — | Input image file, or a directory for batch mode. |
| `-o, --output` | — | Output file (single) or directory (batch). |
| `--edge-mode {hard,matte}` | `hard` | `hard` = near-binary edges for logos/graphics; `matte` = soft continuous alpha for photographic subjects. |
| `--rembg-model NAME` | `birefnet-general` | rembg session. BiRefNet variants: `birefnet-general`, `birefnet-general-lite`, `birefnet-portrait`, `birefnet-dis`, `birefnet-hrsod`, … |
| `--working-resolution N` | `1024` | Resolution the segmentation model runs at. The mask is upsampled back to native with LANCZOS; the RGB is never downscaled. |
| `--no-alpha-matting` | (on) | Disable rembg's alpha-matting edge refinement. |
| `--ignore-existing-alpha` | (off) | By default, an input that is already a transparent cutout is respected (its alpha is kept, then decontaminated + upscaled) rather than re-segmented. This flag forces background removal anyway. |
| `--no-decontaminate` | (on) | Disable edge color decontamination. **Not recommended for DTF** — this is the anti-halo step. |
| `--upscale-model NAME` | `realesrgan-x4plus-anime` | Real-ESRGAN model — see tradeoff below. |
| `--scale N` | `4` | Integer upscale factor, used only when `--target-inches` is not given. |
| `--alpha-upsampler {lanczos,realesrgan}` | `lanczos` | How the alpha channel is upscaled. LANCZOS keeps the cleanest edge on graphics. |
| `--tile N` | `0` | Real-ESRGAN tile size for limited VRAM (e.g. `512`/`256`). `0` = no tiling. |
| `--target-inches "W,H"` | — | Target physical size in inches, e.g. `"10,12"`. Computes pixels at `--dpi`, fits the artwork (aspect preserved) inside that box, tags the file. Takes priority over `--scale`. |
| `--dpi N` | `300` | DPI written to the file. |
| `--format {png,tiff}` | `png` | Output format. Both preserve transparency and DPI. |
| `--trim` / `--no-trim` | auto | Trim transparent margins before sizing so the physical size refers to the artwork itself. Auto = on when `--target-inches` is set. |
| `--device {auto,cpu,cuda,mps}` | `auto` | Compute device. Auto-detects NVIDIA CUDA or Apple MPS. |
| `--no-fallback` | (off) | Error out instead of using classical fallbacks if the ML libraries/weights are missing. |
| `--debug` | (off) | Write magenta QA composites at the post-removal and post-upscale stages (plus the stage-1 RGBA cutout). |
| `--quiet` | (off) | Reduce logging. |
| `--list-models` | — | List upscale models and exit. |

---

## Upscale-model tradeoff

| Model | Best for | Edge character |
|-------|----------|----------------|
| `realesrgan-x4plus-anime` *(default)* | Logos, flat graphics, line art | **Crispest, hardest edges.** Built for anime/line art, which is exactly the flat-color, hard-edge regime logos live in. |
| `realesr-general-x4v3` | Mixed / mildly photographic | Balanced; good general-purpose detail without over-sharpening. |
| `realesrgan-x4plus` | Photographic subjects | Best texture/detail reconstruction on photos; softer, more natural edges. |
| `realesrgan-x2plus` | When you only need 2× | General photo model at 2× netscale. |
| `realesr-animevideov3` | Fast line-art | Lighter/faster anime model. |

Rule of thumb: **graphics → `realesrgan-x4plus-anime`** (default) for the
sharpest logo edges; **photos → `realesrgan-x4plus`** (pair with
`--edge-mode matte`).

---

## How sizing works (`--target-inches`)

`--target-inches "10,12"` means "fit the artwork within 10″ × 12″ at 300 DPI"
= within 3000 × 3600 px. The art is scaled **aspect-preserving** (a logo is never
stretched), so a roughly-square logo comes out around 3000 × 3000 px (≈ 10″ ×
10″) and the file is tagged 300 DPI. Transparent margins are trimmed first (by
default) so the inches refer to the actual art, not empty canvas. The upscale
factor is chosen so Real-ESRGAN reaches at least the required pixels, then the
result is resized to the exact size.

Without `--target-inches`, the image is upscaled by `--scale` and simply tagged
at 300 DPI.

---

## QA: verify edges on magenta (`--debug`)

Halos and jaggies are invisible on white but obvious on saturated magenta. With
`--debug`, for each image the pipeline writes:

- `*_debug_1_removed_magenta.png` — cutout composited on magenta **after background removal**
- `*_debug_1_removed_rgba.png` — the stage-1 RGBA cutout
- `*_debug_2_upscaled_magenta.png` — composited on magenta **after upscaling**

Compare the two magenta images to confirm the clean edge **survives upscaling**.
A correctly decontaminated edge shows the art color fading straight into magenta
with no bright rim of the original background color.

---

## Project structure

```
pipeline.py                 # CLI entry point
make_test_images.py         # generate 4 synthetic test images into ./inputs/
requirements.txt
dtf_pipeline/
    config.py               # PipelineConfig dataclass — every knob
    io_utils.py             # load/normalize (CMYK, palette, EXIF, 16-bit) + save w/ DPI
    segment.py              # BiRefNet via rembg (+ classical fallback)
    refine.py               # hard + matte edge refinement
    decontaminate.py        # pymatting estimate_foreground_ml (+ defringe fallback)
    upscale.py              # Real-ESRGAN, alpha-aware split-channel (+ Lanczos fallback)
    export.py               # target-inches sizing + 300 DPI PNG/TIFF
    qa.py                   # magenta composite
    pipeline.py             # orchestrator (models load once; single + batch)
    utils.py                # logging, device detection, helpers
```

---

## Troubleshooting

- **"No GPU detected → running on CPU"** — expected without CUDA/MPS. It still
  works, just slowly. On the RTX 3070, ensure the CUDA torch wheel installed
  (`python -c "import torch; print(torch.cuda.is_available())"` → `True`) and
  install `onnxruntime-gpu` for GPU segmentation.
- **`ModuleNotFoundError: torchvision.transforms.functional_tensor`** — handled
  automatically; the code installs a compatibility shim so `basicsr` imports on
  modern torchvision. No action needed.
- **CUDA out of memory on huge inputs** — add `--tile 512` (or `256`). The 3070's
  8 GB is plenty for typical logos without tiling.
- **First run is slow / needs internet** — it's downloading model weights once.
- **Classical fallback warnings** — if `rembg`/`pymatting`/`realesrgan` aren't
  installed, the pipeline still runs using classical methods (color segmentation,
  Lanczos upscale, dilation defringe) so you always get output. Install
  `requirements.txt` for full ML quality; the ML path is used automatically when
  available. Use `--no-fallback` to make missing ML a hard error instead.
