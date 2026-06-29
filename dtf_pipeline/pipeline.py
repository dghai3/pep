"""Pipeline orchestrator.

Wires the stages together and, importantly, loads the segmentation and upscaling
models **once** so batch runs don't pay the load cost per image.

    Stage 1  load/normalize -> segment -> refine edge -> decontaminate -> RGBA
    Stage 2  alpha-aware upscale (RGB + alpha split, recombine)
    Stage 3  fit to target inches + export at 300 DPI
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import PipelineConfig
from .decontaminate import decontaminate
from .export import export, finalize, plan_output, trim_to_content
from .io_utils import SUPPORTED_INPUT_EXT, load_image, save_rgb
from .qa import magenta_composite
from .refine import refine_alpha
from .segment import Segmenter
from .upscale import Upscaler
from .utils import detect_device, log, timed, to_uint8


class Pipeline:
    """End-to-end DTF print-prep pipeline. Construct once, reuse for many files."""

    def __init__(self, cfg: PipelineConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = detect_device(cfg.device)
        # Models load once here (not per image).
        self.segmenter = Segmenter(cfg, self.device)
        self.upscaler = Upscaler(cfg, self.device)

    # ------------------------------------------------------------------ #
    def process(self, in_path: str | Path, out_path: str | Path) -> Dict:
        """Run the full pipeline on a single image. Returns an info dict."""
        cfg = self.cfg
        in_path, out_path = Path(in_path), Path(out_path)
        log.info("=== %s ===", in_path.name)

        # --- Stage 1: load / normalize ---
        rgb, existing_alpha, meta = load_image(in_path)

        used_existing_alpha = existing_alpha is not None and cfg.respect_existing_alpha
        if used_existing_alpha:
            # Already a cut-out: keep its matte, skip segmentation + refine.
            log.info(
                "Input already transparent -> respecting existing alpha "
                "(skipping segmentation/refine; still decontaminate + upscale)"
            )
            alpha = existing_alpha
        else:
            # --- Stage 1: segment ---
            with timed("segment"):
                alpha = self.segmenter.mask(rgb)
            # --- Stage 1: edge refine ---
            alpha = refine_alpha(rgb, alpha, cfg)

        # --- Stage 1: decontaminate (clean foreground RGB on the rim) ---
        with timed("decontaminate"):
            clean_rgb01 = decontaminate(rgb, alpha, cfg)

        rgba_native = np.dstack([to_uint8(clean_rgb01), to_uint8(alpha)])

        # --- Optional trim so target-inches refers to the artwork itself ---
        if cfg.resolve_trim():
            rgba_native = trim_to_content(rgba_native, cfg.trim_margin_px)

        debug_paths: List[str] = []
        if cfg.debug:
            debug_paths += self._write_debug(out_path, "1_removed", rgba_native)

        # --- Stage 2: upscale (alpha-aware) ---
        nh, nw = rgba_native.shape[:2]
        outscale, final_size = plan_output(nw, nh, cfg)
        with timed("upscale"):
            rgb_up, alpha_up = self.upscaler.upscale_rgba(
                rgba_native[:, :, :3], rgba_native[:, :, 3], outscale
            )
        del rgba_native, clean_rgb01
        rgba_up = np.dstack([rgb_up, alpha_up])
        del rgb_up, alpha_up

        # --- Stage 3: fit to exact target size + export at DPI ---
        rgba_final = finalize(rgba_up, final_size)
        del rgba_up
        if cfg.debug:
            debug_paths += self._write_debug(out_path, "2_upscaled", rgba_final)

        export(rgba_final, out_path, cfg)

        out_w, out_h = rgba_final.shape[1], rgba_final.shape[0]
        del rgba_final, rgb

        return {
            "input": str(in_path),
            "output": str(out_path),
            "native_size": (nw, nh),
            "output_size": (out_w, out_h),
            "physical_inches": (
                round(out_w / cfg.dpi, 3),
                round(out_h / cfg.dpi, 3),
            ),
            "dpi": cfg.dpi,
            "used_existing_alpha": used_existing_alpha,
            "segment_ml": self.segmenter.using_ml,
            "upscale_ml": self.upscaler.using_ml,
            "outscale": round(outscale, 3),
            "debug": debug_paths,
        }

    # ------------------------------------------------------------------ #
    def process_batch(self, in_dir: str | Path, out_dir: str | Path) -> List[Dict]:
        """Process every supported image in ``in_dir`` -> ``out_dir``."""
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in in_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_INPUT_EXT
        )
        if not files:
            log.warning("No supported images found in %s", in_dir)
            return []

        ext = ".tiff" if self.cfg.output_format == "tiff" else ".png"
        results = []
        log.info("Batch: %d image(s)", len(files))
        for i, f in enumerate(files, 1):
            log.info("[%d/%d] %s", i, len(files), f.name)
            out_path = out_dir / f"{f.stem}_print{ext}"
            try:
                results.append(self.process(f, out_path))
            except Exception as exc:  # keep going on a bad file
                log.error("Failed on %s: %s", f.name, exc)
                results.append({"input": str(f), "error": str(exc)})
            finally:
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------ #
    def _write_debug(self, out_path: Path, tag: str, rgba: np.ndarray) -> List[str]:
        """Write the magenta QA composite (and a stage-1 cutout for tag 1)."""
        stem = out_path.with_suffix("")
        written: List[str] = []

        mag_path = Path(f"{stem}_debug_{tag}_magenta.png")
        save_rgb(magenta_composite(rgba), mag_path, dpi=self.cfg.dpi)
        written.append(str(mag_path))

        if tag.startswith("1"):
            from .io_utils import save_rgba

            cut_path = Path(f"{stem}_debug_{tag}_rgba.png")
            save_rgba(rgba, cut_path, dpi=self.cfg.dpi)
            written.append(str(cut_path))
        return written
