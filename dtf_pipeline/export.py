"""Stage 3 - print-ready sizing & export.

DPI is just metadata mapping pixels to physical inches (pixels = inches x DPI).
So a target physical size implies a pixel size at 300 DPI.  We:

  * (optionally) trim transparent margins so the physical size refers to the
    actual artwork, not empty canvas;
  * fit the artwork (aspect-preserving - never distort a logo) inside the target
    box and compute the exact output pixel size;
  * choose the Real-ESRGAN scale factor so the upscale reaches at least that many
    pixels, then resize to the exact size;
  * save as RGBA PNG (or TIFF) tagged at 300 DPI, transparency preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import save_rgba
from .utils import alpha_bbox, log


def fit_inside(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    """Largest (w', h') with the same aspect ratio that fits in (max_w, max_h)."""
    scale = min(max_w / float(w), max_h / float(h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def plan_output(
    native_w: int, native_h: int, cfg: PipelineConfig
) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Return ``(outscale, final_size)`` for the upscale + export stages.

    * With ``target_inches``: final size = artwork fit inside the target box at
      ``dpi``; outscale reaches at least that size (clamped to >= 1).
    * Otherwise: outscale = integer ``scale``; final size is left to the
      upscaler (no resize-to-exact).
    """
    target = cfg.target_pixels()
    if target is not None:
        tw, th = target
        fit_w, fit_h = fit_inside(native_w, native_h, tw, th)
        outscale = max(fit_w / native_w, fit_h / native_h, 1.0)
        return outscale, (fit_w, fit_h)
    return float(cfg.scale), None


def trim_to_content(rgba: np.ndarray, margin: int) -> np.ndarray:
    """Crop away fully-transparent borders, keeping a small margin."""
    x0, y0, x1, y1 = alpha_bbox(rgba[:, :, 3], threshold=0)
    h, w = rgba.shape[:2]
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return rgba
    log.info("Trimmed to content: %dx%d -> %dx%d", w, h, x1 - x0, y1 - y0)
    return np.ascontiguousarray(rgba[y0:y1, x0:x1])


def finalize(
    rgba_up: np.ndarray, final_size: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Resize to the exact target pixel size (if any). RGB & alpha both LANCZOS."""
    if final_size is None:
        return rgba_up
    fw, fh = final_size
    if (rgba_up.shape[1], rgba_up.shape[0]) == (fw, fh):
        return rgba_up
    rgb = cv2.resize(rgba_up[:, :, :3], (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    alpha = cv2.resize(rgba_up[:, :, 3], (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    return np.dstack([rgb, alpha])


def export(
    rgba: np.ndarray, out_path: str | Path, cfg: PipelineConfig
) -> np.ndarray:
    """Save the final RGBA at ``cfg.dpi`` and log the physical size."""
    save_rgba(
        rgba, out_path, dpi=cfg.dpi, fmt=cfg.output_format,
        tiff_compression=cfg.tiff_compression,
    )
    w_in = rgba.shape[1] / cfg.dpi
    h_in = rgba.shape[0] / cfg.dpi
    log.info(
        "Final: %dx%d px = %.2f x %.2f in @ %d DPI",
        rgba.shape[1], rgba.shape[0], w_in, h_in, cfg.dpi,
    )
    return rgba
