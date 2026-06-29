"""Stage 1b - edge refinement.

Two swappable paths:

* ``hard`` (default, for logos / graphics): produce a near-binary alpha.
  threshold -> morphological open/close -> optional erode off the contaminated
  rim -> ~1px feather for clean anti-aliasing.

* ``matte`` (for photographic / soft subjects): keep the soft, continuous
  alpha.  When pymatting is available the edge is refined with a trimap-based
  closed-form matte; otherwise the soft alpha is passed through.  No
  thresholding is ever applied on this path.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import EDGE_HARD, EDGE_MATTE, PipelineConfig
from .utils import log, to_float01

try:
    from pymatting import estimate_alpha_cf  # type: ignore

    _PYMATTING_ALPHA = True
except Exception:  # pragma: no cover
    estimate_alpha_cf = None  # type: ignore
    _PYMATTING_ALPHA = False


def refine_alpha(rgb: np.ndarray, alpha: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Dispatch to the configured edge path. ``alpha`` is float32 [0,1]."""
    if cfg.edge_mode == EDGE_HARD:
        return _refine_hard(alpha, cfg)
    if cfg.edge_mode == EDGE_MATTE:
        return _refine_matte(rgb, alpha, cfg)
    raise ValueError(f"Unknown edge_mode: {cfg.edge_mode}")


# --------------------------------------------------------------------------- #
def _refine_hard(alpha: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Near-binary alpha for crisp graphics edges."""
    # 1) Threshold to binary.
    binary = (alpha >= cfg.hard_threshold).astype(np.uint8) * 255

    # 2) Morphological open (kill speckles) then close (fill pinholes).
    if cfg.morph_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    # 3) Optional erode to pull the edge in off the background-contaminated rim.
    if cfg.erode_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cfg.erode_px + 1, 2 * cfg.erode_px + 1)
        )
        binary = cv2.erode(binary, ek, iterations=1)

    # 4) Feather ~1px so the edge is anti-aliased, not jagged.
    out = binary.astype(np.float32) / 255.0
    if cfg.feather_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(cfg.feather_px))
    log.info(
        "Edge refine: HARD (thr=%.2f, morph=%d, erode=%dpx, feather=%.1fpx)",
        cfg.hard_threshold, cfg.morph_kernel, cfg.erode_px, cfg.feather_px,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
def _refine_matte(rgb: np.ndarray, alpha: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Keep soft alpha; optionally sharpen the matte with pymatting."""
    if not _PYMATTING_ALPHA:
        log.info("Edge refine: MATTE (soft passthrough; pymatting not installed)")
        # Gentle edge-preserving smooth so the soft alpha isn't noisy.
        return np.clip(
            cv2.bilateralFilter(alpha.astype(np.float32), 5, 0.1, 3), 0.0, 1.0
        )

    # Build a trimap from the soft alpha and refine at a capped resolution
    # (closed-form matting solves a sparse linear system; keep it tractable),
    # then upsample the refined alpha back to native with LANCZOS.
    native_h, native_w = alpha.shape[:2]
    cap = cfg.working_resolution
    scale = min(1.0, cap / float(max(native_h, native_w)))
    if scale < 1.0:
        small_rgb = cv2.resize(
            rgb, (round(native_w * scale), round(native_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        small_alpha = cv2.resize(
            alpha, (small_rgb.shape[1], small_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        small_rgb, small_alpha = rgb, alpha

    band = cfg.matte_trimap_band
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    binary = (small_alpha >= 0.5).astype(np.uint8)
    sure_fg = cv2.erode(binary, k, iterations=1)
    sure_bg = cv2.erode(1 - binary, k, iterations=1)

    trimap = np.full(small_alpha.shape, 0.5, np.float64)
    trimap[sure_fg > 0] = 1.0
    trimap[sure_bg > 0] = 0.0

    try:
        refined = estimate_alpha_cf(
            to_float01(small_rgb).astype(np.float64), trimap
        ).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        log.warning("pymatting matte failed (%s); keeping soft alpha", exc)
        refined = small_alpha.astype(np.float32)

    if refined.shape[:2] != (native_h, native_w):
        refined = cv2.resize(
            refined, (native_w, native_h), interpolation=cv2.INTER_LANCZOS4
        )
    log.info("Edge refine: MATTE (pymatting closed-form, trimap band=%dpx)", band)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)
