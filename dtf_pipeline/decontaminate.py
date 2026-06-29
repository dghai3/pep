"""Stage 1c - colour decontamination  (THE most important step for DTF).

On a DTF film, whatever you submit is printed exactly - so any residual
background colour bled into the semi-transparent edge pixels prints as a visible
halo / outline around the artwork on the garment.  This stage recovers clean
*foreground* RGB for every partially-transparent pixel so the rim carries no
background colour.

Primary path: pymatting ``estimate_foreground_ml`` (un-multiplies the alpha to
solve for true foreground colour).  Fallback: an iterative dilation "defringe"
that bleeds known foreground colour outward into the edge band.  Both run BEFORE
upscaling so the RGB is clean everywhere under (and just beyond) the alpha.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import log, to_float01

try:
    from pymatting import estimate_foreground_ml  # type: ignore

    _PYMATTING_FG = True
except Exception:  # pragma: no cover
    estimate_foreground_ml = None  # type: ignore
    _PYMATTING_FG = False

# Cap the resolution foreground estimation runs at, then upsample the (smooth)
# foreground colour back to native.  Keeps pymatting tractable on large inputs.
_FG_RES_CAP = 2048


def decontaminate(rgb: np.ndarray, alpha: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Return clean foreground RGB as float32 [0,1], same HxW as input.

    ``rgb`` may be uint8 or float; ``alpha`` is float32 [0,1].
    """
    rgb01 = to_float01(rgb)
    if not cfg.decontaminate:
        return rgb01

    if _PYMATTING_FG:
        return _decontaminate_pymatting(rgb01, alpha, cfg)
    if not cfg.allow_fallback:
        raise RuntimeError(
            "pymatting not installed and allow_fallback=False. "
            "Install requirements.txt to enable estimate_foreground_ml."
        )
    log.warning(
        "pymatting not installed -> using dilation DEFRINGE fallback for "
        "decontamination (install requirements.txt for estimate_foreground_ml)."
    )
    return _decontaminate_defringe(rgb01, alpha, cfg)


# --------------------------------------------------------------------------- #
def _decontaminate_pymatting(
    rgb01: np.ndarray, alpha: np.ndarray, cfg: PipelineConfig
) -> np.ndarray:
    """Un-multiply via pymatting's multilevel foreground estimation."""
    native_h, native_w = alpha.shape[:2]
    scale = min(1.0, _FG_RES_CAP / float(max(native_h, native_w)))

    if scale < 1.0:
        small_rgb = cv2.resize(
            rgb01, (round(native_w * scale), round(native_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        small_alpha = cv2.resize(
            alpha, (small_rgb.shape[1], small_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        small_rgb, small_alpha = rgb01, alpha

    try:
        fg = estimate_foreground_ml(
            small_rgb.astype(np.float64), small_alpha.astype(np.float64)
        ).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        log.warning("estimate_foreground_ml failed (%s); using defringe", exc)
        return _decontaminate_defringe(rgb01, alpha, cfg)

    if fg.shape[:2] != (native_h, native_w):
        fg = cv2.resize(fg, (native_w, native_h), interpolation=cv2.INTER_LANCZOS4)

    # Keep original colour where fully opaque; trust the estimate on the rim.
    log.info("Decontaminate: pymatting estimate_foreground_ml")
    return np.clip(fg, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
def _decontaminate_defringe(
    rgb01: np.ndarray, alpha: np.ndarray, cfg: PipelineConfig
) -> np.ndarray:
    """Iterative dilation defringe.

    Treat near-opaque pixels as 'known clean foreground' and repeatedly bleed
    their colour outward into the edge band, so semi-transparent / just-outside
    pixels take foreground colour instead of background colour.
    """
    known = (alpha >= 0.9).astype(np.uint8)
    if known.sum() == 0:
        known = (alpha >= 0.5).astype(np.uint8)
    if known.sum() == 0:
        return rgb01  # nothing to anchor on

    filled = rgb01.copy()
    cur = known.copy()
    kernel = np.ones((3, 3), np.uint8)

    for _ in range(max(1, cfg.decontam_radius)):
        dil = cv2.dilate(cur, kernel)
        newly = (dil > 0) & (cur == 0)
        if not newly.any():
            break
        wk = cur.astype(np.float32)
        num = cv2.blur(filled * wk[:, :, None], (3, 3))
        den = cv2.blur(wk, (3, 3))[:, :, None] + 1e-6
        avg = num / den
        filled[newly] = avg[newly]
        cur[newly] = 1

    log.info("Decontaminate: dilation defringe (radius=%dpx)", cfg.decontam_radius)
    return np.clip(filled, 0.0, 1.0).astype(np.float32)
