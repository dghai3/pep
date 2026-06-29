"""Built-in QA: composite RGBA onto saturated magenta.

Fringe, halos and jaggies are invisible against white but glaringly obvious on
saturated magenta (#ff00ff).  Compositing the cutout onto magenta at both the
post-removal and post-upscale stages is how edge quality is verified - and how
you confirm the clean edge survives upscaling.
"""

from __future__ import annotations

import numpy as np

MAGENTA = np.array([255, 0, 255], dtype=np.float32)


def magenta_composite(rgba: np.ndarray, bg: np.ndarray = MAGENTA) -> np.ndarray:
    """Alpha-composite an HxWx4 uint8 image over a solid colour. Returns HxWx3."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"magenta_composite expects HxWx4, got {rgba.shape}")
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    out = rgb * alpha + bg.reshape(1, 1, 3) * (1.0 - alpha)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
