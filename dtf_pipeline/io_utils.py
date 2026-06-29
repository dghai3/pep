"""Load / normalize inputs and save print-ready outputs.

Handles the messy real-world inputs a print shop receives: CMYK files, palette
('P') PNGs, grayscale, 16-bit, and photos with EXIF orientation.  All loads are
normalized to a clean ``HxWx3`` uint8 RGB array so downstream stages never have
to worry about colour mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .utils import log

# Formats we accept as inputs.
SUPPORTED_INPUT_EXT = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
}


def _cmyk_to_rgb(img: Image.Image) -> Image.Image:
    """Convert a CMYK image to sRGB.

    Uses the embedded ICC profile when present (accurate), otherwise PIL's naive
    CMYK->RGB conversion (good enough for print artwork without a profile).
    """
    icc = img.info.get("icc_profile")
    if icc:
        try:
            from io import BytesIO

            src = ImageCms.ImageCmsProfile(BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
        except Exception as exc:  # pragma: no cover - profile edge cases
            log.warning("CMYK ICC conversion failed (%s); using naive convert", exc)
    return img.convert("RGB")


def _is_meaningful_alpha(a: np.ndarray) -> bool:
    """True if an alpha channel carries real transparency (not all-opaque)."""
    return bool(a.min() < 250 and (a < 250).mean() > 0.005)


def load_image(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Load any supported image, normalized for the pipeline.

    Returns ``(rgb, alpha, meta)``:
      * ``rgb``   - ``HxWx3`` uint8, sRGB.
      * ``alpha`` - ``HxW`` float32 [0,1] **only if** the input already carries
        meaningful transparency (an existing cutout); otherwise ``None``.
      * ``meta``  - original mode/size and source DPI.

    Normalization performed:
      * EXIF orientation is applied (so phone photos are upright).
      * CMYK -> sRGB (profile-aware when possible).
      * Palette ('P'/'PA') and transparent palettes -> RGB(+alpha).
      * Grayscale / 16-bit -> 8-bit RGB.
    """
    path = Path(path)
    alpha: Optional[np.ndarray] = None
    with Image.open(path) as im:
        im.load()
        original_mode = im.mode
        source_dpi = im.info.get("dpi")
        has_palette_transparency = im.mode == "P" and "transparency" in im.info

        # 1) Respect EXIF orientation before anything else.
        im = ImageOps.exif_transpose(im)

        # 2) Colour-mode normalization (+ alpha extraction where present).
        if im.mode in ("RGBA", "LA", "PA") or has_palette_transparency:
            rgba = im.convert("RGBA")
            a = np.asarray(rgba)[:, :, 3]
            if _is_meaningful_alpha(a):
                alpha = a.astype(np.float32) / 255.0
            # Keep stored RGB (convert drops alpha without compositing).
            rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
        elif im.mode == "CMYK":
            rgb = np.asarray(_cmyk_to_rgb(im), dtype=np.uint8)
        elif im.mode == "P":
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        elif im.mode in ("I", "I;16", "F"):
            arr = np.asarray(im).astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max() * 255.0
            rgb = np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
        elif im.mode == "RGB":
            rgb = np.asarray(im, dtype=np.uint8)
        else:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    if rgb.ndim == 2:  # safety: any stray grayscale
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = np.ascontiguousarray(rgb[:, :, :3])

    meta = {
        "path": str(path),
        "original_mode": original_mode,
        "source_dpi": source_dpi,
        "native_size": (rgb.shape[1], rgb.shape[0]),  # (W, H)
        "has_existing_alpha": alpha is not None,
    }
    log.info(
        "Loaded %s  mode=%s  %dx%d%s",
        path.name, original_mode, rgb.shape[1], rgb.shape[0],
        "  (existing transparency)" if alpha is not None else "",
    )
    return rgb, alpha, meta


def save_rgba(
    rgba: np.ndarray,
    path: str | Path,
    dpi: int = 300,
    fmt: str = "png",
    tiff_compression: str = "tiff_lzw",
) -> None:
    """Save an ``HxWx4`` uint8 RGBA array, tagged at ``dpi`` DPI.

    PNG: DPI is written to the pHYs chunk (pixels-per-metre) via Pillow's ``dpi``
    save argument.  TIFF: DPI is written to the resolution tags.  Transparency is
    always preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"save_rgba expects HxWx4, got shape {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"save_rgba expects uint8, got {rgba.dtype}")

    img = Image.fromarray(rgba, mode="RGBA")
    if fmt == "png":
        # compress_level 6 = good size/speed balance; `optimize=True` is far too
        # slow on large (multi-thousand-px) print files for negligible savings.
        img.save(path, format="PNG", dpi=(dpi, dpi), compress_level=6)
    elif fmt == "tiff":
        img.save(
            path,
            format="TIFF",
            dpi=(dpi, dpi),
            compression=tiff_compression,
        )
    else:
        raise ValueError(f"Unknown output format: {fmt}")
    log.info("Saved %s  %dx%d @ %d DPI", path.name, rgba.shape[1], rgba.shape[0], dpi)


def save_rgb(rgb: np.ndarray, path: str | Path, dpi: int = 300) -> None:
    """Save an ``HxWx3`` uint8 RGB array as PNG (used for magenta QA composites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", dpi=(dpi, dpi))


def read_dpi(path: str | Path) -> Tuple[float, float] | None:
    """Read back the DPI tag of a saved file (used by the verification step)."""
    with Image.open(path) as im:
        return im.info.get("dpi")
