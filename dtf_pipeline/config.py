"""Configuration for the DTF print-prep pipeline.

Every tunable knob lives on :class:`PipelineConfig`.  The CLI in ``pipeline.py``
builds one of these from command-line flags and hands the *same* object to every
stage, so behaviour is fully reproducible from a single config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
# Edge-refinement modes
# --------------------------------------------------------------------------- #
EDGE_HARD = "hard"
EDGE_MATTE = "matte"


@dataclass
class PipelineConfig:
    """All pipeline knobs in one place.

    Attributes are grouped by stage.  Defaults are tuned for the common DTF
    case: logos / vector-style graphics that want crisp, near-binary edges.
    """

    # ----------------------------- Stage 1: segmentation ------------------- #
    rembg_model: str = "birefnet-general"
    """rembg session name.  BiRefNet variants give the best edges for graphics:
    ``birefnet-general`` (default), ``birefnet-general-lite`` (faster),
    ``birefnet-portrait``, ``birefnet-dis``, ``birefnet-hrsod`` …"""

    working_resolution: int = 1024
    """Longest-side resolution the segmentation model runs at.  BiRefNet infers
    around 1024 px; the RGB is downscaled to this size *only* to compute the
    mask, which is then upsampled back to native resolution with LANCZOS.  The
    full-resolution RGB is never destructively resized."""

    alpha_matting: bool = True
    """Enable rembg's alpha-matting refinement (wraps pymatting) for softer,
    more accurate edges straight out of segmentation."""

    respect_existing_alpha: bool = True
    """If the input is already a transparent cutout, use its alpha instead of
    re-segmenting it (still decontaminated + upscaled).  Set False to force
    background removal even on transparent inputs."""

    alpha_matting_foreground_threshold: int = 240
    alpha_matting_background_threshold: int = 10
    alpha_matting_erode_size: int = 0  # we run our own erode in the refine stage

    # ----------------------------- Stage 1: edge refinement ---------------- #
    edge_mode: str = EDGE_HARD
    """``hard`` (default) -> near-binary alpha for logos/graphics.
    ``matte`` -> keep the soft continuous alpha for photographic subjects."""

    hard_threshold: float = 0.5
    """Alpha cutoff used to binarise in the hard path."""

    morph_kernel: int = 3
    """Kernel size (px) for the morphological open/close that removes speckles
    and fills pinholes in the hard path."""

    erode_px: int = 1
    """Erode the binary mask by this many px to pull the edge off the
    background-contaminated rim (hard path).  Set 0 to disable."""

    feather_px: float = 1.0
    """Gaussian feather radius (px) applied after thresholding so the edge is
    anti-aliased rather than jagged (hard path)."""

    matte_trimap_band: int = 8
    """Width (px) of the 'unknown' band around the edge used to build a trimap
    for pymatting in the matte path."""

    # ----------------------------- Stage 1: decontamination ---------------- #
    decontaminate: bool = True
    """Recover clean foreground RGB on semi-transparent edge pixels so the rim
    carries no residual background colour.  THIS is what prevents the DTF halo.
    Applied on both edge paths, *before* upscaling."""

    decontam_radius: int = 12
    """Fallback defringe reach (px): how far clean foreground colour is bled
    outward into the edge band when pymatting is unavailable."""

    # ----------------------------- Stage 2: upscaling ---------------------- #
    upscale_model: str = "realesrgan-x4plus-anime"
    """Real-ESRGAN model.  Default is the anime/line-art variant which gives the
    crispest edges on flat graphics & logos.  Use ``realesr-general-x4v3`` or
    ``realesrgan-x4plus`` for photographic subjects (see README tradeoff)."""

    scale: int = 4
    """Integer upscale factor when no ``target_inches`` is supplied."""

    alpha_upsampler: str = "lanczos"
    """How the alpha channel is upscaled: ``lanczos`` (default, cleanest edge on
    graphics) or ``realesrgan`` (run the model on alpha-as-RGB)."""

    tile: int = 0
    """Real-ESRGAN tile size for limited VRAM (0 = no tiling).  Try 256/512 on
    an 8 GB card if you hit out-of-memory on very large inputs."""

    tile_pad: int = 10
    pre_pad: int = 0
    fp16: Optional[bool] = None
    """Half precision for Real-ESRGAN.  ``None`` = auto (on for CUDA, off for
    CPU/MPS)."""

    # ----------------------------- Stage 3: export ------------------------- #
    target_inches: Optional[Tuple[float, float]] = None
    """Target physical size (width, height) in inches.  Pixel dimensions are
    computed at ``dpi`` and the artwork is fit (aspect-preserving) inside this
    box.  When set, takes priority over ``scale``."""

    dpi: int = 300
    """Output DPI written to the file's pHYs chunk."""

    output_format: str = "png"  # "png" or "tiff"
    tiff_compression: str = "tiff_lzw"

    trim: Optional[bool] = None
    """Trim fully-transparent margins before applying ``target_inches`` so the
    physical size refers to the actual artwork.  ``None`` = auto (on when
    ``target_inches`` is given)."""

    trim_margin_px: int = 2
    """Tiny transparent margin (px, native res) kept around trimmed artwork."""

    # ----------------------------- Runtime --------------------------------- #
    device: str = "auto"  # auto / cpu / cuda / mps
    debug: bool = False
    """Write magenta-composite QA images at the post-removal and post-upscale
    stages, plus intermediate artefacts."""

    allow_fallback: bool = True
    """If the ML libraries / model weights are unavailable, fall back to
    high-quality classical methods (color segmentation, Lanczos upscale,
    dilation defringe) instead of crashing.  The real ML path is always
    preferred when present."""

    quiet: bool = False

    # --------------------------------------------------------------------- #
    def resolve_trim(self) -> bool:
        """Whether trimming should run, resolving the ``auto`` default."""
        if self.trim is None:
            return self.target_inches is not None
        return self.trim

    def target_pixels(self) -> Optional[Tuple[int, int]]:
        """Target (width, height) in pixels from ``target_inches`` * ``dpi``."""
        if self.target_inches is None:
            return None
        w_in, h_in = self.target_inches
        return (max(1, round(w_in * self.dpi)), max(1, round(h_in * self.dpi)))

    def validate(self) -> None:
        if self.edge_mode not in (EDGE_HARD, EDGE_MATTE):
            raise ValueError(f"edge_mode must be '{EDGE_HARD}' or '{EDGE_MATTE}'")
        if self.alpha_upsampler not in ("lanczos", "realesrgan"):
            raise ValueError("alpha_upsampler must be 'lanczos' or 'realesrgan'")
        if self.output_format not in ("png", "tiff"):
            raise ValueError("output_format must be 'png' or 'tiff'")
        if self.working_resolution < 64:
            raise ValueError("working_resolution too small (min 64)")
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
