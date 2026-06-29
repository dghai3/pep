"""DTF print-prep pipeline.

Background removal (BiRefNet/rembg) -> color decontamination -> upscaling
(Real-ESRGAN) -> 300 DPI print-ready RGBA export.

Designed for prepping logos / graphics for DTF film printing, where any residual
background-colour halo on the edges prints as a visible outline on the garment.
"""

from .config import PipelineConfig, EDGE_HARD, EDGE_MATTE
from .pipeline import Pipeline

__all__ = ["PipelineConfig", "Pipeline", "EDGE_HARD", "EDGE_MATTE"]

__version__ = "1.0.0"
