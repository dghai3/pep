#!/usr/bin/env python3
"""DTF print-prep pipeline - command-line interface.

Background removal (BiRefNet) -> color decontamination -> upscaling
(Real-ESRGAN) -> 300 DPI print-ready RGBA PNG.

Single image:
    python pipeline.py input.png -o output.png --target-inches "10,12"

Batch (directory in, directory out):
    python pipeline.py ./inputs/ -o ./outputs/ --target-inches "10,12"

Run ``python pipeline.py --list-models`` to see upscale model choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

from dtf_pipeline.config import EDGE_HARD, EDGE_MATTE, PipelineConfig
from dtf_pipeline.pipeline import Pipeline
from dtf_pipeline.upscale import DEFAULT_MODEL, available_models
from dtf_pipeline.utils import get_logger

IMG_EXT = {".png", ".tif", ".tiff"}


def parse_target_inches(value: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse ``"W,H"`` (e.g. ``"10,12"``) into a ``(float, float)`` tuple."""
    if value is None:
        return None
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            'target-inches must be "W,H", e.g. "10,12"'
        )
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid target-inches: {exc}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("target-inches values must be > 0")
    return (w, h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Prep artwork for DTF printing: remove bg -> upscale -> "
                    "300 DPI print-ready PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="Input image file OR directory (batch).")
    p.add_argument("-o", "--output", help="Output file (single) or directory (batch).")

    # Edge / segmentation
    p.add_argument("--edge-mode", choices=[EDGE_HARD, EDGE_MATTE], default=EDGE_HARD,
                   help="hard = near-binary edges for logos/graphics; "
                        "matte = soft alpha for photographic subjects.")
    p.add_argument("--rembg-model", default="birefnet-general",
                   help="rembg session name (BiRefNet variants recommended).")
    p.add_argument("--working-resolution", type=int, default=1024,
                   help="Resolution the segmentation model runs at (mask only).")
    p.add_argument("--no-alpha-matting", dest="alpha_matting", action="store_false",
                   help="Disable rembg alpha matting.")
    p.add_argument("--ignore-existing-alpha", dest="respect_existing_alpha",
                   action="store_false",
                   help="Re-segment even if the input is already a transparent "
                        "cutout (default: respect existing transparency).")

    # Decontamination
    p.add_argument("--no-decontaminate", dest="decontaminate", action="store_false",
                   help="Disable edge colour decontamination (NOT recommended for "
                        "DTF - this is what prevents the halo).")

    # Upscaling
    p.add_argument("--upscale-model", default=DEFAULT_MODEL,
                   choices=available_models(),
                   help="Real-ESRGAN model (see README for the tradeoff).")
    p.add_argument("--scale", type=int, default=4,
                   help="Integer upscale factor when --target-inches is not set.")
    p.add_argument("--alpha-upsampler", choices=["lanczos", "realesrgan"],
                   default="lanczos",
                   help="How to upscale the alpha channel.")
    p.add_argument("--tile", type=int, default=0,
                   help="Real-ESRGAN tile size for low VRAM (0 = off).")

    # Export
    p.add_argument("--target-inches", type=parse_target_inches, default=None,
                   metavar='"W,H"',
                   help='Target physical size in inches, e.g. "10,12". '
                        "Computes pixels at --dpi and fits the artwork (aspect "
                        "preserved) inside that box.")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI tag.")
    p.add_argument("--format", choices=["png", "tiff"], default="png",
                   dest="output_format", help="Output file format.")
    p.add_argument("--no-trim", dest="trim", action="store_const", const=False,
                   default=None,
                   help="Do not trim transparent margins before sizing.")
    p.add_argument("--trim", dest="trim", action="store_const", const=True,
                   help="Force trimming transparent margins before sizing.")

    # Runtime
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="Compute device (auto-detects GPU).")
    p.add_argument("--no-fallback", dest="allow_fallback", action="store_false",
                   help="Error out instead of using classical fallbacks when the "
                        "ML libraries/weights are missing.")
    p.add_argument("--debug", action="store_true",
                   help="Write magenta QA composites (post-removal & post-upscale).")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")
    p.add_argument("--list-models", action="store_true",
                   help="List available upscale models and exit.")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        rembg_model=args.rembg_model,
        working_resolution=args.working_resolution,
        alpha_matting=args.alpha_matting,
        respect_existing_alpha=args.respect_existing_alpha,
        edge_mode=args.edge_mode,
        decontaminate=args.decontaminate,
        upscale_model=args.upscale_model,
        scale=args.scale,
        alpha_upsampler=args.alpha_upsampler,
        tile=args.tile,
        target_inches=args.target_inches,
        dpi=args.dpi,
        output_format=args.output_format,
        trim=args.trim,
        device=args.device,
        debug=args.debug,
        allow_fallback=args.allow_fallback,
        quiet=args.quiet,
    )


def _default_single_output(inp: Path, fmt: str) -> Path:
    ext = ".tiff" if fmt == "tiff" else ".png"
    return inp.with_name(f"{inp.stem}_print{ext}")


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger(args.quiet)

    if args.list_models:
        from dtf_pipeline.upscale import _MODEL_REGISTRY

        print("Available upscale models:")
        for name, spec in _MODEL_REGISTRY.items():
            star = "  (default)" if name == DEFAULT_MODEL else ""
            print(f"  {name:24} {spec['note']}{star}")
        return 0

    if not args.input:
        build_parser().error("input is required")
    inp = Path(args.input)
    if not inp.exists():
        build_parser().error(f"input does not exist: {inp}")

    cfg = config_from_args(args)
    try:
        cfg.validate()
    except ValueError as exc:
        build_parser().error(str(exc))

    pipe = Pipeline(cfg)

    if inp.is_dir():
        out_dir = Path(args.output) if args.output else inp / "dtf_output"
        results = pipe.process_batch(inp, out_dir)
        ok = [r for r in results if "error" not in r]
        log.info("Batch done: %d ok, %d failed", len(ok), len(results) - len(ok))
        print(json.dumps(results, indent=2))
        return 0 if len(ok) == len(results) else 1

    # single file
    if args.output:
        out = Path(args.output)
        if out.is_dir() or out.suffix.lower() not in IMG_EXT:
            out = out / _default_single_output(inp, cfg.output_format).name
    else:
        out = _default_single_output(inp, cfg.output_format)

    info = pipe.process(inp, out)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
