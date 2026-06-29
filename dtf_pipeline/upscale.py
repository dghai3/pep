"""Stage 2 - upscaling with Real-ESRGAN (alpha-aware).

Real-ESRGAN does not handle a 4-channel RGBA image well, so we split the
channels: the RGB is upscaled by the model, the ALPHA channel is upscaled
separately (LANCZOS by default - cleanest edge on graphics - or by the model),
then the two are recombined at the upscaled resolution.  Decontamination has
already run, so the RGB is clean everywhere under the alpha before it is scaled.

If torch / realesrgan / the weights are unavailable, a high-quality LANCZOS
upscale is used instead so the pipeline still runs end-to-end.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import log

WEIGHTS_DIR = Path.home() / ".cache" / "dtf_pipeline" / "weights"
_GH = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# Friendly name -> weight URL + arch spec.  Arch is built lazily (needs torch).
_MODEL_REGISTRY = {
    "realesrgan-x4plus": {
        "url": f"{_GH}/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "netscale": 4,
        "arch": "rrdb", "num_block": 23,
        "note": "general photo model",
    },
    "realesrgan-x4plus-anime": {
        "url": f"{_GH}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "netscale": 4,
        "arch": "rrdb", "num_block": 6,
        "note": "anime/line-art - crispest edges on flat graphics & logos",
    },
    "realesrgan-x2plus": {
        "url": f"{_GH}/v0.2.1/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "netscale": 2,
        "arch": "rrdb", "num_block": 23,
        "note": "general photo model, 2x",
    },
    "realesr-general-x4v3": {
        "url": f"{_GH}/v0.2.5.0/realesr-general-x4v3.pth",
        "file": "realesr-general-x4v3.pth",
        "netscale": 4,
        "arch": "srvgg", "num_conv": 32,
        "note": "general-purpose, good balance for mixed/photographic subjects",
    },
    "realesr-animevideov3": {
        "url": f"{_GH}/v0.2.5.0/realesr-animevideov3.pth",
        "file": "realesr-animevideov3.pth",
        "netscale": 4,
        "arch": "srvgg", "num_conv": 16,
        "note": "fast anime model",
    },
}

DEFAULT_MODEL = "realesrgan-x4plus-anime"


def available_models() -> list[str]:
    return list(_MODEL_REGISTRY)


def _install_torchvision_shim() -> None:
    """Make basicsr importable on modern torchvision.

    ``basicsr`` does ``from torchvision.transforms.functional_tensor import
    rgb_to_grayscale``, but that module was removed in torchvision >= 0.17.  We
    alias the function from ``torchvision.transforms.functional`` so the import
    succeeds without pinning an ancient torchvision.
    """
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except Exception:
        pass
    try:
        import torchvision.transforms.functional as F

        shim = types.ModuleType(name)
        shim.rgb_to_grayscale = F.rgb_to_grayscale  # type: ignore[attr-defined]
        sys.modules[name] = shim
        log.info("Installed torchvision.functional_tensor compatibility shim")
    except Exception as exc:  # pragma: no cover
        log.warning("Could not install torchvision shim (%s)", exc)


def _build_arch(spec: dict):
    """Construct the Real-ESRGAN network for a registry spec (needs torch)."""
    if spec["arch"] == "rrdb":
        from basicsr.archs.rrdbnet_arch import RRDBNet

        return RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=spec["num_block"], num_grow_ch=32, scale=spec["netscale"],
        )
    if spec["arch"] == "srvgg":
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        return SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=spec["num_conv"], upscale=spec["netscale"], act_type="prelu",
        )
    raise ValueError(f"Unknown arch: {spec['arch']}")


def _ensure_weights(url: str, filename: str) -> str:
    """Download model weights to the cache dir if missing; return local path."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local = WEIGHTS_DIR / filename
    if local.exists():
        return str(local)
    from basicsr.utils.download_util import load_file_from_url

    log.info("Downloading Real-ESRGAN weights: %s", filename)
    return load_file_from_url(
        url=url, model_dir=str(WEIGHTS_DIR), progress=True, file_name=filename
    )


class Upscaler:
    """Alpha-aware Real-ESRGAN upscaler.  Model is loaded **once**."""

    def __init__(self, cfg: PipelineConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.upsampler = None
        self.using_ml = False
        self.netscale = 4

        model_name = cfg.upscale_model
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown upscale model '{model_name}'. "
                f"Choices: {', '.join(available_models())}"
            )
        spec = _MODEL_REGISTRY[model_name]
        self.netscale = spec["netscale"]

        try:
            _install_torchvision_shim()
            import torch
            from realesrgan import RealESRGANer

            arch = _build_arch(spec)
            model_path = _ensure_weights(spec["url"], spec["file"])
            half = cfg.fp16 if cfg.fp16 is not None else (device == "cuda")

            self.upsampler = RealESRGANer(
                scale=spec["netscale"],
                model_path=model_path,
                model=arch,
                tile=cfg.tile,
                tile_pad=cfg.tile_pad,
                pre_pad=cfg.pre_pad,
                half=half,
                device=torch.device(device),
            )
            self.using_ml = True
            log.info(
                "Upscaler: Real-ESRGAN/%s (%s, netscale=%d, half=%s) on %s",
                model_name, spec["note"], spec["netscale"], half, device,
            )
        except Exception as exc:
            if not cfg.allow_fallback:
                raise
            log.warning(
                "Real-ESRGAN unavailable (%s) -> using LANCZOS upscale fallback "
                "(install requirements.txt for Real-ESRGAN).",
                exc,
            )

    # ------------------------------------------------------------------ #
    def upscale_rgba(
        self, rgb: np.ndarray, alpha: np.ndarray, outscale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Upscale RGB and ALPHA separately by ``outscale``; return both uint8.

        ``rgb``  : HxWx3 uint8 (already decontaminated)
        ``alpha``: HxW uint8
        """
        outscale = max(1.0, float(outscale))
        h, w = rgb.shape[:2]
        target_w, target_h = round(w * outscale), round(h * outscale)

        if self.using_ml and self.upsampler is not None:
            rgb_up = self._esrgan(rgb, outscale)
            if self.cfg.alpha_upsampler == "realesrgan":
                a3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
                alpha_up = self._esrgan(a3, outscale)[:, :, 0]
            else:
                alpha_up = _lanczos(alpha, target_w, target_h)
        else:
            rgb_up = _lanczos(rgb, target_w, target_h)
            alpha_up = _lanczos(alpha, target_w, target_h)

        # Guarantee identical dimensions before recombining.
        if alpha_up.shape[:2] != rgb_up.shape[:2]:
            alpha_up = _lanczos(alpha_up, rgb_up.shape[1], rgb_up.shape[0])

        log.info(
            "Upscaled %dx%d -> %dx%d (x%.3f, alpha=%s)",
            w, h, rgb_up.shape[1], rgb_up.shape[0], outscale,
            self.cfg.alpha_upsampler if self.using_ml else "lanczos",
        )
        return rgb_up, alpha_up

    def _esrgan(self, img_uint8_3ch: np.ndarray, outscale: float) -> np.ndarray:
        out, _ = self.upsampler.enhance(img_uint8_3ch, outscale=outscale)
        return np.ascontiguousarray(out[:, :, :3])


def _lanczos(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    interp = cv2.INTER_LANCZOS4
    return cv2.resize(img, (target_w, target_h), interpolation=interp)
