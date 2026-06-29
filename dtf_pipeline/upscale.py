"""Stage 2 - upscaling with Real-ESRGAN (alpha-aware).

Real-ESRGAN does not handle a 4-channel RGBA image well, so we split the
channels: the RGB is upscaled by the model, the ALPHA channel is upscaled
separately (LANCZOS by default - cleanest edge on graphics - or by the model),
then the two are recombined at the upscaled resolution.  Decontamination has
already run, so the RGB is clean everywhere under the alpha before it is scaled.

If torch or the weights are unavailable, a high-quality LANCZOS upscale is used
instead so the pipeline still runs end-to-end.

RRDBNet and SRVGGNetCompact are inlined here so this project has no dependency
on the unmaintained basicsr / realesrgan packages.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .config import PipelineConfig
from .utils import log

WEIGHTS_DIR = Path.home() / ".cache" / "dtf_pipeline" / "weights"
_GH = "https://github.com/xinntao/Real-ESRGAN/releases/download"

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


# ---------------------------------------------------------------------------
# Inlined Real-ESRGAN architectures (weight-compatible with xinntao releases)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as _F

    class _ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class _RRDB(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class _RRDBNet(nn.Module):
        """RRDB generator used by x4plus, x4plus-anime, x2plus weight files."""

        def __init__(
            self,
            num_block: int,
            scale: int,
            num_feat: int = 64,
            num_grow_ch: int = 32,
        ):
            super().__init__()
            self._scale = scale
            num_in = 3 * (4 if scale == 2 else 16 if scale == 1 else 1)
            self.conv_first = nn.Conv2d(num_in, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
            )
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if scale == 8:
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self._scale == 2:
                feat = _F.pixel_unshuffle(x, downscale_factor=2)
            elif self._scale == 1:
                feat = _F.pixel_unshuffle(x, downscale_factor=4)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(
                self.conv_up1(_F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(_F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            if self._scale == 8:
                feat = self.lrelu(
                    self.conv_up3(_F.interpolate(feat, scale_factor=2, mode="nearest"))
                )
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    class _SRVGGNetCompact(nn.Module):
        """Compact VGG-style generator used by general-x4v3 and animevideov3."""

        def __init__(self, num_conv: int, scale: int, num_feat: int = 64):
            super().__init__()
            self._scale = scale
            layers: list[nn.Module] = [
                nn.Conv2d(3, num_feat, 3, 1, 1),
                nn.PReLU(num_feat),
            ]
            for _ in range(num_conv):
                layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
            layers.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            self.upsampler = nn.PixelShuffle(scale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = _F.interpolate(x, scale_factor=self._scale, mode="nearest")
            return out + base

    _TORCH_AVAILABLE = True

except Exception:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_arch(spec: dict) -> "nn.Module":
    if spec["arch"] == "rrdb":
        return _RRDBNet(num_block=spec["num_block"], scale=spec["netscale"])
    if spec["arch"] == "srvgg":
        return _SRVGGNetCompact(num_conv=spec["num_conv"], scale=spec["netscale"])
    raise ValueError(f"Unknown arch: {spec['arch']}")


def _ensure_weights(url: str, filename: str) -> str:
    """Download model weights to the cache dir if missing; return local path."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local = WEIGHTS_DIR / filename
    if local.exists():
        return str(local)
    log.info("Downloading Real-ESRGAN weights: %s", filename)
    urllib.request.urlretrieve(url, str(local))
    return str(local)


def _load_weights(model: "nn.Module", path: str) -> None:
    """Load a Real-ESRGAN checkpoint; handles params_ema / params wrappers."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Public upscaler
# ---------------------------------------------------------------------------

class Upscaler:
    """Alpha-aware Real-ESRGAN upscaler.  Model is loaded **once**."""

    def __init__(self, cfg: PipelineConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.torch_device = None
        self.using_ml = False
        self.netscale = 4
        self.half = False

        model_name = cfg.upscale_model
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown upscale model '{model_name}'. "
                f"Choices: {', '.join(available_models())}"
            )
        spec = _MODEL_REGISTRY[model_name]
        self.netscale = spec["netscale"]

        try:
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch not available")
            arch = _build_arch(spec)
            model_path = _ensure_weights(spec["url"], spec["file"])
            _load_weights(arch, model_path)

            self.half = cfg.fp16 if cfg.fp16 is not None else (device == "cuda")
            self.torch_device = torch.device(device)
            arch = arch.to(self.torch_device)
            if self.half:
                arch = arch.half()
            arch.eval()
            self.model = arch
            self.using_ml = True
            log.info(
                "Upscaler: Real-ESRGAN/%s (%s, netscale=%d, half=%s) on %s",
                model_name, spec["note"], spec["netscale"], self.half, device,
            )
        except Exception as exc:
            if not cfg.allow_fallback:
                raise
            log.warning(
                "Real-ESRGAN unavailable (%s) -> using LANCZOS upscale fallback "
                "(install torch and run the pipeline to enable Real-ESRGAN).",
                exc,
            )

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

        if self.using_ml and self.model is not None:
            rgb_up = self._esrgan(rgb, outscale)
            if self.cfg.alpha_upsampler == "realesrgan":
                a3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
                alpha_up = self._esrgan(a3, outscale)[:, :, 0]
            else:
                alpha_up = _lanczos(alpha, target_w, target_h)
        else:
            rgb_up = _lanczos(rgb, target_w, target_h)
            alpha_up = _lanczos(alpha, target_w, target_h)

        if alpha_up.shape[:2] != rgb_up.shape[:2]:
            alpha_up = _lanczos(alpha_up, rgb_up.shape[1], rgb_up.shape[0])

        log.info(
            "Upscaled %dx%d -> %dx%d (x%.3f, alpha=%s)",
            w, h, rgb_up.shape[1], rgb_up.shape[0], outscale,
            self.cfg.alpha_upsampler if self.using_ml else "lanczos",
        )
        return rgb_up, alpha_up

    def _esrgan(self, img_rgb: np.ndarray, outscale: float) -> np.ndarray:
        """Run model on HxWx3 uint8 RGB; return HxWx3 uint8 RGB."""
        h, w = img_rgb.shape[:2]

        x = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(
            np.ascontiguousarray(x.transpose(2, 0, 1))
        ).unsqueeze(0)
        if self.half:
            tensor = tensor.half()
        tensor = tensor.to(self.torch_device)

        with torch.no_grad():
            out = self._tile_process(tensor) if self.cfg.tile > 0 else self.model(tensor)

        arr = out.squeeze(0).float().cpu().clamp(0, 1).numpy()
        arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

        if abs(outscale - self.netscale) > 0.005:
            ow, oh = round(w * outscale), round(h * outscale)
            arr = cv2.resize(arr, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

        return arr

    def _tile_process(self, x: "torch.Tensor") -> "torch.Tensor":
        """Tile-based inference for VRAM-limited GPUs."""
        _, c, h, w = x.shape
        tile = self.cfg.tile
        pad = self.cfg.tile_pad
        scale = self.netscale

        out = torch.zeros(1, c, h * scale, w * scale, dtype=x.dtype, device=x.device)
        tiles_y = math.ceil(h / tile)
        tiles_x = math.ceil(w / tile)

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                y0 = max(iy * tile - pad, 0)
                y1 = min((iy + 1) * tile + pad, h)
                x0 = max(ix * tile - pad, 0)
                x1 = min((ix + 1) * tile + pad, w)

                tile_out = self.model(x[:, :, y0:y1, x0:x1])

                # Strip padding from output tile
                out_y0 = iy * tile * scale
                out_y1 = min((iy + 1) * tile, h) * scale
                out_x0 = ix * tile * scale
                out_x1 = min((ix + 1) * tile, w) * scale
                th, tw = out_y1 - out_y0, out_x1 - out_x0
                py = (iy * tile - y0) * scale
                px = (ix * tile - x0) * scale

                out[:, :, out_y0:out_y1, out_x0:out_x1] = (
                    tile_out[:, :, py : py + th, px : px + tw]
                )

        return out


def _lanczos(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
