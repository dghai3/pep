"""Stage 1a - background segmentation.

Primary path: BiRefNet via the ``rembg`` library with alpha matting enabled.
BiRefNet infers at a fixed working resolution (~1024 px), so we run it on a
downscaled copy of the RGB and only the resulting **mask** makes the round trip
back to native resolution (upsampled with LANCZOS).  The full-resolution RGB is
never destructively resized.

Fallback path (no torch / rembg / model weights): a classical colour-distance +
GrabCut segmenter so the pipeline still runs end-to-end and produces a real
cutout.  The ML path is always preferred when available.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .utils import log

try:  # ML path is optional at import time
    import rembg  # type: ignore

    _REMBG_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on machines without rembg
    rembg = None  # type: ignore
    _REMBG_AVAILABLE = False


def _ort_providers(device: str) -> List[str]:
    """onnxruntime execution providers for the resolved device."""
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resize_long_side(rgb: np.ndarray, target_long: int) -> np.ndarray:
    """Downscale so the longest side == ``target_long`` (never upscales)."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long:
        return rgb
    scale = target_long / float(long_side)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


class Segmenter:
    """Produces a native-resolution alpha mask from an RGB image.

    The model (or fallback) is initialised **once** so batch runs don't reload
    weights per image.
    """

    def __init__(self, cfg: PipelineConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.session = None
        self.using_ml = False

        if _REMBG_AVAILABLE:
            try:
                providers = _ort_providers(device)
                try:
                    self.session = rembg.new_session(
                        cfg.rembg_model, providers=providers
                    )
                except TypeError:
                    # Older rembg signatures don't take providers kwarg.
                    self.session = rembg.new_session(cfg.rembg_model)
                self.using_ml = True
                log.info(
                    "Segmenter: rembg/%s (alpha_matting=%s) on %s",
                    cfg.rembg_model,
                    cfg.alpha_matting,
                    device,
                )
            except Exception as exc:
                if not cfg.allow_fallback:
                    raise
                log.warning(
                    "rembg session '%s' failed (%s); using classical fallback "
                    "segmentation.",
                    cfg.rembg_model,
                    exc,
                )
        else:
            if not cfg.allow_fallback:
                raise RuntimeError(
                    "rembg is not installed and allow_fallback=False. "
                    "Install requirements.txt to enable BiRefNet."
                )
            log.warning(
                "rembg not installed -> using CLASSICAL FALLBACK segmentation "
                "(install requirements.txt for BiRefNet quality)."
            )

    # ------------------------------------------------------------------ #
    def mask(self, rgb: np.ndarray) -> np.ndarray:
        """Return a native-resolution float32 alpha in ``[0, 1]`` for ``rgb``."""
        native_h, native_w = rgb.shape[:2]
        small = _resize_long_side(rgb, self.cfg.working_resolution)

        if self.using_ml and self.session is not None:
            small_alpha = self._mask_rembg(small)
        else:
            small_alpha = self._mask_classical(small)

        # Only the MASK makes the round trip back to native res (LANCZOS).
        if small_alpha.shape[:2] != (native_h, native_w):
            alpha = cv2.resize(
                small_alpha, (native_w, native_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            alpha = small_alpha
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _mask_rembg(self, small_rgb: np.ndarray) -> np.ndarray:
        """Run BiRefNet via rembg and return the (matted) alpha in [0,1]."""
        pil = Image.fromarray(small_rgb, mode="RGB")
        out = rembg.remove(
            pil,
            session=self.session,
            alpha_matting=self.cfg.alpha_matting,
            alpha_matting_foreground_threshold=self.cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=self.cfg.alpha_matting_erode_size,
            only_mask=False,
            post_process_mask=True,
        )
        rgba = np.asarray(out.convert("RGBA"))
        return rgba[:, :, 3].astype(np.float32) / 255.0

    # ------------------------------------------------------------------ #
    def _mask_classical(self, small_rgb: np.ndarray) -> np.ndarray:
        """Classical fallback segmentation.

        Primary method is **flood-fill from the border**: the background is the
        background-coloured region *connected to the image edge*.  Anything not
        reachable from the border stays foreground - so a white letter fill
        enclosed by a dark outline is kept even though it matches the white
        background colour (the white-on-white case these DTF graphics hit).

        If that produces a degenerate result, fall back to a colour-distance +
        Otsu + GrabCut segmentation (better for soft/photographic subjects).

        Not a substitute for BiRefNet, but produces a genuine, clean cutout so
        the rest of the pipeline runs end-to-end.
        """
        h, w = small_rgb.shape[:2]
        fg = self._floodfill_bg(small_rgb)
        if fg is not None:
            frac = float((fg > 0).mean())
            if 0.003 < frac < 0.97:
                return self._finish_fg(fg)
        return self._mask_color_otsu(small_rgb)

    def _floodfill_bg(self, small_rgb: np.ndarray):
        """Background = border-connected, background-coloured region."""
        h, w = small_rgb.shape[:2]
        lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        frame = max(2, int(round(0.03 * min(h, w))))
        border = np.concatenate([
            lab[:frame, :, :].reshape(-1, 3), lab[-frame:, :, :].reshape(-1, 3),
            lab[:, :frame, :].reshape(-1, 3), lab[:, -frame:, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)

        dist = cv2.GaussianBlur(np.linalg.norm(lab - bg, axis=2), (0, 0), sigmaX=1.0)
        border_dist = np.linalg.norm(border - bg, axis=1)
        tol = float(np.clip(np.percentile(border_dist, 99) * 1.6, 12.0, 60.0))
        bg_similar = (dist < tol).astype(np.uint8)

        n, labels = cv2.connectedComponents(bg_similar, connectivity=8)
        if n <= 1:
            return None
        edge_labels = set(labels[0, :]) | set(labels[-1, :]) \
            | set(labels[:, 0]) | set(labels[:, -1])
        edge_labels.discard(0)  # 0 = the non-bg-similar region
        if not edge_labels:
            return None
        background = np.isin(labels, list(edge_labels)) & (bg_similar > 0)
        fg = np.where(background, 0, 255).astype(np.uint8)
        return fg

    def _finish_fg(self, fg: np.ndarray) -> np.ndarray:
        """Clean a binary foreground mask and return a lightly-AA'd float alpha."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = self._drop_small_components(fg, min_area_frac=0.0005)
        alpha = cv2.GaussianBlur(fg.astype(np.float32) / 255.0, (0, 0), sigmaX=0.8)
        return np.clip(alpha, 0.0, 1.0)

    def _mask_color_otsu(self, small_rgb: np.ndarray) -> np.ndarray:
        """Colour-distance + Otsu + GrabCut (good for soft/photographic subjects)."""
        h, w = small_rgb.shape[:2]
        lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        frame = max(2, int(round(0.03 * min(h, w))))
        border = np.concatenate([
            lab[:frame, :, :].reshape(-1, 3), lab[-frame:, :, :].reshape(-1, 3),
            lab[:, :frame, :].reshape(-1, 3), lab[:, -frame:, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)

        dist = cv2.GaussianBlur(np.linalg.norm(lab - bg, axis=2), (0, 0), sigmaX=1.0)
        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
        binary = self._drop_small_components(binary, min_area_frac=0.0005)
        if binary.max() == 0:
            return np.zeros((h, w), np.float32)

        refined = self._grabcut_refine(small_rgb, binary)
        if refined is not None and (refined > 0).sum() >= 0.5 * float((binary > 0).sum()):
            binary = refined

        alpha = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), sigmaX=0.8)
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _drop_small_components(binary: np.ndarray, min_area_frac: float) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return binary
        min_area = min_area_frac * binary.shape[0] * binary.shape[1]
        keep = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        return keep

    @staticmethod
    def _grabcut_refine(rgb: np.ndarray, binary: np.ndarray) -> Optional[np.ndarray]:
        try:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            sure_fg = cv2.erode(binary, k, iterations=3)
            sure_bg = cv2.dilate(binary, k, iterations=6)

            gc = np.full(binary.shape, cv2.GC_PR_BGD, np.uint8)
            gc[sure_bg == 0] = cv2.GC_BGD
            gc[binary > 0] = cv2.GC_PR_FGD
            gc[sure_fg > 0] = cv2.GC_FGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                rgb, gc, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK
            )
            out = np.where(
                (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
            return out
        except Exception as exc:  # pragma: no cover
            log.warning("GrabCut refine skipped (%s)", exc)
            return None
