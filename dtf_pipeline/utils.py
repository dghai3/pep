"""Shared helpers: logging, device detection, small array conversions."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Tuple

import numpy as np

_LOGGER_NAME = "dtf"


def get_logger(quiet: bool = False) -> logging.Logger:
    """Return the package logger, configured once."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    return logger


log = get_logger()


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Context manager that logs how long a block took."""
    start = time.perf_counter()
    yield
    log.info("%s took %.2fs", label, time.perf_counter() - start)


def detect_device(preference: str = "auto") -> str:
    """Resolve the compute device.

    Returns one of ``"cuda"``, ``"mps"``, ``"cpu"``.  Falls back gracefully when
    torch is not installed (returns ``"cpu"``).  Warns when no GPU is found, as
    BiRefNet and Real-ESRGAN are slow on CPU.
    """
    if preference not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device preference: {preference}")

    try:
        import torch  # noqa: WPS433 (local import is intentional)
    except Exception:  # torch not installed -> classical fallback path
        if preference not in ("auto", "cpu"):
            log.warning("torch not installed; ignoring device=%s, using CPU", preference)
        return "cpu"

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_ok = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )

    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not cuda_ok:
            log.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return "cuda"
    if preference == "mps":
        if not mps_ok:
            log.warning("MPS requested but unavailable; falling back to CPU")
            return "cpu"
        return "mps"

    # auto
    if cuda_ok:
        gpu = torch.cuda.get_device_name(0)
        log.info("Using CUDA GPU: %s", gpu)
        return "cuda"
    if mps_ok:
        log.info("Using Apple Metal (MPS) GPU")
        return "mps"
    log.warning(
        "No GPU detected -> running on CPU. BiRefNet + Real-ESRGAN are SLOW on "
        "CPU; expect tens of seconds to minutes per image."
    )
    return "cpu"


def to_float01(arr: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint8 [0,255] (rounded, clamped)."""
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def alpha_bbox(alpha: np.ndarray, threshold: int = 0) -> Tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of non-transparent pixels.

    Returns the full frame if the mask is empty.  ``x1``/``y1`` are exclusive.
    """
    ys, xs = np.where(alpha > threshold)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
