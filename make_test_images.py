#!/usr/bin/env python3
"""Generate synthetic test images for the DTF pipeline.

Produces four cases that exercise the hard parts of the pipeline:

  1. logo_on_white.png   - logo on solid pure white (#ffffff)
  2. logo_on_color.png   - logo on a solid saturated color (worst case for halos)
  3. logo_on_noisy.png   - logo on a noisy off-white / gradient background
  4. photo_sphere.png    - a shaded sphere ("photographic-ish") on a studio gradient

The logos are rendered with genuine anti-aliased edges that blend the artwork
color with the background color at the rim - exactly the situation that produces
a DTF halo if decontamination is skipped, so the magenta QA images are meaningful.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).parent / "inputs"
SIZE = 1000          # native size of the logo test images
SS = 4               # supersample factor for anti-aliasing


# --------------------------------------------------------------------------- #
def _star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_logo(size: int) -> Image.Image:
    """A bold geometric emblem as a transparent RGBA with anti-aliased edges."""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = big // 2

    navy = (13, 42, 92, 255)
    teal = (26, 163, 163, 255)
    yellow = (247, 201, 72, 255)
    red = (214, 64, 54, 255)
    white = (255, 255, 255, 255)

    # Outer navy disk, then a slightly smaller teal disk -> leaves a navy ring.
    r = int(big * 0.46)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=navy)
    r2 = int(big * 0.40)
    d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=teal)

    # Thin white ring outline (tests survival of fine features).
    rw = int(big * 0.43)
    d.ellipse([cx - rw, cy - rw, cx + rw, cy + rw], outline=white,
              width=max(2, int(big * 0.010)))

    # Central yellow 5-point star (sharp corners + curves to stress the edge).
    star = _star_points(cx, cy, int(big * 0.30), int(big * 0.135), 5)
    d.polygon(star, fill=yellow)

    # Small red accent dot (separate component -> tests speckle/component logic).
    ar = int(big * 0.06)
    ax, ay = cx + int(big * 0.30), cy - int(big * 0.30)
    d.ellipse([ax - ar, ay - ar, ax + ar, ay + ar], fill=red)

    return img.resize((size, size), Image.Resampling.LANCZOS)


def _composite(logo: Image.Image, bg_rgb: np.ndarray) -> Image.Image:
    base = Image.fromarray(bg_rgb.astype(np.uint8), "RGB").convert("RGBA")
    base.alpha_composite(logo)
    return base.convert("RGB")


# --------------------------------------------------------------------------- #
def bg_solid(color, size=SIZE) -> np.ndarray:
    return np.full((size, size, 3), color, dtype=np.uint8)


def bg_noisy_offwhite(size=SIZE, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([244, 241, 234], np.float32)
    # vertical gradient + gaussian noise + faint blotches
    grad = np.linspace(-10, 12, size).reshape(size, 1, 1)
    img = base.reshape(1, 1, 3) + grad
    img = img + rng.normal(0, 6.0, (size, size, 3))
    # a couple of soft off-white blobs
    Y, X = np.mgrid[0:size, 0:size]
    for (bx, by, br, amp) in [(0.25, 0.3, 0.25, 10), (0.75, 0.7, 0.30, -12)]:
        d2 = ((X - bx * size) ** 2 + (Y - by * size) ** 2) / (br * size) ** 2
        img += (amp * np.exp(-d2))[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_sphere(size=SIZE, seed=3) -> np.ndarray:
    """A shaded orange sphere with a soft cast shadow on a studio gradient."""
    H = W = size
    rng = np.random.default_rng(seed)

    # Studio background: soft vertical gradient, slightly cool, light noise.
    top, bot = np.array([232, 232, 236]), np.array([198, 200, 208])
    t = np.linspace(0, 1, H).reshape(H, 1, 1)
    bg = (top.reshape(1, 1, 3) * (1 - t) + bot.reshape(1, 1, 3) * t)
    bg = bg + rng.normal(0, 2.0, (H, W, 3))

    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, R = W * 0.5, H * 0.46, size * 0.34

    # Soft elliptical cast shadow beneath the sphere.
    sh = np.exp(-(((X - cx) / (R * 1.15)) ** 2 + ((Y - (cy + R * 0.95)) / (R * 0.28)) ** 2))
    bg = bg - (sh[:, :, None] * np.array([60, 60, 64]))
    bg = np.clip(bg, 0, 255)

    # Sphere shading (Lambert + specular), light from upper-left.
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    z = np.sqrt(np.clip(R ** 2 - (X - cx) ** 2 - (Y - cy) ** 2, 0, None))
    nx, ny, nz = (X - cx) / R, (Y - cy) / R, z / R
    L = np.array([-0.5, -0.6, 0.62])
    L = L / np.linalg.norm(L)
    lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    base = np.array([235, 138, 38], np.float32)
    shade = 0.30 + 0.85 * lam
    spec = np.clip(lam, 0, 1) ** 24 * 200
    sphere = np.clip(base.reshape(1, 1, 3) * shade[:, :, None] + spec[:, :, None], 0, 255)

    # Soft (~1.3px) anti-aliased sphere edge.
    edge = np.clip((R - dist) / 1.3 + 0.5, 0, 1)[:, :, None]
    out = sphere * edge + bg * (1 - edge)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logo = draw_logo(SIZE)

    cases = {
        "logo_on_white.png": _composite(logo, bg_solid((255, 255, 255))),
        "logo_on_color.png": _composite(logo, bg_solid((230, 120, 40))),  # orange
        "logo_on_noisy.png": _composite(logo, bg_noisy_offwhite()),
    }
    for name, img in cases.items():
        img.save(OUT_DIR / name, dpi=(72, 72))
        print(f"wrote {OUT_DIR / name}  ({img.size[0]}x{img.size[1]})")

    sphere = Image.fromarray(make_sphere(SIZE), "RGB")
    sphere.save(OUT_DIR / "photo_sphere.png", dpi=(72, 72))
    print(f"wrote {OUT_DIR / 'photo_sphere.png'}  ({sphere.size[0]}x{sphere.size[1]})")


if __name__ == "__main__":
    main()
