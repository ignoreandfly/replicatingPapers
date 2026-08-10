"""Rasteriser: Scene -> uint8 image. Pure numpy, no PIL, no antialiasing.

No antialiasing is a deliberate choice. Every pixel is exactly the background
colour or exactly one shape's colour, so `render()` is bit-exact reproducible
across machines and a test can assert on exact pixel counts.
"""

from __future__ import annotations

import numpy as np

from .scene import BACKGROUND_RGB, CANVAS, COLOR_RGB, Scene, Shape


def render(scene: Scene, size: int = CANVAS) -> np.ndarray:
    """Rasterise a scene.

    Returns uint8 (H, W, 3) in [0, 255]. Shapes are painted in list order, but
    the scene sampler guarantees non-overlap, so paint order is irrelevant.
    """
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:, :] = np.array(BACKGROUND_RGB, dtype=np.uint8)

    yy, xx = np.mgrid[0:size, 0:size]  # (H, W) each, integer pixel centres
    for shape in scene.shapes:
        mask = shape_mask(shape, xx, yy)  # (H, W) bool
        img[mask] = np.array(COLOR_RGB[shape.color], dtype=np.uint8)
    return img


def shape_mask(shape: Shape, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Boolean coverage mask for one primitive. (H, W) -> (H, W)."""
    dx = xx - shape.cx
    dy = yy - shape.cy
    r = shape.radius

    if shape.kind == "circle":
        return (dx * dx + dy * dy) <= r * r

    if shape.kind == "square":
        return (np.abs(dx) <= r) & (np.abs(dy) <= r)

    if shape.kind == "triangle":
        # Upward isoceles: apex (cx, cy - r), base corners (cx +/- r, cy + r).
        # Inside iff below the apex-level, above the base, and within the two
        # slanted edges: |dx| <= (dy + r) / 2.
        return (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2.0)

    raise ValueError(f"unknown shape kind: {shape.kind!r}")


def to_float_chw(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]. Model-facing layout."""
    return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0


def render_batch(scenes: list[Scene], size: int = CANVAS) -> np.ndarray:
    """(N, H, W, 3) uint8 for a list of scenes."""
    out = np.empty((len(scenes), size, size, 3), dtype=np.uint8)
    for i, sc in enumerate(scenes):
        out[i] = render(sc, size=size)
    return out


def save_contact_sheet(images: np.ndarray, path: str, cols: int = 8) -> None:
    """Dump a grid of rendered images to a PNG so I can eyeball the renderer.

    A synthetic harness that is never looked at is a synthetic harness with a
    silent bug in it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for ax, img in zip(np.ravel(np.atleast_1d(axes)), images):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in np.ravel(np.atleast_1d(axes))[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
