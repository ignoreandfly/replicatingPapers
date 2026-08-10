"""Scene description: the ground truth, defined before any pixel exists.

The whole point of rung 0 is that I control the answer key. A `Scene` is the
answer key; `render.py` is the only thing allowed to turn it into pixels, and
`qa.py` is the only thing allowed to turn it into questions.

Invariant enforced here: shapes never overlap and never touch the border, so
every ground-truth attribute (colour, count, position) is recoverable from the
image by construction. No occlusion, no antialiasing, no ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

# --- vocabulary -------------------------------------------------------------
# These lists are the closed answer vocabularies. Every answer emitted anywhere
# in the harness is a member of one of them. Order is fixed and load-bearing:
# it defines label indices for the tokenizer and the scorer.

COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow", "purple", "cyan")
SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
RELATIONS: tuple[str, ...] = ("left", "right", "above", "below")
MAX_SHAPES = 5
COUNT_WORDS: tuple[str, ...] = tuple(str(i) for i in range(MAX_SHAPES + 1))

# RGB values chosen to be far apart in both RGB and luminance, so "which colour
# is this" is never a near-tie for a small model.
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (50, 90, 220),
    "yellow": (240, 200, 40),
    "purple": (150, 60, 200),
    "cyan": (40, 200, 210),
}
BACKGROUND_RGB: tuple[int, int, int] = (245, 245, 245)

# --- canvas geometry --------------------------------------------------------
# 64x64 with patch 8 -> 8x8 = 64 visual tokens at rung 1 and rung 3.
CANVAS = 64
PATCH = 8
GRID = CANVAS // PATCH  # 8

MIN_RADIUS = 5
MAX_RADIUS = 7
BORDER_MARGIN = 2  # px of background guaranteed between a shape and the edge
MIN_GAP = 3  # px of background guaranteed between two shapes

# Shapes are placed on a jittered 3x3 cell grid rather than by rejection
# sampling. Two reasons. Packing five non-overlapping shapes into 64px by
# rejection sampling fails often enough to matter, and a guaranteed cell
# separation means every left/right question clears RELATION_MARGIN by
# construction — position is never a near-tie that rendering jitter could flip.
CELLS = 3  # 3x3 = 9 slots, MAX_SHAPES=5 of them filled
CELL_JITTER = 2  # px of random offset from the cell centre

# A left/right (or above/below) question is only emitted when the two centres
# differ by at least this many pixels on that axis. Below it, the relation is a
# coin flip on rendering jitter and would poison the eval. Adjacent cells are
# >=18px apart after jitter, so this threshold cleanly separates "different
# column" from "same column".
RELATION_MARGIN = 10

ShapeName = Literal["circle", "square", "triangle"]

# Scene "recipes" decide which question axes a scene can support. A scene with
# one shape cannot test attribute binding; a monochrome scene cannot either.
# Sampling a mix means every axis has enough support without any single scene
# being contrived for one axis.
Recipe = Literal["single", "uniform_color", "mixed", "dupes"]
RECIPE_WEIGHTS: dict[Recipe, float] = {
    "single": 0.18,
    "uniform_color": 0.18,
    "mixed": 0.42,
    "dupes": 0.22,
}


@dataclass(frozen=True)
class Shape:
    """One rendered primitive.

    (cx, cy) is the centre in pixel coords with origin at top-left, y down.
    `radius` is the circumradius-ish half-extent: for a square it is the half
    side length, for a circle the radius, for a triangle half the base width.
    """

    kind: ShapeName
    color: str
    cx: float
    cy: float
    radius: float

    @property
    def half_extent(self) -> float:
        return self.radius


@dataclass(frozen=True)
class Scene:
    """A full answer key for one image."""

    scene_id: str
    split: str
    recipe: Recipe
    shapes: tuple[Shape, ...]
    seed: int

    # --- derived views the Q/A templates query ------------------------------

    @property
    def n(self) -> int:
        return len(self.shapes)

    def color_counts(self) -> dict[str, int]:
        out = {c: 0 for c in COLORS}
        for s in self.shapes:
            out[s.color] += 1
        return out

    def kind_counts(self) -> dict[str, int]:
        out = {k: 0 for k in SHAPES}
        for s in self.shapes:
            out[s.kind] += 1
        return out

    def colors_present(self) -> list[str]:
        return [c for c in COLORS if self.color_counts()[c] > 0]

    def kinds_present(self) -> list[str]:
        return [k for k in SHAPES if self.kind_counts()[k] > 0]

    def unique_kind(self, kind: str) -> Shape | None:
        """The only shape of this kind, or None if there are zero or many."""
        hits = [s for s in self.shapes if s.kind == kind]
        return hits[0] if len(hits) == 1 else None

    def unique_color(self, color: str) -> Shape | None:
        hits = [s for s in self.shapes if s.color == color]
        return hits[0] if len(hits) == 1 else None

    def unique_referents(self) -> list[Shape]:
        """Shapes identifiable by the phrase "the {color} {kind}"."""
        pairs: dict[tuple[str, str], int] = {}
        for s in self.shapes:
            pairs[(s.color, s.kind)] = pairs.get((s.color, s.kind), 0) + 1
        return [s for s in self.shapes if pairs[(s.color, s.kind)] == 1]

    def caption(self) -> str:
        """Deterministic caption, shapes ordered left-to-right.

        Used as the text side of the rung-1 CLIP pairs and as the target of the
        rung-3 caption loss. Left-to-right order means the caption carries the
        spatial information a bag-of-words model cannot fake.
        """
        ordered = sorted(self.shapes, key=lambda s: (round(s.cx), round(s.cy)))
        parts = [f"a {s.color} {s.kind}" for s in ordered]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    def to_json(self) -> dict:
        d = asdict(self)
        d["shapes"] = [asdict(s) for s in self.shapes]
        return d


def sample_scene(rng: np.random.Generator, scene_id: str, split: str, seed: int) -> Scene:
    """Sample one non-overlapping scene according to a randomly chosen recipe."""
    recipes = list(RECIPE_WEIGHTS)
    weights = np.array([RECIPE_WEIGHTS[r] for r in recipes], dtype=np.float64)
    recipe: Recipe = recipes[int(rng.choice(len(recipes), p=weights / weights.sum()))]

    if recipe == "single":
        n = 1
    else:
        n = int(rng.integers(2, MAX_SHAPES + 1))

    kinds, colors = _sample_attributes(rng, recipe, n)
    placements = _sample_placements(rng, n)

    shapes = tuple(
        Shape(kind=k, color=c, cx=cx, cy=cy, radius=r)
        for k, c, (cx, cy, r) in zip(kinds, colors, placements)
    )
    return Scene(scene_id=scene_id, split=split, recipe=recipe, shapes=shapes, seed=seed)


def _sample_attributes(
    rng: np.random.Generator, recipe: Recipe, n: int
) -> tuple[list[str], list[str]]:
    """Pick (kind, colour) per shape so the recipe's guarantee actually holds."""
    if recipe == "single":
        return [str(rng.choice(SHAPES))], [str(rng.choice(COLORS))]

    if recipe == "uniform_color":
        color = str(rng.choice(COLORS))
        kinds = [str(rng.choice(SHAPES)) for _ in range(n)]
        return kinds, [color] * n

    if recipe == "mixed":
        # Guarantee: >=2 distinct colours AND >=2 distinct kinds, so neither the
        # colour multiset nor the kind multiset alone determines a binding
        # answer. This is the only recipe the binding axis draws from.
        kinds = list(rng.choice(SHAPES, size=2, replace=False))
        colors = list(rng.choice(COLORS, size=2, replace=False))
        for _ in range(n - 2):
            kinds.append(str(rng.choice(SHAPES)))
            colors.append(str(rng.choice(COLORS)))
        order = rng.permutation(n)
        return [str(kinds[i]) for i in order], [str(colors[i]) for i in order]

    # "dupes": deliberately repeat kinds and colours so counting has to count
    # rather than detect presence.
    kind_pool = list(rng.choice(SHAPES, size=max(1, n // 2), replace=True))
    color_pool = list(rng.choice(COLORS, size=max(1, n // 2), replace=True))
    kinds = [str(rng.choice(kind_pool)) for _ in range(n)]
    colors = [str(rng.choice(color_pool)) for _ in range(n)]
    return kinds, colors


def cell_centers() -> list[float]:
    """The CELLS grid-line positions, chosen so the largest shape still fits.

    Centres span [MAX_RADIUS + BORDER_MARGIN, CANVAS - 1 - MAX_RADIUS -
    BORDER_MARGIN], which for the default constants is [9, 54].
    """
    lo = MAX_RADIUS + BORDER_MARGIN
    hi = CANVAS - 1 - MAX_RADIUS - BORDER_MARGIN
    return [lo + (hi - lo) * k / (CELLS - 1) for k in range(CELLS)]


def _sample_placements(rng: np.random.Generator, n: int) -> list[tuple[float, float, float]]:
    """Place n shapes in n distinct jittered grid cells.

    Non-overlap is what makes the answer key exact: no shape can hide another,
    so `count` is never a judgement call. Here it is guaranteed by geometry
    rather than by rejection, and then asserted anyway.
    """
    if n > CELLS * CELLS:
        raise ValueError(f"cannot place {n} shapes in {CELLS * CELLS} cells")

    centers = cell_centers()
    slots = rng.choice(CELLS * CELLS, size=n, replace=False)

    placed: list[tuple[float, float, float]] = []
    for slot in slots:
        row, col = divmod(int(slot), CELLS)
        r = float(rng.integers(MIN_RADIUS, MAX_RADIUS + 1))
        lo, hi = r + BORDER_MARGIN, CANVAS - 1 - r - BORDER_MARGIN
        jx = float(rng.integers(-CELL_JITTER, CELL_JITTER + 1))
        jy = float(rng.integers(-CELL_JITTER, CELL_JITTER + 1))
        cx = float(min(max(round(centers[col] + jx), lo), hi))
        cy = float(min(max(round(centers[row] + jy), lo), hi))
        placed.append((cx, cy, r))

    for i, (ax, ay, ar) in enumerate(placed):
        for bx, by, br in placed[i + 1 :]:
            if not _separated(ax, ay, ar, bx, by, br):
                raise RuntimeError(
                    "grid geometry no longer guarantees separation; "
                    "check CELLS / CELL_JITTER / MAX_RADIUS / MIN_GAP"
                )
    return placed


def _separated(ax: float, ay: float, ar: float, bx: float, by: float, br: float) -> bool:
    """Chebyshev separation: correct for squares, conservative for the rest."""
    return max(abs(ax - bx), abs(ay - by)) >= ar + br + MIN_GAP
