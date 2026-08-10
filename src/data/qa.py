"""Templated Q/A over four axes: colour, count, spatial relation, binding.

Every question carries an explicit `answer_space`: the closed set of strings
that could legally answer *this* question. That field is load-bearing three
times over:

  1. it makes chance level exact rather than assumed;
  2. it lets a blind baseline be question-aware, which is the honest floor
     (a model that reads "left or right?" is already at 1/2, not 1/4);
  3. it is the constrained decoding vocabulary at rung 3, when an LM generates
     the answer instead of classifying it.

Axis design note. The colour axis and the binding axis use overlapping surface
forms on purpose. The difference is the *scene*: colour questions are only
emitted when the answer follows from "which colours are present", binding
questions only when it does not. Any gap between the two axes is the model
failing to bind attributes to objects rather than failing to see colour.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .scene import (
    COLORS,
    COUNT_WORDS,
    MAX_SHAPES,
    RELATION_MARGIN,
    SHAPES,
    Scene,
)

AXES: tuple[str, ...] = ("color", "count", "relation", "binding")


@dataclass(frozen=True)
class QAItem:
    qid: str
    scene_id: str
    axis: str
    template: str
    question: str
    answer: str
    answer_space: tuple[str, ...]

    def to_json(self) -> dict:
        d = asdict(self)
        d["answer_space"] = list(self.answer_space)
        return d


def normalize_answer(text: str) -> str:
    """Canonical form for exact-match scoring.

    Lowercase, strip, drop trailing punctuation and a leading article. Number
    words map to digits so "three" and "3" score the same — the answer space
    always uses digits.
    """
    t = text.strip().lower().rstrip(".!?").strip()
    for article in ("the ", "a ", "an "):
        if t.startswith(article):
            t = t[len(article) :]
    return _NUMBER_WORDS.get(t, t)


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "none": "0",
}


def _plural(kind: str) -> str:
    return kind + "s"


# --- per-axis candidate generators -------------------------------------------
# Each returns every question the scene legally supports on that axis. build.py
# does the sampling and the answer-distribution balancing.


def color_questions(scene: Scene) -> list[QAItem]:
    """Answerable from "which colours are present" alone. The easy floor."""
    out: list[QAItem] = []
    if scene.n == 1:
        s = scene.shapes[0]
        out.append(
            _mk(
                scene,
                "color",
                "color.single",
                f"What color is the {s.kind}?",
                s.color,
                COLORS,
            )
        )
    elif scene.recipe == "uniform_color":
        color = scene.shapes[0].color
        out.append(
            _mk(
                scene,
                "color",
                "color.uniform",
                "What color are the shapes?",
                color,
                COLORS,
            )
        )
    return out


def count_questions(scene: Scene) -> list[QAItem]:
    """Includes queries whose true answer is 0, so presence != count."""
    out: list[QAItem] = []
    out.append(
        _mk(
            scene,
            "count",
            "count.total",
            "How many shapes are in the image?",
            str(scene.n),
            COUNT_WORDS[1:],  # a scene always has at least one shape
        )
    )
    kind_counts = scene.kind_counts()
    for kind in SHAPES:
        out.append(
            _mk(
                scene,
                "count",
                "count.kind",
                f"How many {_plural(kind)} are in the image?",
                str(kind_counts[kind]),
                COUNT_WORDS,
                key=kind,
            )
        )
    color_counts = scene.color_counts()
    for color in COLORS:
        out.append(
            _mk(
                scene,
                "count",
                "count.color",
                f"How many {color} shapes are in the image?",
                str(color_counts[color]),
                COUNT_WORDS,
                key=color,
            )
        )
    return out


def relation_questions(scene: Scene) -> list[QAItem]:
    """Two-alternative forced choice, only above a hard pixel margin."""
    out: list[QAItem] = []
    referents = scene.unique_referents()
    for i, a in enumerate(referents):
        for b in referents[i + 1 :]:
            if abs(a.cx - b.cx) >= RELATION_MARGIN:
                ans = "left" if a.cx < b.cx else "right"
                out.append(
                    _mk(
                        scene,
                        "relation",
                        "relation.horizontal",
                        f"Is the {a.color} {a.kind} to the left or to the right "
                        f"of the {b.color} {b.kind}?",
                        ans,
                        ("left", "right"),
                        key=f"{a.color}{a.kind}|{b.color}{b.kind}",
                    )
                )
            if abs(a.cy - b.cy) >= RELATION_MARGIN:
                ans = "above" if a.cy < b.cy else "below"
                out.append(
                    _mk(
                        scene,
                        "relation",
                        "relation.vertical",
                        f"Is the {a.color} {a.kind} above or below "
                        f"the {b.color} {b.kind}?",
                        ans,
                        ("above", "below"),
                        key=f"{a.color}{a.kind}|{b.color}{b.kind}",
                    )
                )
    return out


def binding_questions(scene: Scene) -> list[QAItem]:
    """Requires attaching an attribute to a specific object.

    Emitted only from `mixed` scenes, which are guaranteed to contain at least
    two colours and at least two shape kinds. So neither "which colours are
    present" nor "which kinds are present" determines the answer: a
    bag-of-attributes model is stuck guessing among the present attributes.
    `BagOfAttributesOracle` in src/eval/baselines.py measures exactly that
    ceiling, and it is the number to compare a real model against on this axis.
    """
    out: list[QAItem] = []
    if scene.recipe != "mixed":
        return out
    if len(scene.colors_present()) < 2 or len(scene.kinds_present()) < 2:
        return out

    # forward: object -> attribute
    for kind in SHAPES:
        s = scene.unique_kind(kind)
        if s is not None:
            out.append(
                _mk(
                    scene,
                    "binding",
                    "binding.color_of_kind",
                    f"What color is the {kind}?",
                    s.color,
                    COLORS,
                    key=kind,
                )
            )
    # reverse: attribute -> object
    for color in COLORS:
        s = scene.unique_color(color)
        if s is not None:
            out.append(
                _mk(
                    scene,
                    "binding",
                    "binding.kind_of_color",
                    f"Which shape is {color}?",
                    s.kind,
                    SHAPES,
                    key=color,
                )
            )
    return out


GENERATORS = {
    "color": color_questions,
    "count": count_questions,
    "relation": relation_questions,
    "binding": binding_questions,
}


def all_questions(scene: Scene) -> list[QAItem]:
    out: list[QAItem] = []
    for axis in AXES:
        out.extend(GENERATORS[axis](scene))
    return out


def _mk(
    scene: Scene,
    axis: str,
    template: str,
    question: str,
    answer: str,
    answer_space,
    key: str = "",
) -> QAItem:
    space = tuple(answer_space)
    assert answer in space, f"{template}: answer {answer!r} not in its own answer space"
    qid = f"{scene.scene_id}:{template}:{key}" if key else f"{scene.scene_id}:{template}"
    return QAItem(
        qid=qid,
        scene_id=scene.scene_id,
        axis=axis,
        template=template,
        question=question,
        answer=answer,
        answer_space=space,
    )


def sanity_check_answer_key(scene: Scene, rendered: np.ndarray) -> list[str]:
    """Re-derive every answer from the *pixels*, not from the Scene object.

    This is the harness auditing itself: if the renderer and the answer key
    ever disagree, every accuracy number downstream is fiction. Returns a list
    of human-readable mismatches (empty means clean).

    Colour recovery is exact because rendering has no antialiasing.
    """
    from .scene import COLOR_RGB

    problems: list[str] = []
    flat = rendered.reshape(-1, 3)
    for color, rgb in COLOR_RGB.items():
        pixels = int(np.all(flat == np.array(rgb, dtype=np.uint8), axis=1).sum())
        expected = scene.color_counts()[color]
        if (pixels > 0) != (expected > 0):
            problems.append(
                f"{scene.scene_id}: colour {color} has {pixels}px but "
                f"{expected} shapes in the answer key"
            )
    if scene.n > MAX_SHAPES:
        problems.append(f"{scene.scene_id}: {scene.n} shapes exceeds MAX_SHAPES")
    return problems
