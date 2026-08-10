"""Baselines that never look at the image.

Purpose: know the floor on every axis *before* training anything. A VLM that
scores 0.62 on the count axis has learned nothing if answering "1" to every
count question scores 0.61.

Four blind baselines, in increasing order of how much of the answer they can
extract from the question text alone:

  UniformBlind        uniform over this question's answer space. The honest
                      "coin flip", question-aware: 1/2 on relation, 1/6 on
                      colour, 1/3 on `binding.kind_of_color`.
  AxisUniformBlind    uniform over the whole *axis* vocabulary, ignoring which
                      subset this question allows. Lower than UniformBlind on
                      the relation axis; kept to make the difference visible,
                      because quoting 0.25 there would flatter every model.
  MajorityBlind       the single most frequent train answer per axis. Constant
                      output, and usually the strongest blind baseline.
  PriorBlind          samples from the train answer distribution per axis. Not
                      a bar to beat — it is here because it should score *worse*
                      than MajorityBlind, and if it doesn't, my train and eval
                      answer distributions have drifted apart.

And one non-blind diagnostic:

  BagOfAttributesOracle   sees the ground-truth multiset of colours and the
                      multiset of shape kinds, but *not* which colour goes with
                      which shape. It is a perfect object detector with no
                      binding. Its score on the binding axis is the ceiling for
                      any model that fails to bind — beat that, or the binding
                      axis number means nothing.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..data.qa import normalize_answer


class Baseline:
    """Produces {qid: answer} for a list of question records."""

    name = "baseline"

    def fit(self, train_questions: list[dict]) -> "Baseline":
        return self

    def predict(self, questions: list[dict]) -> dict[str, str]:
        raise NotImplementedError


class UniformBlind(Baseline):
    """Uniform over each question's own answer space."""

    name = "uniform"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def predict(self, questions: list[dict]) -> dict[str, str]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        return {q["qid"]: str(rng.choice(q["answer_space"])) for q in questions}


class AxisUniformBlind(Baseline):
    """Uniform over the axis's full vocabulary, ignoring the question."""

    name = "axis-unif"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.axis_vocab: dict[str, list[str]] = {}

    def fit(self, train_questions: list[dict]) -> "AxisUniformBlind":
        vocab: dict[str, set[str]] = defaultdict(set)
        for q in train_questions:
            vocab[q["axis"]].update(q["answer_space"])
        self.axis_vocab = {a: sorted(v) for a, v in vocab.items()}
        return self

    def predict(self, questions: list[dict]) -> dict[str, str]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        return {q["qid"]: str(rng.choice(self.axis_vocab[q["axis"]])) for q in questions}


class MajorityBlind(Baseline):
    """The most frequent train answer for the axis. Constant per axis."""

    name = "majority"

    def __init__(self, per_template: bool = False) -> None:
        self.per_template = per_template
        self.answer: dict[str, str] = {}

    def _key(self, q: dict) -> str:
        return q["template"] if self.per_template else q["axis"]

    def fit(self, train_questions: list[dict]) -> "MajorityBlind":
        counts: dict[str, Counter] = defaultdict(Counter)
        for q in train_questions:
            counts[self._key(q)][normalize_answer(q["answer"])] += 1
        # sort by (-count, answer) so ties resolve deterministically
        self.answer = {
            k: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            for k, c in counts.items()
        }
        return self

    def predict(self, questions: list[dict]) -> dict[str, str]:
        return {q["qid"]: self.answer.get(self._key(q), "") for q in questions}


class PriorBlind(Baseline):
    """Samples from the train answer distribution for the axis."""

    name = "prior"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.dist: dict[str, tuple[list[str], np.ndarray]] = {}

    def fit(self, train_questions: list[dict]) -> "PriorBlind":
        counts: dict[str, Counter] = defaultdict(Counter)
        for q in train_questions:
            counts[q["axis"]][normalize_answer(q["answer"])] += 1
        for axis, c in counts.items():
            items = sorted(c.items())
            p = np.array([v for _, v in items], dtype=np.float64)
            self.dist[axis] = ([k for k, _ in items], p / p.sum())
        return self

    def predict(self, questions: list[dict]) -> dict[str, str]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        out = {}
        for q in questions:
            answers, p = self.dist[q["axis"]]
            out[q["qid"]] = str(rng.choice(answers, p=p))
        return out


class BagOfAttributesOracle(Baseline):
    """Perfect detection, zero binding. NOT blind — a diagnostic ceiling.

    Sees, per image: the multiset of colours and the multiset of shape kinds,
    and the total shape count. Does not see which colour belongs to which
    shape, or where anything is.

    So it is exact on the count axis and on colour questions, and reduced to
    guessing among the attributes it knows are present on binding and relation.
    That gap is the whole reason the binding axis exists.
    """

    name = "bag-oracle"

    def __init__(self, scenes: list[dict], seed: int = 0) -> None:
        self.seed = seed
        self.by_index = {s["image_index"]: s for s in scenes}

    def predict(self, questions: list[dict]) -> dict[str, str]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        out: dict[str, str] = {}
        for q in questions:
            scene = self.by_index[q["image_index"]]
            colors = [s["color"] for s in scene["shapes"]]
            kinds = [s["kind"] for s in scene["shapes"]]
            space = list(q["answer_space"])

            if q["axis"] == "count":
                out[q["qid"]] = self._count(q, colors, kinds)
            elif q["axis"] == "color":
                # colour questions are answerable from presence alone, which is
                # exactly what this oracle has
                present = sorted(set(colors))
                out[q["qid"]] = present[0] if len(present) == 1 else str(rng.choice(present))
            elif q["axis"] == "binding":
                # It knows which attributes are in the image, just not what they
                # are attached to, so it guesses among the present ones.
                present = sorted(set(colors)) if _is_color_space(space) else sorted(set(kinds))
                pool = [a for a in present if a in space] or space
                out[q["qid"]] = str(rng.choice(pool))
            else:  # relation — no positions available, pure guess
                out[q["qid"]] = str(rng.choice(space))
        return out

    @staticmethod
    def _count(q: dict, colors: list[str], kinds: list[str]) -> str:
        tmpl = q["template"]
        if tmpl == "count.total":
            return str(len(colors))
        token = q["qid"].rsplit(":", 1)[-1]
        if tmpl == "count.kind":
            return str(sum(1 for k in kinds if k == token))
        if tmpl == "count.color":
            return str(sum(1 for c in colors if c == token))
        return "1"


def _is_color_space(space: list[str]) -> bool:
    from ..data.scene import COLORS

    return set(space) == set(COLORS)


def all_blind_baselines(seed: int = 0) -> list[Baseline]:
    return [
        UniformBlind(seed=seed),
        AxisUniformBlind(seed=seed + 1),
        MajorityBlind(),
        PriorBlind(seed=seed + 2),
    ]
