"""Per-axis scoring. There is deliberately no aggregate accuracy.

A single number over colour + count + relation + binding is a weighted average
over four unrelated abilities, and its main effect is to let a model that
learned the colour axis and nothing else look like it is "at 61%". Every
consumer here gets an axis breakdown or nothing.

Each axis also carries its *own* floor:

  `uniform_chance`  — a guesser that reads the question and picks uniformly
                      from that question's answer space. This is the honest
                      floor for a two-alternative relation question (0.50), not
                      the 0.25 you would get from the axis vocabulary.
  `majority`        — the best constant-per-axis answer fitted on train. Where
                      the answer distribution could not be flattened, this is
                      higher than uniform, and it is the real bar.

Beating the *max* of those two, per axis, is the only evidence that anything
was learned.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..data.qa import normalize_answer


@dataclass
class AxisResult:
    axis: str
    n: int
    n_correct: int
    uniform_chance: float
    per_template: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")

    @property
    def ci95(self) -> tuple[float, float]:
        return wilson_interval(self.n_correct, self.n)

    def template_rows(self) -> list[tuple[str, int, float]]:
        rows = []
        for tmpl, (correct, total) in sorted(self.per_template.items()):
            rows.append((tmpl, total, correct / total if total else float("nan")))
        return rows


@dataclass
class Report:
    """Per-axis results. Note the absence of an `.overall` property."""

    axes: dict[str, AxisResult]
    name: str = ""

    def accuracy(self, axis: str) -> float:
        return self.axes[axis].accuracy

    def headroom(self, axis: str) -> float:
        """Accuracy minus that axis's uniform floor. Negative is a red flag."""
        a = self.axes[axis]
        return a.accuracy - a.uniform_chance

    def to_table(self, show_templates: bool = False) -> str:
        head = f"{'axis':<10}{'n':>7}{'acc':>9}{'95% CI':>18}{'chance':>9}{'delta':>9}"
        lines = [f"== {self.name} ==" if self.name else "", head, "-" * len(head)]
        for axis in sorted(self.axes):
            a = self.axes[axis]
            lo, hi = a.ci95
            lines.append(
                f"{axis:<10}{a.n:>7}{a.accuracy:>9.3f}"
                f"{f'[{lo:.3f}, {hi:.3f}]':>18}{a.uniform_chance:>9.3f}"
                f"{a.accuracy - a.uniform_chance:>+9.3f}"
            )
            if show_templates:
                for tmpl, n, acc in a.template_rows():
                    lines.append(f"  {tmpl:<26}{n:>5}{acc:>9.3f}")
        return "\n".join(x for x in lines if x != "" or len(lines) > 1)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "axes": {
                axis: {
                    "n": a.n,
                    "n_correct": a.n_correct,
                    "accuracy": a.accuracy,
                    "ci95": list(a.ci95),
                    "uniform_chance": a.uniform_chance,
                    "per_template": {
                        t: {"n": n, "accuracy": acc} for t, n, acc in a.template_rows()
                    },
                }
                for axis, a in sorted(self.axes.items())
            },
        }


def score(
    questions: list[dict],
    predictions: dict[str, str],
    name: str = "",
    strict: bool = True,
) -> Report:
    """Exact-match scoring after `normalize_answer`, broken out per axis.

    `questions` are records as written by src/data/build.py; `predictions` maps
    qid -> predicted answer string. A missing prediction counts as wrong (and
    raises under `strict`, because silently scoring a partial run as if it were
    a full one is how a broken eval loop looks like a bad model).
    """
    missing = [q["qid"] for q in questions if q["qid"] not in predictions]
    if missing and strict:
        raise KeyError(
            f"{len(missing)} of {len(questions)} questions have no prediction "
            f"(first: {missing[0]}). Pass strict=False to score them as wrong."
        )

    n: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    chance_sum: dict[str, float] = defaultdict(float)
    per_template: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for q in questions:
        axis = q["axis"]
        gold = normalize_answer(q["answer"])
        pred = normalize_answer(predictions.get(q["qid"], ""))
        hit = int(pred == gold)

        n[axis] += 1
        correct[axis] += hit
        space = q["answer_space"]
        chance_sum[axis] += 1.0 / len(space) if space else float("nan")
        cell = per_template[axis][q["template"]]
        cell[0] += hit
        cell[1] += 1

    axes = {
        axis: AxisResult(
            axis=axis,
            n=n[axis],
            n_correct=correct[axis],
            uniform_chance=chance_sum[axis] / n[axis],
            per_template={t: (c[0], c[1]) for t, c in per_template[axis].items()},
        )
        for axis in sorted(n)
    }
    return Report(axes=axes, name=name)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at the extremes, unlike normal-approx.

    With ~200 eval questions on an axis, the difference between a model at 0.99
    and one at 1.00 is inside the noise; this is here so I read the number with
    the right amount of confidence.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compare(reports: list[Report], axes: list[str] | None = None) -> str:
    """Side-by-side table of several reports, one column per report."""
    if not reports:
        return ""
    axes = axes or sorted({a for r in reports for a in r.axes})
    width = max(max(len(r.name) for r in reports), 7) + 2
    lines = [f"{'axis':<10}{'n':>7}" + "".join(f"{r.name:>{width}}" for r in reports)]
    lines.append("-" * len(lines[0]))
    for axis in axes:
        n = next((r.axes[axis].n for r in reports if axis in r.axes), 0)
        row = f"{axis:<10}{n:>7}"
        for r in reports:
            row += (
                f"{r.axes[axis].accuracy:>{width}.3f}"
                if axis in r.axes
                else f"{'-':>{width}}"
            )
        lines.append(row)
    return "\n".join(lines)
