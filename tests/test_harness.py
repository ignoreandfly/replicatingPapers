"""Rung-0 tests. These must pass — the harness is the answer key.

If anything here breaks, every accuracy number produced by this repo is
fiction, so these are the tests that matter most and the ones worth being
paranoid in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.data import qa as qa_mod
from src.data.build import build_split, load_split, scene_fingerprint
from src.data.render import render, render_batch, shape_mask, to_float_chw
from src.data.scene import (
    BACKGROUND_RGB,
    CANVAS,
    COLOR_RGB,
    GRID,
    MAX_RADIUS,
    MAX_SHAPES,
    MIN_GAP,
    MIN_RADIUS,
    PATCH,
    RELATION_MARGIN,
    SHAPES,
    Scene,
    Shape,
    sample_scene,
)
from src.data.tokenizer import Tokenizer
from src.eval.baselines import (
    BagOfAttributesOracle,
    MajorityBlind,
    UniformBlind,
    all_blind_baselines,
)
from src.eval.scorer import score, wilson_interval

DATA = Path("data/shapes")


@pytest.fixture(scope="module")
def small_split() -> tuple[list[Scene], list]:
    return build_split("eval", 60)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


# --- geometry / renderer ----------------------------------------------------


def test_canvas_divides_into_the_patch_grid() -> None:
    assert CANVAS % PATCH == 0
    assert GRID * GRID == 64, "rung 1 and rung 3 assume 64 visual tokens"


def test_render_is_deterministic() -> None:
    scene = sample_scene(_rng(3), "t-0", "eval", 3)
    assert np.array_equal(render(scene), render(scene))


def test_render_uses_only_declared_colors() -> None:
    """No antialiasing means every pixel is exactly one of the palette values."""
    scenes = [sample_scene(_rng(i), f"t-{i}", "eval", i) for i in range(25)]
    allowed = {BACKGROUND_RGB} | set(COLOR_RGB.values())
    for scene in scenes:
        present = {tuple(int(v) for v in px) for px in np.unique(render(scene).reshape(-1, 3), axis=0)}
        assert present <= allowed, f"{scene.scene_id} has off-palette pixels: {present - allowed}"


def test_every_shape_is_actually_visible() -> None:
    """A shape in the answer key that renders to zero pixels is a silent lie."""
    for i in range(50):
        scene = sample_scene(_rng(100 + i), f"t-{i}", "eval", i)
        img = render(scene)
        flat = img.reshape(-1, 3)
        for color, n in scene.color_counts().items():
            px = int(np.all(flat == np.array(COLOR_RGB[color], np.uint8), axis=1).sum())
            if n > 0:
                assert px >= n * 20, f"{scene.scene_id}: {color} only {px}px for {n} shapes"
            else:
                assert px == 0, f"{scene.scene_id}: {color} present but not in the answer key"


def test_shapes_never_overlap_and_never_touch_the_border() -> None:
    for i in range(100):
        scene = sample_scene(_rng(200 + i), f"t-{i}", "eval", i)
        for a in range(scene.n):
            sa = scene.shapes[a]
            assert sa.cx - sa.radius >= 0 and sa.cx + sa.radius <= CANVAS - 1
            assert sa.cy - sa.radius >= 0 and sa.cy + sa.radius <= CANVAS - 1
            for b in range(a + 1, scene.n):
                sb = scene.shapes[b]
                gap = max(abs(sa.cx - sb.cx), abs(sa.cy - sb.cy)) - sa.radius - sb.radius
                assert gap >= MIN_GAP, f"{scene.scene_id}: shapes {a},{b} only {gap}px apart"


def test_shape_kinds_are_distinguishable_by_pixel_count() -> None:
    """circle < triangle areas differ enough that shape is recoverable at all.

    A square of half-side r covers (2r+1)^2, a circle ~pi r^2, a triangle ~half
    the square. If these ever collide, "which shape is red" is unanswerable
    from the image and the binding axis is measuring noise.
    """
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    for r in range(MIN_RADIUS, MAX_RADIUS + 1):
        areas = {
            kind: int(
                np.count_nonzero(
                    shape_mask(Shape(kind, "red", 32.0, 32.0, float(r)), xx, yy)
                )
            )
            for kind in SHAPES
        }
        assert areas["triangle"] < areas["circle"] < areas["square"], (r, areas)


def test_to_float_chw_layout_and_range() -> None:
    img = render(sample_scene(_rng(7), "t", "eval", 7))
    x = to_float_chw(img)
    assert x.shape == (3, CANVAS, CANVAS) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


# --- answer key -------------------------------------------------------------


def test_every_answer_is_inside_its_own_answer_space(small_split) -> None:
    _, questions = small_split
    for q in questions:
        assert q.answer in q.answer_space, q


def test_answer_key_agrees_with_the_pixels(small_split) -> None:
    scenes, _ = small_split
    images = render_batch(scenes)
    problems = []
    for scene, img in zip(scenes, images):
        problems += qa_mod.sanity_check_answer_key(scene, img)
    assert not problems, problems[:5]


def test_count_answers_match_the_scene(small_split) -> None:
    scenes, questions = small_split
    by_id = {s.scene_id: s for s in scenes}
    for q in questions:
        if q.template != "count.total":
            continue
        assert int(q.answer) == by_id[q.scene_id].n


def test_relation_answers_clear_the_margin(small_split) -> None:
    """Every emitted relation question must be unambiguous by construction."""
    scenes, questions = small_split
    by_id = {s.scene_id: s for s in scenes}
    for q in questions:
        if q.axis != "relation":
            continue
        a_key, b_key = q.qid.rsplit(":", 1)[-1].split("|")
        scene = by_id[q.scene_id]
        a = _find(scene, a_key)
        b = _find(scene, b_key)
        if q.template == "relation.horizontal":
            assert abs(a.cx - b.cx) >= RELATION_MARGIN
            assert q.answer == ("left" if a.cx < b.cx else "right")
        else:
            assert abs(a.cy - b.cy) >= RELATION_MARGIN
            assert q.answer == ("above" if a.cy < b.cy else "below")


def _find(scene: Scene, key: str) -> Shape:
    for s in scene.shapes:
        if f"{s.color}{s.kind}" == key:
            return s
    raise AssertionError(f"referent {key} not in {scene.scene_id}")


def test_binding_questions_are_not_answerable_from_attribute_presence(small_split) -> None:
    """The defining property of the binding axis.

    Every binding question must come from a scene with >= 2 colours and >= 2
    shape kinds, and its referent must be unique. Otherwise "name the colour
    you can see" would solve it and the axis would measure nothing.
    """
    scenes, questions = small_split
    by_id = {s.scene_id: s for s in scenes}
    for q in questions:
        if q.axis != "binding":
            continue
        scene = by_id[q.scene_id]
        assert len(scene.colors_present()) >= 2, q.qid
        assert len(scene.kinds_present()) >= 2, q.qid
        token = q.qid.rsplit(":", 1)[-1]
        if q.template == "binding.color_of_kind":
            assert scene.unique_kind(token) is not None
        else:
            assert scene.unique_color(token) is not None


def test_color_axis_is_answerable_from_presence(small_split) -> None:
    """The mirror image: colour questions must NOT need binding."""
    scenes, questions = small_split
    by_id = {s.scene_id: s for s in scenes}
    for q in questions:
        if q.axis != "color":
            continue
        scene = by_id[q.scene_id]
        assert len(set(scene.colors_present())) == 1, q.qid


def test_normalize_answer_handles_number_words_and_articles() -> None:
    n = qa_mod.normalize_answer
    assert n("Three.") == "3"
    assert n(" a circle ") == "circle"
    assert n("LEFT") == "left"
    assert n("none") == "0"


# --- splits -----------------------------------------------------------------


def test_splits_are_reproducible() -> None:
    a_scenes, a_q = build_split("eval", 40)
    b_scenes, b_q = build_split("eval", 40)
    assert [s.to_json() for s in a_scenes] == [s.to_json() for s in b_scenes]
    assert [q.to_json() for q in a_q] == [q.to_json() for q in b_q]


def test_a_split_contains_no_duplicate_scenes() -> None:
    scenes, _ = build_split("train", 300)
    fps = [scene_fingerprint(s) for s in scenes]
    assert len(set(fps)) == len(fps)


def test_train_and_eval_are_disjoint() -> None:
    train, _ = build_split("train", 400)
    train_fps = {scene_fingerprint(s) for s in train}
    ev, _ = build_split("eval", 200, exclude=train_fps)
    assert not (train_fps & {scene_fingerprint(s) for s in ev})


def test_all_four_axes_are_populated(small_split) -> None:
    _, questions = small_split
    axes = {q.axis for q in questions}
    assert axes == set(qa_mod.AXES), f"missing axes: {set(qa_mod.AXES) - axes}"


def test_scene_never_exceeds_max_shapes(small_split) -> None:
    scenes, _ = small_split
    assert all(1 <= s.n <= MAX_SHAPES for s in scenes)


# --- tokenizer --------------------------------------------------------------


def test_tokenizer_roundtrips_and_pads() -> None:
    tok = Tokenizer.build(["a red circle and a blue square", "how many shapes are there ?"])
    ids = tok.encode("a red circle", max_len=10)
    assert ids.shape == (10,) and ids.dtype == np.int64
    assert ids[0] == tok.bos_id
    assert tok.pad_id == 0
    nonpad = ids[ids != tok.pad_id]
    assert nonpad[-1] == tok.eos_id, "EOS must be the last non-pad token"
    assert tok.decode(ids) == "a red circle"


def test_tokenizer_truncates_but_keeps_eos() -> None:
    tok = Tokenizer.build(["a red circle and a blue square and a green triangle"])
    ids = tok.encode("a red circle and a blue square and a green triangle", max_len=6)
    assert ids.shape == (6,)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id


def test_tokenizer_vocabulary_is_order_independent() -> None:
    a = Tokenizer.build(["red circle", "blue square"])
    b = Tokenizer.build(["blue square", "red circle"])
    assert a.itos == b.itos


# --- scorer -----------------------------------------------------------------


def _records(split_questions) -> list[dict]:
    out = []
    for q in split_questions:
        rec = q.to_json()
        rec["image_index"] = 0
        out.append(rec)
    return out


def test_scorer_is_per_axis_and_has_no_aggregate(small_split) -> None:
    from src.eval.scorer import Report

    _, questions = small_split
    recs = _records(questions)
    perfect = {q["qid"]: q["answer"] for q in recs}
    report = score(recs, perfect, name="oracle")

    assert set(report.axes) == set(qa_mod.AXES)
    for axis in qa_mod.AXES:
        assert report.accuracy(axis) == 1.0
    assert not hasattr(Report, "overall"), "the scorer must not expose an aggregate"


def test_scorer_counts_wrong_answers(small_split) -> None:
    _, questions = small_split
    recs = _records(questions)
    wrong = {q["qid"]: "definitely not the answer" for q in recs}
    report = score(recs, wrong)
    assert all(report.axes[a].accuracy == 0.0 for a in report.axes)


def test_scorer_normalises_before_matching(small_split) -> None:
    _, questions = small_split
    recs = [q.to_json() | {"image_index": 0} for q in questions if q.axis == "count"]
    words = {"0": "zero", "1": "One", "2": "two", "3": "Three.", "4": "four", "5": "five"}
    preds = {q["qid"]: words[q["answer"]] for q in recs}
    assert score(recs, preds).accuracy("count") == 1.0


def test_scorer_refuses_to_silently_score_a_partial_run(small_split) -> None:
    _, questions = small_split
    recs = _records(questions)
    with pytest.raises(KeyError):
        score(recs, {recs[0]["qid"]: recs[0]["answer"]})


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0
    wide, narrow = wilson_interval(5, 10), wilson_interval(500, 1000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# --- baselines --------------------------------------------------------------


def test_uniform_blind_lands_near_the_analytic_chance(small_split) -> None:
    """The whole point of the blind baseline: it must reproduce 1/|space|.

    Averaged over many seeds, not one draw. A single draw on the ~30-question
    colour axis has a standard error of ~0.07, so a one-seed version of this
    test fails roughly one run in twenty for no reason at all — and a test that
    cries wolf about chance level is worse than no test, because the whole
    point of this file is to be trusted when it complains.
    """
    _, questions = small_split
    recs = _records(questions)
    n_seeds = 60

    totals: dict[str, list[int]] = {}
    for seed in range(n_seeds):
        report = score(recs, UniformBlind(seed=seed).predict(recs))
        for axis, res in report.axes.items():
            acc = totals.setdefault(axis, [0, 0, 0])
            acc[0] += res.n_correct
            acc[1] += res.n
            acc[2] = res.uniform_chance

    for axis, (correct, n, chance) in totals.items():
        empirical = correct / n
        se = math.sqrt(chance * (1 - chance) / n)
        assert abs(empirical - chance) <= 4 * se, (
            f"{axis}: uniform blind scored {empirical:.4f} over {n} draws, "
            f"analytic chance is {chance:.4f} (4 SE = {4 * se:.4f})"
        )


def test_relation_chance_is_one_half_not_one_quarter(small_split) -> None:
    """A question-aware guesser is at 0.5 on a two-alternative question.

    Quoting the axis-vocabulary figure (0.25) here would make any model look
    twice as good as it is.
    """
    _, questions = small_split
    recs = _records(questions)
    report = score(recs, UniformBlind(seed=0).predict(recs))
    assert report.axes["relation"].uniform_chance == pytest.approx(0.5, abs=1e-9)


def test_blind_baselines_cannot_see_the_image(small_split) -> None:
    """Same questions, images replaced with noise -> identical predictions."""
    _, questions = small_split
    recs = _records(questions)
    scrambled = [r | {"image_index": 999} for r in recs]
    for baseline in all_blind_baselines(seed=0):
        baseline.fit(recs)
        assert baseline.predict(recs) == baseline.predict(scrambled), baseline.name


def test_majority_baseline_is_constant_per_axis(small_split) -> None:
    _, questions = small_split
    recs = _records(questions)
    preds = MajorityBlind().fit(recs).predict(recs)
    by_axis: dict[str, set[str]] = {}
    for q in recs:
        by_axis.setdefault(q["axis"], set()).add(preds[q["qid"]])
    assert all(len(v) == 1 for v in by_axis.values()), by_axis


def test_bag_of_attributes_oracle_is_perfect_on_count_and_color(small_split) -> None:
    """It has the multisets, so counting and presence are free. Binding is not."""
    scenes, questions = small_split
    scene_recs = []
    for i, s in enumerate(scenes):
        rec = s.to_json()
        rec["image_index"] = i
        scene_recs.append(rec)
    index = {s.scene_id: i for i, s in enumerate(scenes)}
    q_recs = [q.to_json() | {"image_index": index[q.scene_id]} for q in questions]

    report = score(q_recs, BagOfAttributesOracle(scene_recs, seed=0).predict(q_recs))
    assert report.accuracy("count") == 1.0
    assert report.accuracy("color") == 1.0
    assert report.accuracy("binding") < 0.95, (
        "an unbound bag of attributes should NOT solve the binding axis — "
        "if it does, the binding questions have a shortcut"
    )


# --- the built dataset on disk ----------------------------------------------


@pytest.mark.skipif(not (DATA / "meta.json").exists(), reason="run `make data` first")
def test_built_dataset_is_internally_consistent() -> None:
    meta = json.loads((DATA / "meta.json").read_text())
    assert meta["train_eval_scene_overlap"] == 0

    for split in ("train", "eval"):
        images, scenes, questions = load_split(DATA, split)
        assert images.shape[1:] == (CANVAS, CANVAS, 3)
        assert len(images) == len(scenes) == meta["splits"][split]["n_images"]
        assert {q["axis"] for q in questions} == set(qa_mod.AXES)
        for q in questions[:200]:
            assert 0 <= q["image_index"] < len(images)
            assert q["answer"] in q["answer_space"]


@pytest.mark.skipif(not (DATA / "meta.json").exists(), reason="run `make data` first")
def test_answer_distribution_is_close_to_flat() -> None:
    """Balancing has to actually work, or majority-class beats every model.

    The count axis cannot be made perfectly flat — no scene has five purple
    shapes — so this asserts the achievable version: the most common answer on
    an axis is under 30%.
    """
    _, _, questions = load_split(DATA, "eval")
    for axis in qa_mod.AXES:
        answers = [q["answer"] for q in questions if q["axis"] == axis]
        top = max(answers.count(a) for a in set(answers)) / len(answers)
        assert top < 0.30, f"{axis}: most common answer covers {top:.1%} of questions"
