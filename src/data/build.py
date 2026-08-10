"""Build the rung-0 dataset: deterministic splits, balanced questions, on disk.

Determinism contract
--------------------
Splits are derived from `numpy.random.SeedSequence`, one spawned child per
scene. Train and eval start from different root entropy, so their scenes are
disjoint by construction, and re-running with the same seed reproduces the
dataset bit-for-bit on any machine. Nothing here reads a global RNG.

Balancing contract
------------------
Per axis, questions are chosen greedily to flatten the answer histogram. This
is not cosmetic: it pulls the majority-class baseline down towards uniform, so
"beat chance" means "read the image" rather than "learn that the answer to a
count question is usually 0". Where the answer distribution *cannot* be
flattened (there is no scene with five purple shapes), the residual skew shows
up in the blind baseline, which is exactly where I want to see it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .qa import AXES, GENERATORS, QAItem, sanity_check_answer_key
from .render import render_batch
from .scene import CANVAS, Scene, sample_scene

# Distinct root entropy per split -> disjoint scene streams.
SPLIT_ENTROPY: dict[str, int] = {
    "train": 0xC01D_5EED,
    "eval": 0x0E7A_1000,
}

# At most this many questions per axis per scene, so no single scene dominates
# an axis and the eval set stays image-diverse.
MAX_PER_AXIS_PER_SCENE: dict[str, int] = {
    "color": 1,
    "count": 2,
    "relation": 2,
    "binding": 2,
}


def scene_fingerprint(sc: Scene) -> str:
    """Identity of a scene up to its content, ignoring its id.

    The scene space is discrete and not enormous (3x3 cells, 3 kinds, 6
    colours, 5 sizes), so at 2500 samples exact duplicates happen by birthday
    paradox — about 9 of them across a 2000/500 split. Sampling train and eval
    from disjoint seed streams is therefore *not* enough to prevent an eval
    image from having been trained on. Hence this fingerprint, and the
    rejection pass in `build_split`.
    """
    return "|".join(
        f"{s.kind},{s.color},{s.cx:.0f},{s.cy:.0f},{s.radius:.0f}" for s in sc.shapes
    )


def build_split(
    split: str,
    n_images: int,
    root_entropy: int | None = None,
    exclude: set[str] | None = None,
) -> tuple[list[Scene], list[QAItem]]:
    """Sample `n_images` distinct scenes and a balanced question set over them.

    Scenes whose fingerprint is in `exclude`, or that duplicate one already
    accepted for this split, are rejected and resampled.
    """
    entropy = SPLIT_ENTROPY[split] if root_entropy is None else root_entropy
    seq = np.random.SeedSequence(entropy)
    blocked = set(exclude or ())

    scenes: list[Scene] = []
    seen: set[str] = set()
    pool: list = []
    ptr = 0
    while len(scenes) < n_images:
        if ptr >= len(pool):
            pool.extend(seq.spawn(max(64, n_images // 2)))
        child = pool[ptr]
        ptr += 1
        rng = np.random.Generator(np.random.PCG64(child))
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        sc = sample_scene(
            rng, scene_id=f"{split}-{len(scenes):06d}", split=split, seed=seed
        )
        fp = scene_fingerprint(sc)
        if fp in blocked or fp in seen:
            continue
        seen.add(fp)
        scenes.append(sc)

    questions = _select_balanced(scenes, np.random.Generator(np.random.PCG64(seq.spawn(1)[0])))
    return scenes, questions


def _select_balanced(scenes: list[Scene], rng: np.random.Generator) -> list[QAItem]:
    """Greedy answer-histogram flattening, per axis.

    Scenes are visited in a shuffled order; within a scene, the candidate whose
    answer is currently rarest on that axis wins. Rarest-first over a shuffled
    stream approximates a flat marginal without ever fabricating a question the
    scene does not support.
    """
    hist: dict[str, Counter] = {axis: Counter() for axis in AXES}
    chosen: list[QAItem] = []

    order = rng.permutation(len(scenes))
    for idx in order:
        scene = scenes[int(idx)]
        for axis in AXES:
            candidates = GENERATORS[axis](scene)
            if not candidates:
                continue
            cap = MAX_PER_AXIS_PER_SCENE[axis]
            # tie-break with a per-scene random key so the choice is not
            # alphabetical whenever counts are equal
            keys = rng.random(len(candidates))
            remaining = list(zip(candidates, keys))
            for _ in range(min(cap, len(remaining))):
                remaining.sort(key=lambda ck: (hist[axis][ck[0].answer], ck[1]))
                pick, _k = remaining.pop(0)
                hist[axis][pick.answer] += 1
                chosen.append(pick)

    chosen.sort(key=lambda q: q.qid)
    return chosen


def write_split(out_dir: Path, split: str, scenes: list[Scene], questions: list[QAItem]) -> dict:
    """Serialise one split and return its stats block."""
    out_dir.mkdir(parents=True, exist_ok=True)

    images = render_batch(scenes)  # (N, 64, 64, 3) uint8
    np.save(out_dir / f"{split}_images.npy", images)

    index = {sc.scene_id: i for i, sc in enumerate(scenes)}

    with open(out_dir / f"{split}_scenes.jsonl", "w") as f:
        for i, sc in enumerate(scenes):
            rec = sc.to_json()
            rec["image_index"] = i
            rec["caption"] = sc.caption()
            f.write(json.dumps(rec) + "\n")

    with open(out_dir / f"{split}_questions.jsonl", "w") as f:
        for q in questions:
            rec = q.to_json()
            rec["image_index"] = index[q.scene_id]
            f.write(json.dumps(rec) + "\n")

    # Audit the answer key against the pixels on a sample of the split.
    problems: list[str] = []
    audit_n = min(200, len(scenes))
    for i in range(audit_n):
        problems.extend(sanity_check_answer_key(scenes[i], images[i]))
    if problems:
        raise RuntimeError("answer key disagrees with the renderer:\n" + "\n".join(problems[:10]))

    per_axis = Counter(q.axis for q in questions)
    answer_hist: dict[str, dict[str, int]] = defaultdict(dict)
    for axis in AXES:
        c = Counter(q.answer for q in questions if q.axis == axis)
        answer_hist[axis] = dict(sorted(c.items()))

    return {
        "n_images": len(scenes),
        "n_questions": len(questions),
        "questions_per_axis": dict(sorted(per_axis.items())),
        "answer_histogram": dict(answer_hist),
        "recipes": dict(sorted(Counter(sc.recipe for sc in scenes).items())),
        "audited_scenes": audit_n,
    }


def load_split(data_dir: str | Path, split: str) -> tuple[np.ndarray, list[dict], list[dict]]:
    """(images uint8 (N,64,64,3), scene records, question records)."""
    d = Path(data_dir)
    images = np.load(d / f"{split}_images.npy")
    with open(d / f"{split}_scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    with open(d / f"{split}_questions.jsonl") as f:
        questions = [json.loads(line) for line in f]
    return images, scenes, questions


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the rung-0 synthetic dataset.")
    ap.add_argument("--out", default="data/shapes", type=Path)
    ap.add_argument("--n-train", default=2000, type=int)
    ap.add_argument("--n-eval", default=500, type=int)
    ap.add_argument("--contact-sheet", action="store_true", help="also dump a PNG grid to eyeball")
    args = ap.parse_args()

    meta: dict = {"canvas": CANVAS, "splits": {}}
    fingerprints: dict[str, set[str]] = {}

    # train first; eval then rejects anything train already contains
    for split, n in (("train", args.n_train), ("eval", args.n_eval)):
        scenes, questions = build_split(split, n, exclude=fingerprints.get("train"))
        meta["splits"][split] = write_split(args.out, split, scenes, questions)
        meta["splits"][split]["entropy"] = SPLIT_ENTROPY[split]
        fingerprints[split] = {scene_fingerprint(sc) for sc in scenes}
        print(f"[{split}] {len(scenes)} images, {len(questions)} questions")
        for axis, k in sorted(meta["splits"][split]["questions_per_axis"].items()):
            print(f"    {axis:<9} {k:>6}")

    overlap = fingerprints["train"] & fingerprints["eval"]
    meta["train_eval_scene_overlap"] = len(overlap)
    if overlap:
        raise RuntimeError(f"{len(overlap)} identical scenes survived the exclusion pass")

    with open(args.out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.contact_sheet:
        from .render import save_contact_sheet

        images, _, _ = load_split(args.out, "eval")
        save_contact_sheet(images[:32], str(args.out / "contact_sheet.png"))
        print(f"wrote {args.out / 'contact_sheet.png'}")

    print(f"wrote {args.out}/  (train/eval scene overlap: {len(overlap)})")


if __name__ == "__main__":
    main()
