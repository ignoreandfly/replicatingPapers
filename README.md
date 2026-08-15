# replicatingPapers

Implementations and reproductions of deep learning papers, written for
understanding rather than for reuse. Everything is PyTorch.

The repo has two halves: standalone per-paper reproductions, and **toy-vlm**, a
structured ladder of vision-language papers built on one shared harness.

---

## Standalone reproductions

| Paper | Folder |
|------------------------------------------------------|--------------|
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | [ViT](./ViT) |
| [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) | [GAN](./GAN) |
| [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | [CLIP](./CLIP) |
| [SigLIP / PaliGemma modelling](https://arxiv.org/abs/2303.15343) | [PaliGemma](./PaliGemma) |

---

## toy-vlm — the ladder

Seven rungs, each one paper's idea added to the last, all evaluated on the same
synthetic harness so the comparisons actually mean something. See
[CLAUDE.md](./CLAUDE.md) for the full plan and [NOTES.md](./NOTES.md) for what
happened at each rung.

| # | Rung | Status |
|---|--------------------------------------------------|-------------|
| 0 | Synthetic harness — renderer, Q/A, per-axis scoring | done |
| 1 | CLIP from scratch — ViT + text encoder, InfoNCE | in progress |
| 2 | SigLIP as a one-loss diff | |
| 3 | Frozen encoder + linear projector + frozen LM | |
| 4 | Connector ablation — MLP, Q-Former, Perceiver + gated cross-attn | |
| 5 | Visual instruction tuning with LoRA | |
| 6 | Visual token budget — AnyRes tiling, pixel shuffle | |

```
src/
  models/     hand-written forward passes — attention, losses, connectors
  data/       renderer, Q/A templating, splits, tokenizer, datasets
  eval/       per-axis scorer, blind baselines
  train/      loops, seeding, overfit-8
experiments/  one config per ablation
```

### The harness (rung 0)

64×64 RGB scenes: 1–5 non-overlapping coloured shapes on a jittered grid, no
antialiasing, so every pixel is exactly one palette value and the ground truth
is recoverable from the image by construction. 2000 train / 500 eval images,
scene-disjoint. Questions span four axes, each carrying an explicit closed
answer space.

Scoring is **per-axis, never aggregate** — a single number over four unrelated
abilities mostly serves to hide which one is broken.

**Chance level on eval**, measured before training anything:

| axis | n | best blind floor | bag-of-attributes oracle |
|----------|------|------------------|--------------------------|
| color | 183 | 0.186 | **1.000** |
| count | 1000 | 0.205 | **1.000** |
| relation | 459 | **0.499** | 0.508 |
| binding | 416 | 0.248 | **0.406** |

Two things that table is for. Relation chance is **0.50, not 0.25** — the
questions are two-alternative, so a guesser that merely reads the question is
already at half. And colour and count are **free**: a model with perfect
detection and no attribute binding at all scores 1.000 on both. Only the
relation and binding columns are evidence of anything.

Regenerate with `make data`; the build is deterministic and bit-reproducible.

---

## Getting started

Python 3.11 and a cu118 PyTorch build:

```bash
make venv      # uv venv + pinned deps
make data      # generate the rung-0 dataset
make chance    # print the per-axis floors
make test      # full suite
```

> **CUDA note.** The dev box runs NVIDIA driver 515, which is CUDA 11.7-era, so
> torch must be a `+cu118` wheel — every default cu12x wheel from PyPI installs
> fine and then fails to initialise. `make venv` pins this correctly. On a
> newer driver, relax the pins in `pyproject.toml`.

`make test` exits non-zero while a rung is in progress: the tests for each rung
are written before its implementation, so they fail by design until the forward
pass exists. `make test-harness` covers rung 0 and should always be green — if
it isn't, every accuracy number in the repo is fiction.

---

## Contributions

Issues and pull requests welcome on the standalone reproductions. The `toy-vlm`
model code under `src/models/` is deliberately hand-written as a learning
exercise, so please don't send patches that complete it.
