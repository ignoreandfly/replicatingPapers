# NOTES

One entry per rung: what I expected, what happened, what I got wrong.
This file is mine. Claude Code does not write entries here.

---

## Rung 0 — harness (agent-built)

Built by Claude Code, so there is no "what I got wrong" entry. What I need to
carry forward instead is the answer key's shape and its floors.

**Scene space.** 64x64 RGB, shapes on a jittered 3x3 grid, 1–5 non-overlapping
shapes, 6 colours x 3 kinds. No antialiasing, so every pixel is exactly one
palette value and the answer key is recoverable from the image by
construction. 2000 train / 500 eval images, scene-disjoint (exact duplicates
happen by birthday paradox at this scale and are rejected, not just asserted
away).

**Floors on eval** — the numbers to beat, per axis:

| axis     | n    | best blind floor | via     | bag-of-attributes oracle |
|----------|------|------------------|---------|--------------------------|
| color    | 183  | 0.186            | prior   | 1.000                    |
| count    | 1000 | 0.205            | uniform | 1.000                    |
| relation | 459  | 0.499            | uniform | 0.508                    |
| binding  | 416  | 0.248            | uniform | 0.406                    |

Two things to remember about this table:

- **Relation chance is 0.50, not 0.25.** The questions are two-alternative
  forced choice; a guesser that merely reads the question is already at half.
- **Colour and count are free.** A model with perfect detection and no binding
  at all scores 1.000 on both. They are wiring checks, not evidence. The only
  axes that say anything are relation (0.508 without positions) and binding
  (0.406 without binding). Anything at or below those numbers has learned to
  detect and nothing more.

---

## Rung 1 — CLIP from scratch

_(mine to write)_

Expected:

What happened:

What I got wrong:

Definition of done:
- [ ] overfits 8 examples to near-zero loss
- [ ] numerically diffed against `open_clip` / HF, or noted as having no reference
- [ ] ablation: accuracy vs. batch size, with a plot
- [ ] this paragraph, filled in
- [ ] I can rewrite the forward pass from memory on a blank file
