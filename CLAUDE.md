# CLAUDE.md — toy-vlm

## What this repo is for

This is a **learning repo**. I am re-implementing VLM papers as toy variants to
understand them, while revising vision-language modelling. The artifact is not
the point — my ability to write these forward passes unaided is the point.

Optimising for a working model at the expense of my understanding is a failure,
even if every test passes.

---

## The contract

### You (Claude Code) write freely

- Data pipelines, dataset generation, dataloaders, collators
- The synthetic eval harness (renderer, Q/A templating, scoring, per-axis breakdown)
- Training-loop boilerplate: checkpointing, resumption, config, logging, seeding
- Plotting, results tables, experiment bookkeeping
- Environment setup, dependency pinning, Makefile, CI, remote-GPU sync scripts
- Tests — including tests for code I haven't written yet
- Numerical diffs against reference implementations (HF, timm, open_clip)

### I write, by hand, first

Everything in `src/models/**`. Specifically:

- Attention, InfoNCE, the SigLIP sigmoid loss
- The projector splice into the LM embedding sequence
- Q-Former cross-attention, Perceiver resampler, gated cross-attn
- All loss masking

**Do not write, complete, refactor, or "fix" files under `src/models/` unless I
say the exact phrase `take the wheel`.** If a file there is broken, describe the
bug — do not patch it. If I ask you to implement something there, remind me of
this rule once and ask if I want to override it. One reminder, then respect my
answer.

Scaffolding is allowed in `src/models/`: signatures, docstrings, shape
annotations in comments, and `raise NotImplementedError` bodies. Never a working
body.

---

## How to help when I'm stuck

Escalate in this order. Do not skip rungs, do not pre-empt the next one.

1. **Symptom** — describe what's wrong in observable terms. "Your loss is
   exactly `ln(batch_size)` and flat."
2. **Location** — narrow it to a function or a few lines, without naming the fix.
3. **Concept** — ask the question that would surface the bug. "What does the
   diagonal of your similarity matrix mean here?"
4. **Fix** — only if I say `just tell me`.

If I've been on the same bug for a while and I say `just tell me`, give the
answer plainly and without ceremony. The ladder is a default, not a hazing
ritual.

---

## Review mode

When I say `review`, compare my code against the paper and report divergences.
**Report only — do not edit.** Format:

- Where my implementation differs from the paper
- Whether each difference is a bug, a deliberate toy simplification, or
  ambiguous
- What experiment would tell us which

Be direct about mistakes. Vague encouragement is worse than useless here.

---

## Definition of done for each rung

A rung is not finished when it runs. It's finished when:

- [ ] It overfits 8 examples to near-zero loss
- [ ] Numerically diffed against a reference implementation, or explicitly noted
      as having none
- [ ] One ablation run, with a plot
- [ ] A paragraph in `NOTES.md`: what I expected, what happened, what I got wrong
- [ ] I can rewrite the core forward pass from memory on a blank file

The last one is the real test. If I can't, the rung isn't done regardless of
what the metrics say.

---

## The ladder

Prerequisites assumed: Transformer, ViT, CLIP, LoRA.

### 0. Harness — *agent-built*
Synthetic scene renderer (colored shapes on a canvas), ground-truth attribute
emission, templated Q/A across four axes: **color, count, spatial relation,
attribute binding**. ~2k images. Per-axis scoring, not aggregate.

Built before rung 3, not after. Synthetic first because I control the answer key.

### 1. CLIP from scratch — *I write the loss and encoders*
Small ViT + small text encoder, symmetric InfoNCE, learnable temperature.
Eval: zero-shot classification with prompt templates.
Ablation: accuracy vs. batch size.

### 2. SigLIP as a one-loss diff — *I write the loss*
Swap softmax-CE for pairwise sigmoid with learnable bias `b` and scale `t`.
Everything else held fixed.
Ablation: reproduce the small-batch claim. Both losses, same sweep, one plot.

### 3. Frozen encoder + linear projector + frozen LM — *I write the splice*
LLaVA stage 1. Frozen patch embeddings → `nn.Linear` → prepended to the LM token
embedding sequence → caption loss on **text tokens only**.

The highest-value rung. Expected bugs: embedding-space scale mismatch, attention
mask over image tokens, loss masking. I should feel all three.

### 4. Connector ablation — *I write each connector*
Data and eval frozen from rung 3. Swap only the bridge:
linear → 2-layer MLP (LLaVA-1.5) → Q-Former (BLIP-2) → Perceiver resampler +
gated cross-attn (Flamingo). Four papers, one harness, one table.

### 5. Visual instruction tuning — *I write the LoRA wiring*
Stage 2, LM unfrozen via LoRA, small procedurally generated instruction set.
Real caption data (COCO subset / Flickr8k) enters here, not before.

### 6. Visual token budget — *I write the tiling*
AnyRes tiling, pixel-shuffle compression.
Ablation: accuracy vs. number of visual tokens.

---

## Layout

```
src/
  models/        # mine — you don't write here
  data/          # yours
  eval/          # yours
  train/         # yours — except loss masking, which is mine
experiments/     # configs, one per ablation
NOTES.md         # my reflections, one entry per rung
```

---

## Style

- Shape annotations in comments on every tensor op in `src/models/`
- Every model module runs on random tensors before any training code exists
- One ablation per rung. An implementation with no comparison is a tutorial I
  wrote for myself.

## Hardware

Single RTX 3090, 24 GB, local. No distributed training, no remote sync.

- bf16 everywhere; TF32 matmuls enabled; SDPA/flash attention available
- No `all_gather` in contrastive losses — batch is local. This is the point,
  not a limitation: it's what makes the SigLIP small-batch claim testable here.
- Frozen vision tower outputs are cached to disk before rung 3 and reused.
  Never re-run a frozen encoder inside the training loop.
- `torch.compile` for sweeps only, not for debugging runs.
- Default LM: SmolLM2-360M through rung 4, Qwen2.5-1.5B + LoRA at rung 5.
- Every rung must have a `--smoke` config that runs end-to-end in under
  two minutes, so I can verify plumbing without burning a real run.