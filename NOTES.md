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

---

# Appendix — tensor mechanics

Not a rung entry. Reference notes from writing attention by hand, kept because
these are the moves every rung from here reuses.

## view / transpose / reshape

Three operations, different jobs, and neither of the first two can do the
other's work:

| op | changes | keeps fixed |
|-----------|-----------------------------|-------------------|
| `view` | number and sizes of axes | element count, order |
| `transpose` | order of axes | number and sizes |
| `reshape` | = `view` if possible, else `contiguous().view()` | |

A tensor is one flat array plus a rule for indexing it (strides). `view` and
`transpose` never move data — they only change the rule.

**`reshape` is never numerically wrong.** Verified: `reshape(...)` is exactly
`contiguous().view(...)`, same values always. The only thing at stake is a
hidden copy. `view` is the stricter choice: it refuses rather than copies, so
it tells me when my assumption about layout is wrong. timm uses `reshape`
throughout; nanoGPT uses explicit `.contiguous().view()`. Either is fine, but
pick one and be consistent.

Where it actually bites, measured:

| step | `view` | `reshape` |
|--------------------------------------------|--------|-----------|
| split `(B,L,W)` → `(B,L,H,Dh)` | works | works |
| merge `(B,L,H,Dh)` → `(B,L,W)` after transpose | **fails** | works |

Splitting a stride-1 dimension is always expressible in strides, so `view`
succeeds even on the non-contiguous output of `chunk`. Merging *across* a
transposed layout is not, so `view` errors and the copy has to be explicit.

## The head reshape order — the one silent bug

`(B, L, W)` → `view(B, L, H, Dh)` → `transpose(1, 2)` → `(B, H, L, Dh)`.

**Split the dimension where it already lives, then move it.** `W` is last, so
split there; then reorder. Two operations because they are two different
things.

Doing it in one step — `view(B, H, L, Dh)` — produces the right shape from
scrambled data. With L=4 patches, W=6 features, H=2 heads, values as
`patch.feature`:

```
CORRECT  view(B,L,H,Dh).transpose(1,2)      head 0 = features 0-2 of ALL patches
  head 0:  [0.0 0.1 0.2]                    <- a feature subspace, every position
           [1.0 1.1 1.2]
           [2.0 2.1 2.2]
           [3.0 3.1 3.2]

WRONG    view(B,H,L,Dh)                     head 0 = ALL features of patches 0,1
  head 0:  [0.0 0.1 0.2]                    <- a partition of POSITIONS
           [0.3 0.4 0.5]
           [1.0 1.1 1.2]
           [1.3 1.4 1.5]
```

Memory runs patch-major, so the one-step view just chops the flat run into H
blocks, and the first block is the first two patches. Same shape, different
data, no error, for any H — because `H * Dh == W` always.

Damage if I get it wrong: with 64 patches and 3 heads, head 0 only ever sees
patches 0-21. Structurally incapable of relating the left of an image to the
right, so relation and binding sit near chance forever while colour and count
look fine.

**Why head-major at all:** `matmul` multiplies the last two dims and batches
over the rest. After the transpose the trailing pair is `(L, Dh)`, so
`q @ k.transpose(-2,-1)` gives `(L, L)` — a patch × patch score matrix. Skip
the transpose and the trailing pair is `(H, Dh)`, giving a meaningless head ×
head matrix computed per patch.

Merging back is the exact inverse, in reverse order: transpose first, then
collapse.

## Masks

Both go on `attn_logits` **before** the softmax. `exp(-inf) = 0`, so masked
entries vanish and the survivors renormalise — rows still sum to 1. Zeroing
after the softmax leaves rows summing to less than 1: subtly wrong instead of
loudly wrong.

|         | causal | padding |
|---------|-----------------------|---------------------------|
| shape | `(L, L)`, built here | `(B, L)`, passed in |
| broadcasts to | `(1, 1, L, L)` — implicit | `(B, 1, 1, L)` — I write it |
| masks | future positions | pad keys, per example |
| gated on | `self.causal` | `mask is not None` |

`torch.ones(L, L, dtype=torch.bool, device=x.device).tril()` gives "allowed":
row i = query i, col j = key j, True = may attend.

**Polarity.** `masked_fill` fills where the mask is `True`, but my mask marks
what to *keep*, so it takes `~allowed`. Getting this backwards gives a model
that sees only the future — which still trains and still shows a falling loss.

`(B, L)` → `(B, 1, 1, L)` via `[:, None, None, :]`. `L` must land **last** so
it masks keys (columns). Second-to-last would mask queries instead: different
model, no error.

Note `True = keep` is the opposite of PyTorch's own `attn_mask` convention.

## Things that error confusingly

- **`device=` goes on creation, not operations.** `torch.ones/zeros/arange/randn`
  take it; `masked_fill` and friends inherit from their input. A CPU-created
  mask is invisible until the first CUDA run — same class as the `torch.arange`
  in the InfoNCE loss.
- **Module vs tensor vs tuple.** `self.qkv` is a layer; `self.qkv(x)` is a
  tensor; `.chunk(3, -1)` is a *tuple*. Tensor methods only work on the middle
  one. Three separate slips, all the same shape of mistake: chaining a tensor
  method onto something that is not a tensor.
- **`nn.Parameter` wraps a Tensor, not a Module.** `nn.Linear` already owns its
  weight and bias — assign it bare. `nn.Parameter` is for learnable tensors no
  layer provides: CLS token, positional embeddings, `logit_scale`.
- **`super().__init__()` first**, before any `self.x = <Module>`. Otherwise
  `cannot assign module before Module.__init__() call`, pointing at a line that
  is fine.
- **`float("-inf")`**, not `float(-inf)`. The minus goes inside the quotes.
- **`F.gelu` is lowercase**; `nn.GELU` is the capitalised Module.
- **`nn.Linear` needs int sizes.** `mlp_ratio * width` is a float — `int()` it.

## dim or no dim

Elementwise ops take no `dim`: `gelu`, `relu`, `sigmoid`, `tanh`, `exp`,
`dropout`. Anything that normalises, reduces or compares across an axis does:
`softmax`, `sum`, `mean`, `max`, `norm`, `cat`, `argmax`. `LayerNorm` is the
middle case — told at construction, `nn.LayerNorm(width)`, always over the last
dims.

## The scale factor

`Dh ** -0.5`, negative exponent — *divide* by sqrt(head_dim). Wrote `** 0.5`
first: with Dh=64 that is multiplying by 8 instead of dividing by 8, a factor
of 64. Softmax saturates, gradients vanish.

**Shapes are unchanged, so every shape test passes.** This is the category that
costs time: right shape, wrong numbers, no exception. The other two members so
far are the head reshape order and mask polarity. Tests written before the code
are the only thing that catches them.
