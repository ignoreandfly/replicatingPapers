"""Rung 1 tests. These FAIL right now, on purpose.

src/models/clip.py is a scaffold; every body raises NotImplementedError. This
file is the specification I am writing the model against, and the order below
is roughly the order worth implementing in:

    1. clip_infonce_loss        test_infonce_*        (no model needed)
    2. MultiHeadSelfAttention   test_attention_*
    3. VisionTransformer        test_vision_*
    4. TextTransformer          test_text_*
    5. CLIP                     test_clip_*
    6. everything together      test_overfit_8

Run just this file:      make test-rung1
Run everything else:     make test-harness
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.models.clip import (
    INIT_TEMPERATURE,
    MAX_LOGIT_SCALE,
    CLIP,
    Block,
    MultiHeadSelfAttention,
    TextTransformer,
    VisionTransformer,
    clip_infonce_loss,
)

pytestmark = pytest.mark.rung1

# Rung-0 harness geometry: 64x64, patch 8 -> 64 visual tokens.
IMAGE_SIZE, PATCH, N_PATCHES = 64, 8, 64
VOCAB, MAX_LEN, EMBED_DIM = 64, 16, 32
WIDTH, HEADS = 48, 4


def tiny_clip(**kw) -> CLIP:
    cfg = dict(
        vocab_size=VOCAB,
        embed_dim=EMBED_DIM,
        image_size=IMAGE_SIZE,
        patch_size=PATCH,
        vision_width=WIDTH,
        vision_depth=2,
        vision_heads=HEADS,
        text_width=WIDTH,
        text_depth=2,
        text_heads=HEADS,
        max_len=MAX_LEN,
    )
    cfg.update(kw)
    return CLIP(**cfg)


def fake_batch(b: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    images = torch.rand(b, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    tokens = torch.randint(1, VOCAB, (b, MAX_LEN), generator=g)
    tokens[:, -3:] = 0  # pad tail, pad_id == 0
    return images, tokens


# --- the loss ---------------------------------------------------------------


@pytest.mark.parametrize("b", [2, 8, 64])
def test_infonce_equals_ln_batch_size_on_uniform_logits(b: int) -> None:
    """Constant logits -> uniform softmax -> loss is exactly ln(B).

    This is the value an untrained CLIP sits at, and the number to check the
    very first training step against.
    """
    logits = torch.zeros(b, b)
    loss = clip_infonce_loss(logits, logits.t())
    assert loss.shape == ()
    assert loss.item() == pytest.approx(math.log(b), abs=1e-5)


def test_infonce_is_invariant_to_a_constant_shift() -> None:
    """Adding a constant to every logit changes nothing (softmax shift-invariance)."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(8, 8, generator=g)
    a = clip_infonce_loss(logits, logits.t())
    b = clip_infonce_loss(logits + 7.5, (logits + 7.5).t())
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


def test_infonce_is_zero_when_the_diagonal_dominates() -> None:
    """A perfectly separated similarity matrix drives the loss to ~0."""
    logits = torch.eye(8) * 50.0
    assert clip_infonce_loss(logits, logits.t()).item() == pytest.approx(0.0, abs=1e-4)


def test_infonce_is_symmetric_in_its_two_directions() -> None:
    """Swapping the two arguments swaps the two directions, same mean."""
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(6, 6, generator=g)
    a = clip_infonce_loss(logits, logits.t())
    b = clip_infonce_loss(logits.t(), logits)
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_infonce_at_random_init_is_near_ln_batch_size() -> None:
    """The statistical version: a real untrained model starts near ln(B).

    Cosine similarities of random unit vectors concentrate near 0, so the
    scaled logits are near-constant and the loss lands close to ln(B). The
    tolerance is loose because the temperature multiplies whatever spread the
    random features do have.
    """
    torch.manual_seed(0)
    model = tiny_clip()
    images, tokens = fake_batch(64)
    with torch.no_grad():
        loss = clip_infonce_loss(*model(images, tokens))
    assert loss.item() == pytest.approx(math.log(64), rel=0.15)


# --- attention --------------------------------------------------------------


def test_attention_shape_is_preserved() -> None:
    attn = MultiHeadSelfAttention(width=WIDTH, n_heads=HEADS)
    x = torch.randn(3, 10, WIDTH)
    assert attn(x).shape == (3, 10, WIDTH)


def test_attention_rejects_width_not_divisible_by_heads() -> None:
    with pytest.raises((AssertionError, ValueError)):
        MultiHeadSelfAttention(width=50, n_heads=4)


def test_causal_attention_cannot_see_the_future() -> None:
    """Perturbing token t must not change the output at any position < t."""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(width=WIDTH, n_heads=HEADS, causal=True)
    x = torch.randn(1, 8, WIDTH)
    with torch.no_grad():
        base = attn(x)
        perturbed = x.clone()
        perturbed[0, 5] += 10.0
        after = attn(perturbed)
    assert torch.allclose(base[0, :5], after[0, :5], atol=1e-5)
    assert not torch.allclose(base[0, 5:], after[0, 5:], atol=1e-5)


def test_padding_mask_changes_the_output() -> None:
    """Masked-out keys must not contribute. If they do, the mask is a no-op."""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(width=WIDTH, n_heads=HEADS)
    x = torch.randn(2, 6, WIDTH)
    keep = torch.ones(2, 6, dtype=torch.bool)
    keep[:, 4:] = False
    with torch.no_grad():
        masked = attn(x, key_padding_mask=keep)
        # changing only the masked-out positions must not move the kept outputs
        y = x.clone()
        y[:, 4:] += 100.0
        masked_again = attn(y, key_padding_mask=keep)
    assert torch.allclose(masked[:, :4], masked_again[:, :4], atol=1e-4)
    assert not torch.allclose(masked[:, :4], attn(x)[:, :4], atol=1e-4)


def test_block_preserves_shape_and_does_something() -> None:
    torch.manual_seed(0)
    block = Block(width=WIDTH, n_heads=HEADS)
    x = torch.randn(2, 7, WIDTH)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x, atol=1e-6), "block is the identity"


# --- towers -----------------------------------------------------------------


def test_vision_transformer_shape_and_patch_count() -> None:
    vit = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH, width=WIDTH, depth=2, n_heads=HEADS
    )
    assert vit.n_patches == N_PATCHES
    out = vit(torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (3, WIDTH), "vision tower must pool to one vector per image"


def test_vision_transformer_is_not_permutation_invariant() -> None:
    """Shuffling the image must change the embedding, or the position
    embeddings are not doing anything."""
    torch.manual_seed(0)
    vit = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH, width=WIDTH, depth=2, n_heads=HEADS
    )
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        a = vit(x)
        b = vit(torch.flip(x, dims=[3]))
    assert not torch.allclose(a, b, atol=1e-4)


def test_text_transformer_shape() -> None:
    txt = TextTransformer(
        vocab_size=VOCAB, max_len=MAX_LEN, width=WIDTH, depth=2, n_heads=HEADS
    )
    tokens = torch.randint(1, VOCAB, (4, MAX_LEN))
    tokens[:, -2:] = 0
    assert txt(tokens).shape == (4, WIDTH)


def test_text_transformer_pools_at_eos_and_ignores_what_follows() -> None:
    """Nothing after the EOS position may affect the pooled embedding.

    Written as junk *non-pad* ids after EOS rather than as extra padding, which
    makes it a real probe instead of a tautology: the causal mask alone already
    guarantees that positions after EOS cannot reach the EOS row, so the only
    way to fail this is to pool over the whole sequence (mean-pooling, or
    last-position pooling) instead of at EOS.

    The companion failure — pooling correctly but forgetting the padding mask —
    is invisible here by construction, and is the reason the padding mask is
    tested at the attention level instead.
    """
    torch.manual_seed(0)
    txt = TextTransformer(
        vocab_size=VOCAB, max_len=MAX_LEN, width=WIDTH, depth=2, n_heads=HEADS
    )
    a = torch.zeros(1, MAX_LEN, dtype=torch.long)
    a[0, :5] = torch.tensor([1, 7, 9, 11, 2])  # 1=bos ... 2=eos, then pad
    b = a.clone()
    b[0, 5:] = torch.randint(3, VOCAB, (MAX_LEN - 5,))
    with torch.no_grad():
        assert torch.allclose(txt(a), txt(b), atol=1e-6)


# --- the full model ---------------------------------------------------------


def test_clip_encode_shapes_and_normalisation() -> None:
    model = tiny_clip()
    images, tokens = fake_batch(5)
    with torch.no_grad():
        img_f = model.encode_image(images)
        txt_f = model.encode_text(tokens)
    assert img_f.shape == (5, EMBED_DIM)
    assert txt_f.shape == (5, EMBED_DIM)
    assert torch.allclose(img_f.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(txt_f.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_clip_forward_logits_are_transposes() -> None:
    model = tiny_clip()
    images, tokens = fake_batch(6)
    with torch.no_grad():
        lpi, lpt = model(images, tokens)
    assert lpi.shape == (6, 6) and lpt.shape == (6, 6)
    assert torch.allclose(lpi, lpt.t(), atol=1e-6)


def test_clip_logits_are_bounded_by_the_logit_scale() -> None:
    """Cosine similarity is in [-1, 1], so |logit| <= scale."""
    model = tiny_clip()
    images, tokens = fake_batch(6)
    with torch.no_grad():
        lpi, _ = model(images, tokens)
        scale = model.clamped_logit_scale()
    assert lpi.abs().max().item() <= scale.item() + 1e-4


def test_logit_scale_is_a_learnable_log_and_starts_at_the_paper_value() -> None:
    model = tiny_clip()
    assert isinstance(model.logit_scale, torch.nn.Parameter)
    assert model.logit_scale.requires_grad
    assert model.logit_scale.item() == pytest.approx(math.log(1 / INIT_TEMPERATURE), abs=1e-4)
    assert model.clamped_logit_scale().item() == pytest.approx(1 / INIT_TEMPERATURE, rel=1e-4)


def test_logit_scale_is_clamped() -> None:
    model = tiny_clip()
    with torch.no_grad():
        model.logit_scale.fill_(math.log(10_000.0))
    assert model.clamped_logit_scale().item() == pytest.approx(MAX_LOGIT_SCALE, rel=1e-4)


def test_gradients_reach_every_parameter() -> None:
    """Every parameter gets a gradient. Catches a tower wired in but unused."""
    model = tiny_clip()
    images, tokens = fake_batch(8)
    clip_infonce_loss(*model(images, tokens)).backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


# --- the definition of done -------------------------------------------------


@pytest.mark.slow
def test_overfit_8() -> None:
    """8 pairs, 400 steps, loss from ln(8) to ~0 and 8/8 retrieval both ways.

    First box on CLAUDE.md's definition-of-done list. Runs on real rung-0 data,
    so `make data` has to have been run.
    """
    data = Path("data/shapes")
    if not (data / "train_images.npy").exists():
        pytest.skip("run `make data` first")

    from src.train.overfit8 import overfit

    result = overfit(data, n=8, steps=400, lr=1e-3, seed=0, device=torch.device("cpu"), verbose=False)

    assert result["loss_initial"] == pytest.approx(math.log(8), rel=0.15), (
        "an untrained CLIP should start at ln(8)"
    )
    assert result["loss_final"] < 0.05, f"did not overfit: final loss {result['loss_final']:.4f}"
    assert result["i2t_at_1"] == 8
    assert result["t2i_at_1"] == 8
