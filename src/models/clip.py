"""Rung 1 — CLIP from scratch. SCAFFOLD ONLY.

Radford et al., "Learning Transferable Visual Models From Natural Language
Supervision" (2021). https://arxiv.org/abs/2103.00020

Every body in this file raises NotImplementedError on purpose. The signatures,
the docstrings and the shape comments are the specification; the forward passes
are mine to write. Nothing here is filled in for me.

Shape conventions used throughout:

    B   batch size
    L   sequence length (visual: N_PATCHES; text: max_len)
    W   transformer width (per tower)
    H   number of attention heads, Dh = W // H
    D   shared embedding dim after the projection heads

Default sizes are set for the rung-0 harness: 64x64 images, patch 8, so
N_PATCHES = (64 // 8) ** 2 = 64 visual tokens, plus 1 CLS = 65.

Order of work suggested by the tests in tests/test_clip_rung1.py:
    clip_infonce_loss -> MultiHeadSelfAttention -> Block -> VisionTransformer
    -> TextTransformer -> CLIP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Temperature is stored as a log so it stays positive under unconstrained
# gradient descent. The paper initialises it to 0.07 and clamps the scale at
# 100 to stop the logits running away early in training.
INIT_TEMPERATURE: float = 0.07
MAX_LOGIT_SCALE: float = 100.0


def clip_infonce_loss(
    logits_per_image: torch.Tensor,  # (B, B)
    logits_per_text: torch.Tensor,  # (B, B)
) -> torch.Tensor:  # scalar
    """Symmetric InfoNCE: mean of image->text and text->image cross-entropy.

    The matched pair (i, i) sits on the diagonal, so the targets are simply
    `arange(B)` in both directions. Both arguments are *already* scaled by the
    learnable temperature — this function does not know about temperature.

    Property the tests pin down: if every entry of both matrices is equal (any
    constant), the softmax over each row is uniform and the loss is exactly
    ln(B). That is the value an untrained model must sit at, and a loss stuck
    at ln(B) forever is the classic symptom of the diagonal carrying no signal.

    Args:
        logits_per_image: (B, B), row i = image i against every text.
        logits_per_text:  (B, B), row j = text j against every image.

    Returns:
        Scalar loss, mean of the two directions.
    """
    B = logits_per_image.shape[0]
    labels = torch.arange(B)
    loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t)/2
    return loss 



class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention, written out rather than nn.MHA.

    Shapes through the forward pass:
        x            (B, L, W)
        qkv          (B, L, 3W)     -> split into three (B, L, W)
        per head     (B, H, L, Dh)  via reshape + transpose
        attn logits  (B, H, L, L)   = q @ k.transpose(-2, -1) * Dh ** -0.5
        attn out     (B, H, L, Dh)  = softmax(logits) @ v
        merged       (B, L, W)      -> output projection -> (B, L, W)
    """

    def __init__(self, width: int, n_heads: int, causal: bool = False) -> None:
        """
        Args:
            width: W, must be divisible by n_heads.
            n_heads: H.
            causal: if True, mask position i from attending to j > i. The text
                tower in CLIP is causal; the vision tower is not.
        """
        super().__init__()
        self.n_heads = n_heads 
        self.causal = causal 
        self.width = width
        assert (self.width %  self.n_heads == 0), "Width is not divisible by n_heads"
        self.qkv = nn.Linear(self.width , 3*self.width , bias = True)
        self.projection = nn.Linear(self.width , self.width , bias = True)




    def forward(
        self,
        x: torch.Tensor,  # (B, L, W)
        key_padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = keep
    ) -> torch.Tensor:  # (B, L, W)
        """Attend over the sequence.

        `key_padding_mask` marks real tokens as True and padding as False. It
        has to be broadcast to (B, 1, 1, L) before being applied to the
        (B, H, L, L) logits, and masked positions go to -inf *before* the
        softmax, not after.
        """
        B, L, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim = -1)
        q = q.view(B, L, self.n_heads, self.width// self.n_heads).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.width// self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.width// self.n_heads).transpose(1, 2)
        attn_logits = q @ k.transpose(-2, -1) * (self.width//self.n_heads)** (-0.5) 
        if self.causal == True:
            allowed = torch.ones(L, L, dtype=torch.bool, device = x.device).tril()
            attn_logits = attn_logits.masked_fill(~allowed, float("-inf"))
        if key_padding_mask is not None:
           key_padding_mask = key_padding_mask[:, None, None, :]
           attn_logits = attn_logits.masked_fill(~key_padding_mask, float("-inf"))

        attn_out = torch.nn.functional.softmax(attn_logits, dim = -1)@ v
        merged = attn_out.transpose(1, 2).reshape(B, L, W)
        return self.projection(merged)


class MLP(nn.Module):
    """Position-wise feed-forward: W -> mlp_ratio * W -> W, GELU between."""

    def __init__(self, width: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, int(mlp_ratio * width))
        self.fc2 = nn.Linear(int(mlp_ratio * width), width)


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, W) -> (B, L, W)
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN transformer block.

        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))

    Pre-LN, not post-LN: the residual stream stays un-normalised end to end,
    which is what makes these trainable without a warmup schedule at this size.
    """

    def __init__(
        self, width: int, n_heads: int, mlp_ratio: float = 4.0, causal: bool = False
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, n_heads, causal)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, W)
        key_padding_mask: torch.Tensor | None = None,  # (B, L)
    ) -> torch.Tensor:  # (B, L, W)
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x 



class VisionTransformer(nn.Module):
    """ViT image tower.

        images        (B, 3, 64, 64)
        patchify      (B, 64, W)      conv2d(3, W, kernel=8, stride=8) + flatten
        prepend CLS   (B, 65, W)
        + pos embed   (B, 65, W)      learned, one row per position incl. CLS
        blocks x d    (B, 65, W)
        final LN      (B, 65, W)
        take CLS      (B, W)
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        width: int = 192,
        depth: int = 6,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        raise NotImplementedError

    @property
    def n_patches(self) -> int:
        """(image_size // patch_size) ** 2 — 64 at the rung-0 defaults."""
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, width), the pooled CLS token before projection."""
        raise NotImplementedError


class TextTransformer(nn.Module):
    """Causal transformer text tower.

        tokens        (B, L)          int64, pad_id = 0
        token embed   (B, L, W)
        + pos embed   (B, L, W)
        blocks x d    (B, L, W)       causal mask AND padding mask
        final LN      (B, L, W)
        pool at EOS   (B, W)          the row at the last non-pad position

    Two masks, two different jobs, and it is worth being able to say which is
    which: the causal mask stops a position seeing the future, the padding mask
    stops every position seeing the padding. Dropping the second one is silent
    — the loss still falls, the embeddings are just contaminated by pad.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 32,
        width: int = 192,
        depth: int = 4,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        pad_id: int = 0,
    ) -> None:
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) int64 -> (B, width), pooled at the EOS position."""
        raise NotImplementedError


class CLIP(nn.Module):
    """Two towers, two projection heads, one learnable temperature.

        encode_image  (B, 3, 64, 64) -> (B, D), L2-normalised
        encode_text   (B, L)         -> (B, D), L2-normalised
        forward       -> logits_per_image (B, B), logits_per_text (B, B)

    The projections are bias-free Linear(width -> D), one per tower, and the
    L2 normalisation happens *after* the projection. Both logit matrices are
    the same similarity matrix, transposed — assert that, it catches a whole
    family of bugs for free.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        image_size: int = 64,
        patch_size: int = 8,
        vision_width: int = 192,
        vision_depth: int = 6,
        vision_heads: int = 3,
        text_width: int = 192,
        text_depth: int = 4,
        text_heads: int = 3,
        max_len: int = 32,
        pad_id: int = 0,
        init_temperature: float = INIT_TEMPERATURE,
    ) -> None:
        """Build both towers, both projections, and `logit_scale`.

        `logit_scale` is an `nn.Parameter` holding log(1 / init_temperature),
        so exp(logit_scale) is the multiplier applied to the cosine
        similarities. Store the log, not the temperature.
        """
        raise NotImplementedError

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, D), unit norm along dim=-1."""
        raise NotImplementedError

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, D), unit norm along dim=-1."""
        raise NotImplementedError

    def clamped_logit_scale(self) -> torch.Tensor:
        """exp(logit_scale), clamped to at most MAX_LOGIT_SCALE. Scalar."""
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,  # (B, 3, H, W)
        tokens: torch.Tensor,  # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits_per_image, logits_per_text), each (B, B).

        logits_per_image[i, j] = scale * cos(image_i, text_j)
        logits_per_text = logits_per_image.T
        """
        raise NotImplementedError
