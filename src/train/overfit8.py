"""`make overfit` — drive 8 fixed caption pairs to near-zero loss.

The first item in CLAUDE.md's definition of done. Until src/models/clip.py has
a real forward pass this exits on NotImplementedError, which is the correct
behaviour: the harness is ready and the model is not.

What to expect once the model is written:

  step   0   loss 2.079   (= ln 8)
  step  50   loss 1.4xx
  step 400   loss < 0.01, retrieval 8/8 both directions

A loss that starts at ln(8) and *stays* there is the diagnostic case. So is a
loss that falls while retrieval stays at 1/8 — that one means the similarity
matrix is being driven down uniformly rather than pushed onto the diagonal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from ..data.dataset import MAX_CAPTION_LEN, CaptionPairs
from ..data.tokenizer import build_from_split
from ..models.clip import CLIP, clip_infonce_loss
from .seed import pick_device, seed_everything


def overfit(
    data_dir: str | Path,
    n: int = 8,
    steps: int = 400,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    log_every: int = 50,
    verbose: bool = True,
) -> dict:
    """Train on a fixed n-example batch. Returns the run's summary metrics."""
    seed_everything(seed)
    device = device or pick_device()

    tokenizer = build_from_split(data_dir, "train")
    pairs = CaptionPairs(data_dir, "train", tokenizer, max_len=MAX_CAPTION_LEN, limit=n)

    images = torch.stack([pairs[i]["image"] for i in range(n)]).to(device)  # (n, 3, 64, 64)
    tokens = torch.stack([pairs[i]["tokens"] for i in range(n)]).to(device)  # (n, L)

    model = CLIP(vocab_size=tokenizer.vocab_size, max_len=MAX_CAPTION_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    history: list[float] = []
    for step in range(steps + 1):
        logits_per_image, logits_per_text = model(images, tokens)
        loss = clip_infonce_loss(logits_per_image, logits_per_text)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        history.append(float(loss))
        if verbose and (step % log_every == 0 or step == steps):
            i2t, t2i = _retrieval_at_1(logits_per_image)
            print(f"  step {step:>4}  loss {float(loss):.4f}   i2t {i2t}/{n}  t2i {t2i}/{n}")

    with torch.no_grad():
        logits_per_image, _ = model(images, tokens)
        i2t, t2i = _retrieval_at_1(logits_per_image)

    return {
        "n": n,
        "steps": steps,
        "loss_initial": history[0],
        "loss_final": history[-1],
        "ln_n": math.log(n),
        "i2t_at_1": i2t,
        "t2i_at_1": t2i,
        "history": history,
    }


def _retrieval_at_1(logits_per_image: torch.Tensor) -> tuple[int, int]:
    """How many of the B pairs are each other's argmax, in each direction."""
    n = logits_per_image.shape[0]
    target = torch.arange(n, device=logits_per_image.device)
    i2t = int((logits_per_image.argmax(dim=1) == target).sum())
    t2i = int((logits_per_image.t().argmax(dim=1) == target).sum())
    return i2t, t2i


def main() -> None:
    ap = argparse.ArgumentParser(description="Overfit n caption pairs with rung-1 CLIP.")
    ap.add_argument("--data", default="data/shapes", type=Path)
    ap.add_argument("--n", default=8, type=int)
    ap.add_argument("--steps", default=400, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"overfitting {args.n} pairs on {device}, target loss -> 0 (start = ln {args.n} "
          f"= {math.log(args.n):.4f})")
    try:
        result = overfit(
            args.data, n=args.n, steps=args.steps, lr=args.lr, seed=args.seed, device=device
        )
    except NotImplementedError as exc:
        raise SystemExit(
            "src/models/clip.py is still a scaffold — that is expected until you write it.\n"
            f"  first missing piece: {exc.__class__.__name__} from {_where(exc)}"
        )

    print(
        f"\nloss {result['loss_initial']:.4f} -> {result['loss_final']:.4f}   "
        f"i2t {result['i2t_at_1']}/{result['n']}   t2i {result['t2i_at_1']}/{result['n']}"
    )
    ok = result["loss_final"] < 0.05 and result["i2t_at_1"] == result["n"]
    print("PASS: rung-1 overfit criterion met" if ok else "FAIL: not overfitting")
    raise SystemExit(0 if ok else 1)


def _where(exc: BaseException) -> str:
    tb = exc.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    if tb is None:
        return "unknown"
    return f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"


if __name__ == "__main__":
    main()
