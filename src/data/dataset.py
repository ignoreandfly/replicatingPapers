"""torch Datasets over the rung-0 files.

Two views of the same data:

  `CaptionPairs` — (image, caption tokens). The contrastive view. This is what
  rung 1 (CLIP) and rung 2 (SigLIP) train on, and what rung 3 captions.

  `QADataset` — (image, question tokens, answer string, axis). The eval view.
  Answers stay as strings until the scorer sees them, because scoring is exact
  string match after normalisation, not argmax over a label index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build import load_split
from .render import to_float_chw
from .tokenizer import Tokenizer

MAX_CAPTION_LEN = 32
MAX_QUESTION_LEN = 24


class CaptionPairs(Dataset):
    """(image (3,64,64) float32, tokens (L,) int64) for contrastive training."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        tokenizer: Tokenizer,
        max_len: int = MAX_CAPTION_LEN,
        limit: int | None = None,
    ) -> None:
        images, scenes, _ = load_split(data_dir, split)
        if limit is not None:
            images, scenes = images[:limit], scenes[:limit]
        self.images = images
        self.captions = [s["caption"] for s in scenes]
        self.scene_ids = [s["scene_id"] for s in scenes]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, i: int) -> dict:
        return {
            "image": torch.from_numpy(to_float_chw(self.images[i])),  # (3, 64, 64)
            "tokens": torch.from_numpy(self.tokenizer.encode(self.captions[i], self.max_len)),  # (L,)
            "caption": self.captions[i],
            "scene_id": self.scene_ids[i],
        }


class QADataset(Dataset):
    """(image, question tokens, answer str, axis) for per-axis evaluation."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        tokenizer: Tokenizer | None = None,
        max_len: int = MAX_QUESTION_LEN,
        axis: str | None = None,
    ) -> None:
        images, _, questions = load_split(data_dir, split)
        if axis is not None:
            questions = [q for q in questions if q["axis"] == axis]
        self.images = images
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i: int) -> dict:
        q = self.questions[i]
        item = {
            "image": torch.from_numpy(to_float_chw(self.images[q["image_index"]])),  # (3, 64, 64)
            "question": q["question"],
            "answer": q["answer"],
            "answer_space": q["answer_space"],
            "axis": q["axis"],
            "template": q["template"],
            "qid": q["qid"],
        }
        if self.tokenizer is not None:
            item["tokens"] = torch.from_numpy(
                self.tokenizer.encode(q["question"], self.max_len)
            )  # (L,)
        return item


def collate_qa(batch: list[dict]) -> dict:
    """Stack tensors, keep strings and ragged answer spaces as lists."""
    out: dict = {
        "image": torch.stack([b["image"] for b in batch]),  # (B, 3, 64, 64)
        "question": [b["question"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "answer_space": [b["answer_space"] for b in batch],
        "axis": [b["axis"] for b in batch],
        "template": [b["template"] for b in batch],
        "qid": [b["qid"] for b in batch],
    }
    if "tokens" in batch[0]:
        out["tokens"] = torch.stack([b["tokens"] for b in batch])  # (B, L)
    return out


def collate_pairs(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),  # (B, 3, 64, 64)
        "tokens": torch.stack([b["tokens"] for b in batch]),  # (B, L)
        "caption": [b["caption"] for b in batch],
        "scene_id": [b["scene_id"] for b in batch],
    }


def fixed_subset(dataset: Dataset, n: int, seed: int = 0) -> list[dict]:
    """A deterministic n-example slice, for overfit-8 style sanity runs."""
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.choice(len(dataset), size=n, replace=False)
    return [dataset[int(i)] for i in sorted(idx)]
