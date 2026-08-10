"""Word-level tokenizer over the closed synthetic vocabulary.

The rung-0 corpus is a few hundred distinct words. A BPE here would add nothing
except a place for bugs to hide, and a fixed vocabulary means the text encoder
at rung 1 has an embedding table small enough to overfit 8 examples in seconds.

Special tokens are laid out so that `<pad> == 0`: padding masks are then just
`ids != 0`, and a forgotten mask shows up as a wrong number rather than a
plausible one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS: tuple[str, ...] = (PAD, BOS, EOS, UNK)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[?,.]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Tokenizer:
    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        assert self.itos[0] == PAD, "pad must be id 0"

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @classmethod
    def build(cls, texts: list[str]) -> "Tokenizer":
        """Deterministic vocabulary: specials, then the rest sorted."""
        words = sorted({w for t in texts for w in tokenize(t)})
        return cls(itos=list(SPECIALS) + [w for w in words if w not in SPECIALS])

    def encode(self, text: str, max_len: int, add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
        """str -> int64 (max_len,), right-padded with pad_id, truncated to fit.

        EOS is always the last non-pad token when the text fits, because the
        rung-1 text encoder pools at the EOS position.
        """
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in tokenize(text)]
        budget = max_len - int(add_bos) - int(add_eos)
        ids = ids[:budget]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        out = np.full((max_len,), self.pad_id, dtype=np.int64)
        out[: len(ids)] = ids
        return out

    def decode(self, ids) -> str:
        words = [self.itos[int(i)] for i in ids]
        return " ".join(w for w in words if w not in SPECIALS)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(itos=json.loads(Path(path).read_text())["itos"])


def build_from_split(data_dir: str | Path, split: str = "train") -> Tokenizer:
    """Vocabulary from the train split's captions, questions and answers."""
    from .build import load_split

    _, scenes, questions = load_split(data_dir, split)
    texts = [s["caption"] for s in scenes]
    texts += [q["question"] for q in questions]
    texts += [q["answer"] for q in questions]
    texts += [a for q in questions for a in q["answer_space"]]
    return Tokenizer.build(texts)
