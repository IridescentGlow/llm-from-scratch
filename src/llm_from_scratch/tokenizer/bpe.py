"""Byte-level BPE tokenizer. See docs/01-tokenization.md for the concept walkthrough."""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path

Pair = tuple[int, int]


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer.

    Vocabulary starts as the 256 possible byte values, then grows by
    `train()` merging the most frequent adjacent token pair, one at a time.
    """

    def __init__(self) -> None:
        # merges: pair -> new token id, in the order they were learned.
        # Order matters at encode time: earlier merges must be applied first.
        self.merges: dict[Pair, int] = {}
        # vocab: token id -> the raw bytes it represents.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, corpus: str, vocab_size: int) -> None:
        """Learn merges from `corpus` until the vocabulary reaches `vocab_size`.

        `vocab_size` must be >= 256 (the base byte vocabulary).
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the base byte vocabulary)")

        ids = list(corpus.encode("utf-8"))
        num_merges = vocab_size - 256

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pair_counts = self._count_pairs(ids)
            if not pair_counts:
                break  # corpus fully collapsed to one token; nothing left to merge
            best_pair = max(pair_counts, key=pair_counts.get)

            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = self._merge(ids, best_pair, new_id)

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._count_pairs(ids)
            # Among pairs present in this text, pick the one learned earliest
            # during training -- merges must apply in learned order.
            candidate = min(
                pair_counts, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break  # no learned merge applies to what's left
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        `errors="replace"` because generation (Stage 7) can produce raw byte
        tokens that aren't valid UTF-8 on their own -- a real model gets
        better at avoiding this with training, but decode must not crash on
        an untrained or unlucky sequence. Encoded text always round-trips
        exactly regardless, since it never contains invalid byte sequences.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Persist this exact tokenizer (merges + vocab) to a JSON file.

        See docs/01-tokenization.md, "Tokenizer persistence": a checkpoint's
        weights are only meaningful under the specific tokenizer that
        produced their training data. Saving both `merges` and `vocab` (even
        though `vocab` is derivable from `merges` + the base 256 bytes)
        avoids re-deriving anything at load time. Plain JSON, not `pickle` --
        loading a tokenizer should never risk executing arbitrary code.
        """
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "vocab": {
                str(token_id): base64.b64encode(token_bytes).decode("ascii")
                for token_id, token_bytes in self.vocab.items()
            },
            # Reserved for future additions (e.g. an end-of-text token) so
            # they don't require a second, incompatible file format.
            "special_tokens": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Reconstruct a tokenizer saved by `save()` -- no `train()` call involved."""
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        tokenizer.vocab = {
            int(token_id): base64.b64decode(encoded)
            for token_id, encoded in data["vocab"].items()
        }
        return tokenizer

    @staticmethod
    def _count_pairs(ids: list[int]) -> Counter[Pair]:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged
