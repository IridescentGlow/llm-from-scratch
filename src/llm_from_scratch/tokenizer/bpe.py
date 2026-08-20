"""Byte-level BPE tokenizer. See docs/01-tokenization.md for the concept walkthrough."""

from __future__ import annotations

from collections import Counter

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
        """Decode a list of token ids back into text."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8")

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
