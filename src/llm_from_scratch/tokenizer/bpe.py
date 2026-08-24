"""Byte-level BPE tokenizer. See docs/01-tokenization.md for the concept
walkthrough and docs/tokenizer-performance.md for the pre-tokenization /
efficient-training design."""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path

import regex

Pair = tuple[int, int]

# GPT-2-style pre-tokenization regex: splits text into word/number/
# punctuation/whitespace chunks (leading space stays attached to the
# following word) before BPE ever runs. Uses \p{L}/\p{N} (Unicode letter/
# number categories), which the third-party `regex` package supports and
# stdlib `re` does not. See docs/tokenizer-performance.md.
_PRETOKENIZE_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


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
        # special_tokens: name -> id, for tokens with no underlying bytes
        # (e.g. "eos"). See docs/eos-generation-stopping.md.
        self.special_tokens: dict[str, int] = {}
        # pretokenize: whether this tokenizer's merges were learned with
        # chunk boundaries in effect (train() always sets this True). A
        # tokenizer.json saved before this milestone has no such field, so
        # load() defaults it to False -- see docs/tokenizer-performance.md,
        # "Backward compatibility". Encode must match whichever mode
        # trained the merges, or learned merges may not apply correctly.
        self.pretokenize: bool = True

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eos_token_id(self) -> int | None:
        return self.special_tokens.get("eos")

    def train(self, corpus: str, vocab_size: int) -> None:
        """Learn merges from `corpus` until the vocabulary reaches `vocab_size`.

        `vocab_size` must be >= 256 (the base byte vocabulary).

        Pre-tokenizes `corpus` into chunks (see docs/tokenizer-performance.md)
        and learns merges only within chunks, never across a chunk boundary.
        Unique chunks are counted once, weighted by how often they occur, and
        each merge only touches the (typically few) chunks that contain the
        merged pair -- not the whole corpus -- which is what makes this
        tractable on real-sized corpora instead of O(num_merges x corpus
        length).
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the base byte vocabulary)")

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}
        self.pretokenize = True
        num_merges = vocab_size - 256

        chunks = self._pretokenize(corpus)
        chunk_counts: Counter[str] = Counter(chunks)
        # unique chunk text -> its current token-id list (starts as bytes).
        # Only unique chunks are tracked; `chunk_counts` supplies the weight
        # a repeated chunk (e.g. "the" appearing 50,000 times) contributes,
        # so it's counted once and weighted, not rescanned per occurrence.
        chunk_tokens: dict[str, list[int]] = {
            chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts
        }

        pair_counts: Counter[Pair] = Counter()
        # pair -> set of unique-chunk keys currently containing that pair,
        # so a merge only has to touch chunks that actually contain it.
        pair_chunks: dict[Pair, set[str]] = {}

        def add_chunk_pairs(chunk_key: str) -> None:
            tokens = chunk_tokens[chunk_key]
            weight = chunk_counts[chunk_key]
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += weight
                pair_chunks.setdefault(pair, set()).add(chunk_key)

        def remove_chunk_pairs(chunk_key: str) -> None:
            tokens = chunk_tokens[chunk_key]
            weight = chunk_counts[chunk_key]
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] -= weight
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                chunk_set = pair_chunks.get(pair)
                if chunk_set is not None:
                    chunk_set.discard(chunk_key)
                    if not chunk_set:
                        del pair_chunks[pair]

        for chunk_key in chunk_tokens:
            add_chunk_pairs(chunk_key)

        for i in range(num_merges):
            if not pair_counts:
                break  # no chunk has any repeated adjacent pair left
            best_pair = max(pair_counts, key=pair_counts.get)

            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Only chunks containing best_pair need to change -- everything
            # else is untouched and unrescanned.
            affected_chunks = list(pair_chunks.get(best_pair, ()))
            for chunk_key in affected_chunks:
                remove_chunk_pairs(chunk_key)
                chunk_tokens[chunk_key] = self._merge(
                    chunk_tokens[chunk_key], best_pair, new_id
                )
                add_chunk_pairs(chunk_key)

    def add_eos_token(self) -> int:
        """Reserve an id meaning "end of sequence" and return it.

        See docs/eos-generation-stopping.md. Unlike every other token, this
        id has no corresponding bytes -- it's never produced by encode() on
        real text, and decode() skips it rather than looking up bytes for
        it. The id is the next unused integer (current vocab_size), fixed
        for this tokenizer instance from here on. Call once, right after
        train(), before saving -- calling it twice would silently reassign
        the id future code has already started relying on.
        """
        if "eos" in self.special_tokens:
            raise ValueError("EOS token already added to this tokenizer.")
        eos_id = self.vocab_size
        self.special_tokens["eos"] = eos_id
        return eos_id

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids.

        Must match whichever mode trained `self.merges`
        (`self.pretokenize`, see docs/tokenizer-performance.md): a
        pre-tokenized tokenizer applies merges within each regex chunk
        independently (merges can never cross a chunk boundary, since none
        were learned that way); a legacy (non-pre-tokenized) tokenizer
        applies merges across the whole text at once, exactly as before
        this milestone.
        """
        if not self.pretokenize:
            return self._apply_merges(list(text.encode("utf-8")))
        ids: list[int] = []
        for chunk in self._pretokenize(text):
            ids.extend(self._apply_merges(list(chunk.encode("utf-8"))))
        return ids

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Repeatedly apply the earliest-learned applicable merge to `ids`."""
        while len(ids) >= 2:
            pair_counts = self._count_pairs(ids)
            # Among pairs present here, pick the one learned earliest during
            # training -- merges must apply in learned order.
            candidate = min(
                pair_counts, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break  # no learned merge applies to what's left
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        """Split `text` into GPT-2-style word/number/punctuation/whitespace
        chunks. See docs/tokenizer-performance.md."""
        return _PRETOKENIZE_PATTERN.findall(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Special-token ids (e.g. EOS) have no underlying bytes, so they're
        skipped rather than looked up -- see docs/eos-generation-stopping.md.

        `errors="replace"` because generation (Stage 7) can produce raw byte
        tokens that aren't valid UTF-8 on their own -- a real model gets
        better at avoiding this with training, but decode must not crash on
        an untrained or unlucky sequence. Encoded text always round-trips
        exactly regardless, since it never contains invalid byte sequences.
        """
        special_ids = set(self.special_tokens.values())
        raw = b"".join(self.vocab[i] for i in ids if i not in special_ids)
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
            "special_tokens": dict(self.special_tokens),
            "pretokenize": self.pretokenize,
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
        # .get(...) with a default: files saved before this milestone have
        # no populated special_tokens key (always an empty dict), which is
        # exactly the legacy case this defaults to.
        tokenizer.special_tokens = dict(data.get("special_tokens", {}))
        # .get(...) with a default of False: a file saved before this
        # milestone has no "pretokenize" key at all, which means its merges
        # were learned by the old whole-text algorithm -- encode() must keep
        # using that same algorithm for such a file, not the new chunked
        # one, or a previously-correct id sequence could silently change.
        # See docs/tokenizer-performance.md, "Backward compatibility".
        tokenizer.pretokenize = bool(data.get("pretokenize", False))
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
