"""GPT-style decoder. See docs/03-architecture.md for the concept walkthrough."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention.

    Splits n_embd into n_head chunks, runs scaled dot-product attention on
    each independently (masked so a token can't see future positions), then
    concatenates the results back to n_embd.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout

        # Precomputed causal mask: position i may attend to positions <= i.
        # Registered as a buffer (not a parameter) so it moves with .to(device)
        # but isn't trained or saved as a learnable weight.
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.bool(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, n_embd = x.shape

        q, k, v = self.qkv_proj(x).split(n_embd, dim=2)
        # (B, T, n_embd) -> (B, n_head, T, head_dim) so attention runs per head
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Per-token MLP: expand to 4x, non-linearity, project back down."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: pre-norm attention, then pre-norm feed-forward, both residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """Minimal GPT-style decoder: embeddings -> N transformer blocks -> LM head.

    Weight tying + GPT-style initialization -- see
    docs/weight-tying-initialization.md. There is no separate `lm_head`
    parameter: the final projection to vocab-sized logits reuses
    `token_embedding.weight` directly (`F.linear(x, self.token_embedding.weight)`
    in `forward`), so the same matrix serves as both the input embedding
    table and the output projection.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_final = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)
        # The two projections that write directly onto the residual stream
        # (one per sub-layer per block) get extra scaling so the stream's
        # variance doesn't grow with depth -- see
        # docs/weight-tying-initialization.md, "Why residual/output
        # projections get 1 / sqrt(2 * n_layer) scaling". Applied after
        # self.apply() so it adjusts (not replaces) the base 0.02 init.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff.net[2].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-style init: N(0, 0.02^2) for weights, zero for biases.

        See docs/weight-tying-initialization.md. Applied via
        `self.apply(...)`, so it runs once per submodule, including the
        two `nn.Embedding` tables. LayerNorm keeps its PyTorch default
        (weight=1, bias=0), so it's deliberately not handled here.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """idx: (batch, seq_len) token ids. Returns (logits, loss).

        loss is None unless `targets` (same shape as idx) is given.
        """
        batch_size, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        # Tied output projection: reuse token_embedding.weight instead of a
        # separate lm_head parameter. See docs/weight-tying-initialization.md.
        logits = F.linear(x, self.token_embedding.weight)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Extend `idx` (batch, seq_len) by up to `max_new_tokens` tokens.

        At each step: truncate to the last `context_length` tokens, run a
        forward pass, and pick the next token from the last position's
        logits. `temperature <= 0` picks the highest-probability token
        (greedy, deterministic). `temperature > 0` divides logits by
        temperature, softmaxes, and samples from that distribution.

        If `eos_token_id` is given and that id is produced, generation
        stops immediately (before `max_new_tokens` if it happens sooner) --
        see docs/eos-generation-stopping.md. `max_new_tokens` still applies
        as a hard cap regardless: if EOS is never produced, generation runs
        the full `max_new_tokens` exactly as when `eos_token_id` is None.

        Temporarily switches the model to eval mode (so dropout is off
        during generation) and restores whatever mode it was in before the
        call once generation finishes -- see
        docs/generation-mode-restoration.md.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.context_length :]
                logits, _ = self(idx_cond)
                next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
                if temperature <= 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        finally:
            self.train(was_training)
        return idx
