#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRML Transformer positional-mechanism experiments.

This script implements encoder-only synthetic experiments for studying
positional encodings and positional injection sites:
  - no_pe
  - sinusoidal
  - learned
  - scalar
  - random
  - qk_sin
  - v_sin
  - layerwise_sin
  - gqk_pi

Tasks:
  - bag: order-insensitive classification, label = parity of token sum
  - order: order-sensitive classification on paired reversed sequences,
           label = int(first token > last token)
  - shift: token-level sequence labeling, y_i = x_{i-k}
  - reverse: token-level sequence labeling, y_i = x_{L-1-i}

Outputs:
  - CSV metrics under --out_dir
  - loss curves and extrapolation curves
  - positional heatmaps and gate heatmaps where applicable
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PAD_ID = 0
IGNORE_INDEX = -100


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # faster; set True if exact determinism is required


# -----------------------------
# Positional encodings
# -----------------------------

class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, L]
        max_pos = int(positions.max().item())
        if max_pos >= self.pe.size(0):
            raise ValueError(f"Position {max_pos} exceeds sinusoidal max_len={self.pe.size(0)}")
        return self.pe[positions]


class LearnedPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        max_pos = int(positions.max().item())
        if max_pos >= self.emb.num_embeddings:
            raise ValueError(f"Position {max_pos} exceeds learned max_len={self.emb.num_embeddings}")
        return self.emb(positions)


class ScalarPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.max_len = max_len
        self.proj = nn.Linear(1, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        x = positions.float().unsqueeze(-1) / max(1, self.max_len - 1)
        return self.proj(x)


class RandomFixedPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, std: float = 0.02):
        super().__init__()
        table = torch.randn(max_len, d_model) * std
        self.register_buffer("table", table, persistent=True)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        max_pos = int(positions.max().item())
        if max_pos >= self.table.size(0):
            raise ValueError(f"Position {max_pos} exceeds random max_len={self.table.size(0)}")
        return self.table[positions]


# -----------------------------
# Model modules
# -----------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttentionWithPosition(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        variant: str,
        pos_groups: int = 8,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % pos_groups != 0:
            raise ValueError("d_model must be divisible by pos_groups")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.variant = variant
        self.pos_groups = pos_groups

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        # Used by qk_sin / v_sin.
        self.q_pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_pos_proj = nn.Linear(d_model, d_model, bias=False)

        # Used by Gated QK Positional Injection.
        # Gate is layer-head-frequency-band specific.
        self.gate_logits = nn.Parameter(torch.zeros(n_heads, pos_groups))
        self.gqk_pos_q = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        self.gqk_pos_k = nn.Parameter(torch.empty(n_heads, d_model, self.d_head))
        nn.init.xavier_uniform_(self.gqk_pos_q)
        nn.init.xavier_uniform_(self.gqk_pos_k)

        self.last_attn: Optional[torch.Tensor] = None

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, H, L, Dh]
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def _gated_pos_per_head(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [B, L, D]
        # return [B, H, L, D]
        bsz, seq_len, d_model = pos.shape
        group_dim = d_model // self.pos_groups
        pos_grouped = pos.view(bsz, seq_len, self.pos_groups, group_dim)
        gate = torch.sigmoid(self.gate_logits)  # [H, G]
        gated = pos_grouped.unsqueeze(1) * gate.view(1, self.n_heads, 1, self.pos_groups, 1)
        return gated.reshape(bsz, self.n_heads, seq_len, d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> torch.Tensor:
        # x: [B, L, D]
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.variant == "qk_sin":
            if pos is None:
                raise ValueError("qk_sin requires sinusoidal position tensor")
            q = q + self._reshape_heads(self.q_pos_proj(pos))
            k = k + self._reshape_heads(self.k_pos_proj(pos))
        elif self.variant == "v_sin":
            if pos is None:
                raise ValueError("v_sin requires sinusoidal position tensor")
            v = v + self._reshape_heads(self.v_pos_proj(pos))
        elif self.variant == "gqk_pi":
            if pos is None:
                raise ValueError("gqk_pi requires sinusoidal position tensor")
            pos_h = self._gated_pos_per_head(pos)  # [B, H, L, D]
            q = q + torch.einsum("bhld,hdf->bhlf", pos_h, self.gqk_pos_q)
            k = k + torch.einsum("bhld,hdf->bhlf", pos_h, self.gqk_pos_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True for PAD positions.
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        out = self.out_drop(self.o_proj(out))
        if need_weights:
            self.last_attn = attn.detach().cpu()
        return out

    def gate_values(self) -> Optional[np.ndarray]:
        if self.variant != "gqk_pi":
            return None
        return torch.sigmoid(self.gate_logits.detach()).cpu().numpy()


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        variant: str,
        pos_groups: int = 8,
    ):
        super().__init__()
        self.variant = variant
        self.attn = MultiHeadSelfAttentionWithPosition(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            variant=variant,
            pos_groups=pos_groups,
        )
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        need_weights: bool = False,
    ) -> torch.Tensor:
        # Original paper uses residual + layer norm around each sublayer.
        # For layerwise_sin, inject position before every attention sublayer.
        attn_input = x + pos if (self.variant == "layerwise_sin" and pos is not None) else x
        x = self.norm1(x + self.drop(self.attn(attn_input, key_padding_mask, pos, need_weights=need_weights)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        task: str,
        variant: str,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 2048,
        pos_groups: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.task = task
        self.variant = variant
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.sin_pos = SinusoidalPosition(d_model, max_len=max_len)
        self.learned_pos = LearnedPosition(d_model, max_len=max_len)
        self.scalar_pos = ScalarPosition(d_model, max_len=max_len)
        self.random_pos = RandomFixedPosition(d_model, max_len=max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                variant=variant,
                pos_groups=pos_groups,
            )
            for _ in range(n_layers)
        ])

        if task in {"bag", "order"}:
            self.head = nn.Linear(d_model, 2)
        elif task in {"shift", "reverse"}:
            self.head = nn.Linear(d_model, vocab_size)
        else:
            raise ValueError(f"Unknown task: {task}")

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "token_emb" not in name:
                nn.init.xavier_uniform_(p)

    def _positions(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        return torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)

    def _position_tensor(self, positions: torch.Tensor) -> Optional[torch.Tensor]:
        if self.variant in {"sinusoidal", "qk_sin", "v_sin", "layerwise_sin", "gqk_pi"}:
            return self.sin_pos(positions)
        if self.variant == "learned":
            return self.learned_pos(positions)
        if self.variant == "scalar":
            return self.scalar_pos(positions)
        if self.variant == "random":
            return self.random_pos(positions)
        if self.variant == "no_pe":
            return None
        raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        # x: [B, L]
        key_padding_mask = x.eq(PAD_ID)
        positions = self._positions(x)
        pos = self._position_tensor(positions)

        h = self.token_emb(x) * math.sqrt(self.d_model)
        if self.variant in {"sinusoidal", "learned", "scalar", "random"}:
            h = h + pos

        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask, pos=pos, need_weights=need_weights)

        if self.task in {"bag", "order"}:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            return self.head(pooled)
        return self.head(h)

    def save_gate_heatmaps(self, out_dir: str, prefix: str) -> None:
        if self.variant != "gqk_pi":
            return
        os.makedirs(out_dir, exist_ok=True)
        all_gates = []
        for i, layer in enumerate(self.layers):
            g = layer.attn.gate_values()
            if g is not None:
                all_gates.append(g)
                np.save(os.path.join(out_dir, f"{prefix}_layer{i}_gqk_gates.npy"), g)
                if plt is not None:
                    plt.figure(figsize=(6, 3.5))
                    plt.imshow(g, aspect="auto")
                    plt.xlabel("frequency band")
                    plt.ylabel("head")
                    plt.title(f"GQK-PI gate values, layer {i}")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"{prefix}_layer{i}_gqk_gates.png"), dpi=200)
                    plt.close()
        if all_gates:
            np.save(os.path.join(out_dir, f"{prefix}_all_gqk_gates.npy"), np.stack(all_gates, axis=0))


# -----------------------------
# Data generation
# -----------------------------

@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    lengths: torch.Tensor


def _sample_sequence(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    # Real tokens are 1..vocab_size-1. 0 is PAD.
    return torch.randint(1, vocab_size, (length,), device=device)


def generate_batch(
    task: str,
    batch_size: int,
    length_min: int,
    length_max: int,
    vocab_size: int,
    device: torch.device,
    fixed_length: Optional[int] = None,
    shift_k: int = 3,
) -> Batch:
    if fixed_length is not None:
        lengths = torch.full((batch_size,), fixed_length, dtype=torch.long, device=device)
        max_len = fixed_length
    else:
        low = max(length_min, shift_k + 1 if task == "shift" else length_min)
        lengths = torch.randint(low, length_max + 1, (batch_size,), device=device)
        max_len = int(lengths.max().item())

    x = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long, device=device)

    if task in {"bag", "order"}:
        y = torch.zeros((batch_size,), dtype=torch.long, device=device)
    elif task in {"shift", "reverse"}:
        y = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
    else:
        raise ValueError(f"Unknown task: {task}")

    if task == "order":
        # Use reversed pairs as much as possible.
        # Label is first_token > last_token. Reversing flips the label.
        i = 0
        while i < batch_size:
            l = int(lengths[i].item())
            seq = _sample_sequence(l, vocab_size, device)
            # Avoid ambiguous first == last cases.
            tries = 0
            while seq[0].item() == seq[-1].item() and tries < 20:
                seq = _sample_sequence(l, vocab_size, device)
                tries += 1
            x[i, :l] = seq
            y[i] = int(seq[0].item() > seq[-1].item())
            if i + 1 < batch_size:
                lengths[i + 1] = l
                rev = torch.flip(seq, dims=[0])
                x[i + 1, :l] = rev
                y[i + 1] = int(rev[0].item() > rev[-1].item())
            i += 2
        return Batch(x=x, y=y, lengths=lengths)

    for i in range(batch_size):
        l = int(lengths[i].item())
        seq = _sample_sequence(l, vocab_size, device)
        x[i, :l] = seq
        if task == "bag":
            y[i] = int(seq.sum().item() % 2 == 0)
        elif task == "shift":
            # y_i = x_{i-k}; first k positions ignored.
            if l > shift_k:
                y[i, shift_k:l] = seq[: l - shift_k]
        elif task == "reverse":
            y[i, :l] = torch.flip(seq, dims=[0])

    return Batch(x=x, y=y, lengths=lengths)


# -----------------------------
# Training and evaluation
# -----------------------------

def noam_lr(step: int, d_model: int, warmup: int, factor: float) -> float:
    step = max(step, 1)
    return factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_loss(model: nn.Module, logits: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task in {"bag", "order"}:
        return F.cross_entropy(logits, y)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(
    model: PositionalTransformerEncoder,
    task: str,
    length: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int,
    n_batches: int,
    shift_k: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    total_tokens = 0
    exact_correct = 0
    exact_total = 0

    for _ in range(n_batches):
        batch = generate_batch(
            task=task,
            batch_size=batch_size,
            length_min=length,
            length_max=length,
            vocab_size=vocab_size,
            device=device,
            fixed_length=length,
            shift_k=shift_k,
        )
        logits = model(batch.x)
        loss = compute_loss(model, logits, batch.y, task)
        total_loss += float(loss.item())

        if task in {"bag", "order"}:
            pred = logits.argmax(dim=-1)
            correct += int((pred == batch.y).sum().item())
            total_items += batch.y.numel()
        else:
            pred = logits.argmax(dim=-1)
            mask = batch.y.ne(IGNORE_INDEX)
            correct += int(((pred == batch.y) & mask).sum().item())
            total_tokens += int(mask.sum().item())
            # Exact match over non-ignored positions per sample.
            for i in range(batch.x.size(0)):
                m = mask[i]
                if int(m.sum().item()) == 0:
                    continue
                exact_total += 1
                exact_correct += int(torch.equal(pred[i][m], batch.y[i][m]))

    if task in {"bag", "order"}:
        return {
            "loss": total_loss / n_batches,
            "acc": correct / max(1, total_items),
            "token_acc": float("nan"),
            "exact_match": float("nan"),
        }
    return {
        "loss": total_loss / n_batches,
        "acc": float("nan"),
        "token_acc": correct / max(1, total_tokens),
        "exact_match": exact_correct / max(1, exact_total),
    }


def train_one(
    args: argparse.Namespace,
    task: str,
    variant: str,
    seed: int,
) -> Tuple[List[Dict[str, float]], Dict[int, Dict[str, float]]]:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = PositionalTransformerEncoder(
        vocab_size=args.vocab_size,
        task=task,
        variant=variant,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pos_groups=args.pos_groups,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    train_log: List[Dict[str, float]] = []
    model.train()
    for step in range(1, args.steps + 1):
        lr = noam_lr(step, args.d_model, args.warmup, args.lr_factor)
        set_optimizer_lr(optimizer, lr)
        batch = generate_batch(
            task=task,
            batch_size=args.batch_size,
            length_min=args.train_len_min,
            length_max=args.train_len_max,
            vocab_size=args.vocab_size,
            device=device,
            fixed_length=None,
            shift_k=args.shift_k,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x)
        loss = compute_loss(model, logits, batch.y, task)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "lr": lr,
                "task": task,
                "variant": variant,
                "seed": seed,
            }
            train_log.append(rec)
            print(json.dumps(rec, ensure_ascii=False))

    eval_by_len: Dict[int, Dict[str, float]] = {}
    for length in args.eval_lengths:
        metrics = evaluate(
            model=model,
            task=task,
            length=length,
            vocab_size=args.vocab_size,
            device=device,
            batch_size=args.eval_batch_size,
            n_batches=args.eval_batches,
            shift_k=args.shift_k,
        )
        metrics.update({"task": task, "variant": variant, "seed": seed, "length": length})
        eval_by_len[length] = metrics
        print("EVAL", json.dumps(metrics, ensure_ascii=False))

    prefix = f"{task}_{variant}_seed{seed}"
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.save_checkpoints:
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "task": task,
                "variant": variant,
                "seed": seed,
            },
            os.path.join(ckpt_dir, f"{prefix}.pt"),
        )
    model.save_gate_heatmaps(os.path.join(args.out_dir, "figures"), prefix)
    save_pe_heatmap(model, args, task, variant, seed)
    return train_log, eval_by_len


# -----------------------------
# Output and plots
# -----------------------------

def write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_pe_heatmap(model: PositionalTransformerEncoder, args: argparse.Namespace, task: str, variant: str, seed: int) -> None:
    if plt is None:
        return
    if variant not in {"sinusoidal", "learned", "scalar", "random", "qk_sin", "v_sin", "layerwise_sin", "gqk_pi"}:
        return
    device = next(model.parameters()).device
    positions = torch.arange(min(args.plot_len, args.max_len), device=device).unsqueeze(0)
    with torch.no_grad():
        if variant == "learned":
            pe = model.learned_pos(positions)[0].detach().cpu().numpy()
        elif variant == "scalar":
            pe = model.scalar_pos(positions)[0].detach().cpu().numpy()
        elif variant == "random":
            pe = model.random_pos(positions)[0].detach().cpu().numpy()
        else:
            pe = model.sin_pos(positions)[0].detach().cpu().numpy()
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(pe, aspect="auto")
    plt.xlabel("dimension")
    plt.ylabel("position")
    plt.title(f"PE heatmap: {task}/{variant}/seed{seed}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{task}_{variant}_seed{seed}_pe_heatmap.png"), dpi=200)
    plt.close()


def plot_summary(args: argparse.Namespace, eval_rows: List[Dict], train_rows: List[Dict]) -> None:
    if plt is None:
        print("matplotlib is unavailable; skip plotting.")
        return
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Loss curves.
    groups: Dict[Tuple[str, str, int], List[Dict]] = {}
    for r in train_rows:
        groups.setdefault((r["task"], r["variant"], r["seed"]), []).append(r)
    for task in args.tasks:
        plt.figure(figsize=(8, 5))
        for (t, variant, seed), rows in groups.items():
            if t != task:
                continue
            rows = sorted(rows, key=lambda x: x["step"])
            plt.plot([r["step"] for r in rows], [r["loss"] for r in rows], label=f"{variant}/s{seed}")
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.title(f"Training loss: {task}")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{task}_training_loss.png"), dpi=200)
        plt.close()

    # Evaluation curves by length.
    for task in args.tasks:
        metric = "acc" if task in {"bag", "order"} else "exact_match"
        plt.figure(figsize=(8, 5))
        for variant in args.variants:
            xs, ys = [], []
            for length in args.eval_lengths:
                vals = [float(r[metric]) for r in eval_rows if r["task"] == task and r["variant"] == variant and int(r["length"]) == length]
                if vals:
                    xs.append(length)
                    ys.append(float(np.mean(vals)))
            if xs:
                plt.plot(xs, ys, marker="o", label=variant)
        plt.xlabel("test length")
        plt.ylabel(metric)
        plt.title(f"Length extrapolation: {task}")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{task}_length_extrapolation.png"), dpi=200)
        plt.close()


def aggregate_eval(eval_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = {}
    for r in eval_rows:
        key = (r["task"], r["variant"], int(r["length"]))
        if key not in grouped:
            grouped[key] = {"loss": [], "acc": [], "token_acc": [], "exact_match": []}
        for m in grouped[key]:
            v = r.get(m, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if not math.isnan(v):
                grouped[key][m].append(v)
    out = []
    for (task, variant, length), vals in grouped.items():
        row = {"task": task, "variant": variant, "length": length}
        for m, arr in vals.items():
            if arr:
                row[f"{m}_mean"] = float(np.mean(arr))
                row[f"{m}_std"] = float(np.std(arr))
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
        out.append(row)
    return sorted(out, key=lambda r: (r["task"], r["length"], r["variant"]))


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="artifacts_positional")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tasks", type=str, nargs="+", default=["bag", "order", "shift", "reverse"])
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["no_pe", "scalar", "random", "learned", "sinusoidal", "qk_sin", "v_sin", "layerwise_sin", "gqk_pi"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--vocab_size", type=int, default=32)
    parser.add_argument("--train_len_min", type=int, default=16)
    parser.add_argument("--train_len_max", type=int, default=64)
    parser.add_argument("--eval_lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--shift_k", type=int, default=5)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_groups", type=int, default=8)

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--lr_factor", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--plot_len", type=int, default=128)
    parser.add_argument("--save_checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    all_train_rows: List[Dict] = []
    all_eval_rows: List[Dict] = []
    for task in args.tasks:
        for variant in args.variants:
            for seed in args.seeds:
                print(f"\n===== RUN task={task} variant={variant} seed={seed} =====")
                train_log, eval_by_len = train_one(args, task, variant, seed)
                all_train_rows.extend(train_log)
                all_eval_rows.extend(eval_by_len.values())
                write_csv(os.path.join(args.out_dir, "train_log_partial.csv"), all_train_rows)
                write_csv(os.path.join(args.out_dir, "eval_metrics_partial.csv"), all_eval_rows)

    write_csv(os.path.join(args.out_dir, "train_log.csv"), all_train_rows)
    write_csv(os.path.join(args.out_dir, "eval_metrics.csv"), all_eval_rows)
    agg = aggregate_eval(all_eval_rows)
    write_csv(os.path.join(args.out_dir, "eval_metrics_agg.csv"), agg)
    plot_summary(args, all_eval_rows, all_train_rows)

    print("\nDone.")
    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
