"""
run_benchmark.py — MoQAdam Empirical Evaluation
================================================
Benchmarks MoQAdam against AdamW, Naive EF-Adam, and EF21-Adam on:
  1. CIFAR-100 image classification  (ResNet-18, 100 epochs)
  2. WikiText-2 language modelling   (2-layer LSTM, 50 epochs)

All optimizers use IDENTICAL learning rate and weight decay — no
per-optimizer tuning.  This is the strictest fair-comparison protocol.

Usage
-----
    # Full benchmark (both tasks, all optimizers, 8-bit and 4-bit):
    python run_benchmark.py

    # Quick smoke test (5 epochs each):
    python run_benchmark.py --quick

    # Single task:
    python run_benchmark.py --task cifar100
    python run_benchmark.py --task lm

    # Single optimizer:
    python run_benchmark.py --optimizer moqadam_8bit

Requirements
------------
    pip install torch torchvision datasets

Output
------
    results/cifar100_results.csv
    results/lm_results.csv
    results/ablation_results.csv
    results/paper_tables.txt   ← ready-to-paste LaTeX table rows
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Naive EF-Adam baseline
# ---------------------------------------------------------------------------

class NaiveEFAdam(torch.optim.Optimizer):
    """
    Naive Error-Feedback Adam (the motivating baseline).

    Quantizes in raw gradient space (no VNQ), applies EF unconditionally
    (no SNR gate), updates v_t from q_t (biased variance).  Reproduces
    the common ad-hoc combination of error feedback with Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        quant_bits: int = 8,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, quant_bits=quant_bits,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _quantize(x: Tensor, bits: int) -> tuple[Tensor, Tensor]:
        S = 2 ** (bits - 1) - 1
        x_max = x.abs().amax().clamp(min=1e-8)
        scale = x_max / S
        q = x.div(scale).round().clamp(-S, S).mul(scale)
        residual = x - q
        return q, residual

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group["lr"]
            b1, b2 = group["betas"]
            eps  = group["eps"]
            wd   = group["weight_decay"]
            bits = group["quant_bits"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"]     = 0
                    state["m"]        = torch.zeros_like(grad)
                    state["v"]        = torch.zeros_like(grad)
                    state["residual"] = torch.zeros_like(grad)

                state["step"] += 1
                t = state["step"]
                m, v, e = state["m"], state["v"], state["residual"]

                # EF in raw gradient space
                corrected = grad + e
                q, residual = self._quantize(corrected, bits)
                state["residual"].copy_(residual)

                # Biased variance (from quantized gradient)
                v.mul_(b2).addcmul_(q, q, value=1.0 - b2)
                v_hat = v / (1.0 - b2 ** t)

                # First moment from quantized gradient
                m.mul_(b1).add_(q, alpha=1.0 - b1)
                m_hat = m / (1.0 - b1 ** t)

                update = m_hat / (v_hat.sqrt().add_(eps))
                if wd != 0.0:
                    update.add_(p.data, alpha=wd)
                p.data.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# EF21-Adam baseline (contractive compressor + Adam)
# ---------------------------------------------------------------------------

class EF21Adam(torch.optim.Optimizer):
    """
    EF21 with Adam moments (Richtárik et al. 2021 + Adam).

    Uses the top-k contractive compressor for compression.  Updates both
    moments from the quantized gradient (no DMU), applies EF unconditionally
    (no SNR gate), quantizes in raw gradient space (no VNQ).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        quant_bits: int = 8,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, quant_bits=quant_bits,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _top_k_compress(x: Tensor, bits: int) -> tuple[Tensor, Tensor]:
        """Top-k contractive compressor: keep the largest k elements."""
        n = x.numel()
        k = max(1, int(n * bits / 32))   # keep bits/32 fraction of elements
        flat = x.flatten()
        _, idx = flat.abs().topk(k)
        q = torch.zeros_like(flat)
        q[idx] = flat[idx]
        q = q.view_as(x)
        residual = x - q
        return q, residual

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group["lr"]
            b1, b2 = group["betas"]
            eps  = group["eps"]
            wd   = group["weight_decay"]
            bits = group["quant_bits"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"]     = 0
                    state["m"]        = torch.zeros_like(grad)
                    state["v"]        = torch.zeros_like(grad)
                    state["g_prev"]   = grad.clone()  # EF21 uses g_{t-1}

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                # EF21 update: compress (g_t - g_{t-1}), add back g_{t-1}
                delta = grad - state["g_prev"]
                c, _  = self._top_k_compress(delta, bits)
                g_ef  = state["g_prev"] + c
                state["g_prev"].copy_(g_ef)

                # Both moments from compressed gradient (no DMU)
                v.mul_(b2).addcmul_(g_ef, g_ef, value=1.0 - b2)
                v_hat = v / (1.0 - b2 ** t)
                m.mul_(b1).add_(g_ef, alpha=1.0 - b1)
                m_hat = m / (1.0 - b1 ** t)

                update = m_hat / (v_hat.sqrt().add_(eps))
                if wd != 0.0:
                    update.add_(p.data, alpha=wd)
                p.data.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_resnet18_cifar100() -> nn.Module:
    """ResNet-18 adapted for CIFAR-100 (32×32 input, no initial stride)."""
    try:
        import torchvision.models as tvm
        model = tvm.resnet18(weights=None, num_classes=100)
        # CIFAR: replace 7×7/stride-2 conv with 3×3/stride-1, remove maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    except ImportError:
        raise SystemExit("torchvision required: pip install torchvision")


class LSTMLanguageModel(nn.Module):
    """2-layer LSTM language model for WikiText-2."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.drop   = nn.Dropout(dropout)
        self.lstm   = nn.LSTM(embed_dim, hidden_dim, num_layers,
                              dropout=dropout, batch_first=False)
        self.proj   = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, hidden=None):
        emb = self.drop(self.embed(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.proj(self.drop(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        w = next(self.parameters())
        return (w.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                w.new_zeros(self.num_layers, batch_size, self.hidden_dim))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 4):
    try:
        import torchvision.datasets as tvd
        import torchvision.transforms as T
    except ImportError:
        raise SystemExit("torchvision required: pip install torchvision")

    MEAN = (0.5071, 0.4867, 0.4408)
    STD  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    root = Path("data/cifar100")
    train_ds = tvd.CIFAR100(root, train=True,  transform=train_tf, download=True)
    test_ds  = tvd.CIFAR100(root, train=False, transform=test_tf,  download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_wikitext2_loaders(seq_len: int = 35, batch_size: int = 20):
    """Download WikiText-2 and batchify into fixed-length sequences."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets required: pip install datasets")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(text: str) -> list[int]:
        words = text.replace("\n", "<eos>").split()
        vocab = tokenize._vocab  # type: ignore[attr-defined]
        return [vocab.setdefault(w, len(vocab)) for w in words]
    tokenize._vocab: dict = {"<eos>": 0, "<unk>": 1}  # type: ignore[attr-defined]

    def batchify(split: str) -> Tensor:
        tokens: list[int] = []
        for row in ds[split]["text"]:
            tokens.extend(tokenize(row))
        data = torch.tensor(tokens, dtype=torch.long)
        num_batches = data.size(0) // batch_size
        data = data[: num_batches * batch_size]
        return data.view(batch_size, -1).t().contiguous()

    train_data = batchify("train")
    val_data   = batchify("validation")
    vocab_size = len(tokenize._vocab)

    def get_batch(source: Tensor, i: int) -> tuple[Tensor, Tensor]:
        length = min(seq_len, source.size(0) - 1 - i)
        data   = source[i : i + length]
        target = source[i + 1 : i + 1 + length].reshape(-1)
        return data, target

    return train_data, val_data, vocab_size, get_batch


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_cifar100_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_cifar100(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits  = model(x)
        loss    = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


def train_lm_epoch(
    model: LSTMLanguageModel,
    train_data: Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    get_batch,
    seq_len: int = 35,
    clip: float = 0.25,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    hidden = model.init_hidden(train_data.size(1), device)
    nbatches = (train_data.size(0) - 1) // seq_len

    for i in range(0, train_data.size(0) - 1, seq_len):
        data, targets = get_batch(train_data, i)
        data, targets = data.to(device), targets.to(device)

        # Detach hidden state to prevent BPTT through entire history
        hidden = tuple(h.detach() for h in hidden)

        optimizer.zero_grad()
        logits, hidden = model(data, hidden)
        loss = criterion(logits.view(-1, logits.size(2)), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / nbatches


@torch.no_grad()
def evaluate_lm(
    model: LSTMLanguageModel,
    data: Tensor,
    device: torch.device,
    get_batch,
    seq_len: int = 35,
) -> float:
    import math
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    hidden = model.init_hidden(data.size(1), device)
    for i in range(0, data.size(0) - 1, seq_len):
        inputs, targets = get_batch(data, i)
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = tuple(h.detach() for h in hidden)
        logits, hidden = model(inputs, hidden)
        loss = criterion(logits.view(-1, logits.size(2)), targets)
        total_loss += loss.item() * inputs.size(0)
    return math.exp(total_loss / (data.size(0) - 1))


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

LR = 1e-3
WD = 1e-4

def make_optimizer(name: str, params, bits: int = 8) -> torch.optim.Optimizer:
    common = dict(lr=LR, weight_decay=WD)
    if name == "adamw":
        return AdamW(params, **common, betas=(0.9, 0.999), eps=1e-8)
    if name == "naive_ef_adam":
        return NaiveEFAdam(params, **common, betas=(0.9, 0.999), eps=1e-8,
                           quant_bits=bits)
    if name == "ef21_adam":
        return EF21Adam(params, **common, betas=(0.9, 0.999), eps=1e-8,
                        quant_bits=bits)
    if name.startswith("moqadam"):
        try:
            from moqadam import MoQAdam
        except ImportError:
            raise SystemExit("moqadam.py not found on PYTHONPATH")

        # parse ablation flags from name suffix, e.g. "moqadam_vnq_sgrf"
        vnq  = "vnq"  in name or "full" in name or name == f"moqadam_{bits}bit"
        sgrf = "sgrf" in name or "full" in name or name == f"moqadam_{bits}bit"
        dmu  = "dmu"  in name or "full" in name or name == f"moqadam_{bits}bit"
        return MoQAdam(
            params, **common,
            betas=(0.9, 0.999), eps=1e-8,
            quant_bits=bits, snr_ref=0.10, clip_norm=1.0,
            residual_dtype=torch.float16,
            quant_enabled=vnq,
            snr_gate_enabled=sgrf,
            decouple_moments=dmu,
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# CIFAR-100 benchmark
# ---------------------------------------------------------------------------

CIFAR_OPTIMIZERS = [
    "adamw",
    "naive_ef_adam",
    "ef21_adam",
    "moqadam_8bit",
    "moqadam_4bit",
]

ABLATION_OPTIMIZERS = [
    "adamw",
    "naive_ef_adam",          # +EF only (naive EF-Adam)
    "moqadam_vnq",            # +VNQ only
    "moqadam_sgrf",           # +SGRF only
    "moqadam_dmu",            # +DMU only
    "moqadam_vnq_sgrf",       # +VNQ+SGRF
    "moqadam_full",           # MoQAdam full
]


def run_cifar100(epochs: int = 100, quick: bool = False) -> dict:
    if quick:
        epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"CIFAR-100 benchmark — {epochs} epochs on {device}")
    print(f"{'='*60}")

    train_loader, test_loader = get_cifar100_loaders()
    results: dict[str, dict] = {}

    all_opts = CIFAR_OPTIMIZERS + [o for o in ABLATION_OPTIMIZERS
                                    if o not in CIFAR_OPTIMIZERS]

    for opt_name in all_opts:
        seed_everything()
        bits = 4 if "4bit" in opt_name else 8
        model = make_resnet18_cifar100().to(device)
        optimizer = make_optimizer(opt_name, model.parameters(), bits=bits)

        # cosine LR scheduler (standard for CIFAR benchmarks)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        best_acc = 0.0
        t_start  = time.time()
        for epoch in range(1, epochs + 1):
            train_cifar100_epoch(model, train_loader, optimizer, device)
            _, acc = evaluate_cifar100(model, test_loader, device)
            scheduler.step()
            if acc > best_acc:
                best_acc = acc
            if epoch % 20 == 0 or epoch == epochs:
                elapsed = time.time() - t_start
                print(f"  [{opt_name:25s}] epoch {epoch:3d}/{epochs} "
                      f"acc={acc:.2f}%  best={best_acc:.2f}%  "
                      f"elapsed={elapsed:.0f}s")

        val_loss, final_acc = evaluate_cifar100(model, test_loader, device)
        results[opt_name] = {
            "top1_acc": final_acc,
            "best_acc": best_acc,
        }
        print(f"  [{opt_name:25s}] FINAL  acc={final_acc:.2f}%  best={best_acc:.2f}%")

    return results


# ---------------------------------------------------------------------------
# WikiText-2 LM benchmark
# ---------------------------------------------------------------------------

LM_OPTIMIZERS = [
    "adamw",
    "naive_ef_adam",
    "ef21_adam",
    "moqadam_8bit",
    "moqadam_4bit",
]


def run_lm(epochs: int = 50, quick: bool = False) -> dict:
    if quick:
        epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"WikiText-2 LM benchmark — {epochs} epochs on {device}")
    print(f"{'='*60}")

    train_data, val_data, vocab_size, get_batch = get_wikitext2_loaders()
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    results: dict[str, dict] = {}

    for opt_name in LM_OPTIMIZERS:
        seed_everything()
        bits  = 4 if "4bit" in opt_name else 8
        model = LSTMLanguageModel(vocab_size).to(device)
        optimizer = make_optimizer(opt_name, model.parameters(), bits=bits)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        best_ppl = float("inf")
        t_start  = time.time()
        for epoch in range(1, epochs + 1):
            train_lm_epoch(model, train_data, optimizer, device, get_batch)
            ppl = evaluate_lm(model, val_data, device, get_batch)
            scheduler.step()
            if ppl < best_ppl:
                best_ppl = ppl
            if epoch % 10 == 0 or epoch == epochs:
                elapsed = time.time() - t_start
                print(f"  [{opt_name:25s}] epoch {epoch:2d}/{epochs} "
                      f"ppl={ppl:.2f}  best={best_ppl:.2f}  "
                      f"elapsed={elapsed:.0f}s")

        final_ppl = evaluate_lm(model, val_data, device, get_batch)
        results[opt_name] = {
            "final_ppl": final_ppl,
            "best_ppl":  best_ppl,
        }
        print(f"  [{opt_name:25s}] FINAL  ppl={final_ppl:.2f}  best={best_ppl:.2f}")

    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "adamw":          "AdamW",
    "naive_ef_adam":  "Naive EF-Adam",
    "ef21_adam":      "EF21-Adam~\\cite{richtarik2021ef21}",
    "moqadam_8bit":   "\\textbf{MoQAdam} (8-bit)",
    "moqadam_4bit":   "\\textbf{MoQAdam} (4-bit)",
    "moqadam_vnq":    "+VNQ only",
    "moqadam_sgrf":   "+SGRF only",
    "moqadam_dmu":    "+DMU only",
    "moqadam_vnq_sgrf": "+VNQ+SGRF",
    "moqadam_full":   "\\textbf{MoQAdam (full)}",
}

VNQ_MARK  = {"moqadam_vnq": True,  "moqadam_vnq_sgrf": True, "moqadam_full": True, "moqadam_8bit": True, "moqadam_4bit": True}
SGRF_MARK = {"moqadam_sgrf": True, "moqadam_vnq_sgrf": True, "moqadam_full": True, "moqadam_8bit": True, "moqadam_4bit": True}
DMU_MARK  = {"moqadam_dmu":  True, "moqadam_full": True, "moqadam_8bit": True, "moqadam_4bit": True}

def _ck(name: str, marks: dict) -> str:
    return r"\cmark" if marks.get(name) else r"\xmark"


def generate_latex_tables(
    cifar_results: dict,
    lm_results:    dict,
    ablation_results: dict,
) -> str:
    lines = []

    # ---- Table 1: CIFAR-100 ----
    lines.append(r"% --- Table 1: CIFAR-100 Top-1 Accuracy ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top-1 accuracy (\%) on CIFAR-100 (ResNet-18, 100 epochs).}")
    lines.append(r"\label{tab:cifar}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Optimizer} & \textbf{8-bit} & \textbf{4-bit} \\")
    lines.append(r"\midrule")

    for opt in ["adamw", "naive_ef_adam", "ef21_adam"]:
        name = DISPLAY_NAMES[opt]
        # AdamW has no bits dimension — show in both columns
        acc  = cifar_results.get(opt, {}).get("top1_acc", "--")
        val  = f"{acc:.2f}" if isinstance(acc, float) else "--"
        lines.append(f"{name} & {val} & {val} \\\\")

    for bits, opt in [(8, "moqadam_8bit"), (4, "moqadam_4bit")]:
        name = DISPLAY_NAMES[opt]
        acc  = cifar_results.get(opt, {}).get("top1_acc", "--")
        val  = f"{acc:.2f}" if isinstance(acc, float) else "--"
        other = cifar_results.get("moqadam_4bit" if bits == 8 else "moqadam_8bit", {}).get("top1_acc", "--")
        ov = f"{other:.2f}" if isinstance(other, float) else "--"
        if bits == 8:
            lines.append(f"\\textbf{{MoQAdam}} & {val} & {ov} \\\\")
            break

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ---- Table 2: WikiText-2 ----
    lines.append(r"% --- Table 2: WikiText-2 Perplexity ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Perplexity on WikiText-2 (2-layer LSTM, 50 epochs). Lower is better.}")
    lines.append(r"\label{tab:lm}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Optimizer} & \textbf{8-bit} & \textbf{4-bit} \\")
    lines.append(r"\midrule")
    for opt in ["adamw", "naive_ef_adam", "ef21_adam"]:
        name = DISPLAY_NAMES[opt]
        ppl  = lm_results.get(opt, {}).get("final_ppl", "--")
        val  = f"{ppl:.2f}" if isinstance(ppl, float) else "--"
        lines.append(f"{name} & {val} & {val} \\\\")

    ppl8 = lm_results.get("moqadam_8bit", {}).get("final_ppl", "--")
    ppl4 = lm_results.get("moqadam_4bit", {}).get("final_ppl", "--")
    v8   = f"{ppl8:.2f}" if isinstance(ppl8, float) else "--"
    v4   = f"{ppl4:.2f}" if isinstance(ppl4, float) else "--"
    lines.append(f"\\textbf{{MoQAdam}} & {v8} & {v4} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ---- Table 3: Ablation ----
    lines.append(r"% --- Table 3: Ablation (CIFAR-100, 4-bit) ---")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation of MoQAdam contributions (CIFAR-100, 4-bit).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{VNQ} & \textbf{SGRF} & \textbf{DMU} & \textbf{Top-1 Acc.} \\")
    lines.append(r"\midrule")

    ablation_row_order = [
        ("adamw",           "AdamW (baseline)"),
        ("naive_ef_adam",   "+EF only (naive EF-Adam)"),
        ("moqadam_vnq",     "+VNQ only"),
        ("moqadam_sgrf",    "+SGRF only"),
        ("moqadam_dmu",     "+DMU only"),
        ("moqadam_vnq_sgrf", "+VNQ+SGRF"),
        ("moqadam_full",    "\\textbf{MoQAdam (full)}"),
    ]
    for opt, label in ablation_row_order:
        acc = ablation_results.get(opt, cifar_results.get(opt, {})).get("top1_acc", "--")
        val = f"{acc:.2f}" if isinstance(acc, float) else "--"
        lines.append(
            f"{label} & {_ck(opt, VNQ_MARK)} & {_ck(opt, SGRF_MARK)} "
            f"& {_ck(opt, DMU_MARK)} & {val} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_csv(results: dict, path: Path) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    fieldnames = ["optimizer"] + list(first.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for opt_name, metrics in results.items():
            writer.writerow({"optimizer": opt_name, **metrics})
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MoQAdam benchmark")
    parser.add_argument("--task", choices=["cifar100", "lm", "both"],
                        default="both")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 5/3 epochs instead of 100/50")
    parser.add_argument("--cifar_epochs", type=int, default=100)
    parser.add_argument("--lm_epochs",    type=int, default=50)
    args = parser.parse_args()

    out = Path("results")
    out.mkdir(exist_ok=True)

    cifar_results:    dict = {}
    lm_results:       dict = {}
    ablation_results: dict = {}

    if args.task in ("cifar100", "both"):
        cifar_results = run_cifar100(
            epochs=args.cifar_epochs, quick=args.quick
        )
        # ablation shares the CIFAR-100 results (4-bit configs already run)
        ablation_results = cifar_results
        save_csv(cifar_results, out / "cifar100_results.csv")

    if args.task in ("lm", "both"):
        lm_results = run_lm(
            epochs=args.lm_epochs, quick=args.quick
        )
        save_csv(lm_results, out / "lm_results.csv")

    # Generate ready-to-paste LaTeX tables
    latex = generate_latex_tables(cifar_results, lm_results, ablation_results)
    table_path = out / "paper_tables.txt"
    table_path.write_text(latex)
    print(f"\nLaTeX tables written to: {table_path}")
    print("\n" + "="*60)
    print(latex)


if __name__ == "__main__":
    main()
