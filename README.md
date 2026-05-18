# MoQAdam — Moment-Normalized Quantization Adam

**Distributed training with provably bounded, automatically annealing gradient quantization.**

MoQAdam is a drop-in replacement for AdamW for bandwidth-bottlenecked distributed training. It quantizes gradients in Adam's own normalized space, gates residual feedback by the gradient SNR, and decouples moment updates to preserve an unbiased variance estimate — three tightly coupled contributions, each with independent theoretical guarantees.

---

## Three contributions at a glance

| Contribution | What it does | Why it matters |
|---|---|---|
| **VNQ** — Variance-Normalized Quantization | Quantizes in Adam's normalized space `g_t / σ_t` | Quantization error bound is layer-invariant and distribution-free |
| **SGRF** — SNR-Gated Residual Feedback | Weights residual by gradient SNR `γ_t = clamp(ρ̄_t / ρ_ref, 0, 1)` | Residual automatically vanishes near convergence; no oscillation |
| **DMU** — Decoupled Moment Updates | Updates `v_t` from `g_t`, `m_t` from `q_t` | `v̂_t` remains an unbiased estimator of `E[g_t²]` throughout training |

---

## Installation

MoQAdam has no dependencies beyond PyTorch ≥ 2.1 (for `register_post_accumulate_grad_hook`).

```bash
# copy the two files into your project
cp moqadam.py moqadam_ddp.py your_project/
```

---

## Quick start

### Single GPU

```python
from moqadam import MoQAdam, create_moqadam

model = MyModel()

# Option A — explicit hyperparameters
opt = MoQAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    quant_bits=8,          # 8-bit gradient quantization
    snr_ref=0.10,          # SGRF gate reference SNR
    clip_norm=1.0,         # global L2 gradient clipping
    residual_dtype=torch.float16,  # fp16 residual → 2.5P memory
    quant_enabled=True,
    snr_gate_enabled=True,
    decouple_moments=True,
)

# Option B — factory presets
opt = create_moqadam(model.parameters(), preset="balanced")  # 8-bit
opt = create_moqadam(model.parameters(), preset="low_bit")   # 4-bit
opt = create_moqadam(model.parameters(), preset="memory")    # 8-bit + fp16 residual

for x, y in loader:
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
```

### Multi-GPU DDP (recommended: `torchrun`)

```python
# train.py — launch with: torchrun --nproc_per_node=NUM_GPUS train.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from moqadam import MoQAdam
from moqadam_ddp import MoQAdamDDPHook

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

base_opt = MoQAdam(
    model.parameters(),
    lr=1e-3,
    quant_bits=8,
    residual_dtype=torch.float16,
)
# MoQAdamDDPHook registers both the per-parameter math hook and the
# float16 transport comm hook automatically.
opt = MoQAdamDDPHook(base_opt, model)

for step, (x, y) in enumerate(loader):
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()   # math hook: VNQ + SGRF per param; transport: fp16 AllReduce
    opt.step()        # norm AllReduce → new clip scale; denormalize; m_t update

    # Mandatory under non-IID data (federated, curriculum); optional under IID.
    # Every K ≤ floor(1 / (1 - β₂)) steps — for β₂=0.999, every ≤1 000 steps.
    if step % 1000 == 0:
        opt.sync_variance_states()
```

### Gradient accumulation (multi-GPU)

Use `wrapper.no_sync()` instead of manually setting `wrapper.accumulate`. The context manager activates both the optimizer-level bypass and DDP's AllReduce suppression atomically, preventing raw un-normalized gradients from reaching the float16 transport hook.

```python
ACCUM_STEPS = 4

for i, (x, y) in enumerate(loader):
    is_final = (i + 1) % ACCUM_STEPS == 0
    ctx = contextlib.nullcontext() if is_final else opt.no_sync()
    with ctx:
        loss = criterion(model(x), y) / ACCUM_STEPS
        loss.backward()
    if is_final:
        opt.step()
        opt.zero_grad()
```

---

## Hyperparameter reference

| Parameter | Default | Notes |
|---|---|---|
| `lr` | `1e-3` | Standard AdamW default |
| `betas` | `(0.9, 0.999)` | Standard AdamW default |
| `eps` | `1e-8` | Standard AdamW default |
| `weight_decay` | `1e-4` | AdamW decoupled weight decay |
| `quant_bits` | `8` | Bits for VNQ quantization. Use 4–8; 2-bit is experimental |
| `snr_ref` | `0.10` | SGRF gate reference SNR ρ_ref. Range [0.05, 0.20]; higher = more aggressive suppression |
| `clip_norm` | `1.0` | Global L2 clipping threshold G_max. Set 0 to disable |
| `residual_dtype` | `torch.float32` | `torch.float16` halves residual memory (e → 0.5P) |
| `quant_enabled` | `True` | Set False for ablation baseline (reduces to AdamW + DMU) |
| `snr_gate_enabled` | `True` | Set False to disable SGRF gate (unconditional EF) |
| `decouple_moments` | `True` | Set False to update v_t from q_t (biases variance estimate) |

### Preset configurations

```python
create_moqadam(params, preset="balanced")   # 8-bit, snr_ref=0.10, clip=1.0
create_moqadam(params, preset="low_bit")    # 4-bit, snr_ref=0.05, clip=1.0
create_moqadam(params, preset="memory")     # 8-bit + fp16 residual (2.5P state)
create_moqadam(params, preset="no_quant")   # ablation: AdamW + DMU only
create_moqadam(params, preset="full_quant") # 2-bit experimental
```

---

## Memory and compute overhead

### Optimizer state

| Configuration | State per parameter | vs AdamW |
|---|---|---|
| AdamW | `m + v` = 2P | baseline |
| MoQAdam (fp32 residual) | `m + v + e` = 3P | +1P |
| MoQAdam (fp16 residual) | `m + v + e_fp16` = 2.5P | +0.5P |
| 8-bit Adam (bitsandbytes) | ~0.5P | −1.5P |

MoQAdam is **not** designed for VRAM-bottlenecked single-node training. For that use case, [8-bit Adam](https://github.com/TimDettmers/bitsandbytes) is more appropriate. MoQAdam's value proposition is **communication efficiency with provable error control** in bandwidth-bottlenecked multi-node or federated settings.

### Per-step compute overhead

- **VNQ normalize/denormalize**: 2 × O(n) element-wise ops
- **Quantization range** (`amax`): O(n) GPU reduction, no sort, no host transfer
- **SGRF gate**: O(n) SNR computation + scalar mean (on-device, no pipeline stall)
- **DMU**: free — a substitution of `g_t` for `q_t` in the `v_t` update
- **Global clipping**: one O(P) norm + one GPU→CPU scalar sync (matches `clip_grad_norm_`)

**Total overhead: < 3% of AdamW step time.**

### `sync_variance_states` memory

Allocates one transient 1P float32 flat buffer for the duration of the AllReduce call, then frees it immediately. Called every ~1 000 steps — allocation cost is negligible. Sustained footprint remains 2.5P.

---

## Scope and known limitations

### Mixed precision

MoQAdam supports **native bfloat16** (`torch.autocast(dtype=torch.bfloat16)`) without any changes. bfloat16 shares float32's dynamic range and does not need a `GradScaler`.

**float16 AMP with `GradScaler` is not supported.** The GradScaler inflates loss by up to 2¹⁶ before `backward()`, so gradients arriving at the per-parameter hook are artificially large. The global clipping threshold `G_max` fires aggressively on every step, destroying training. Use bfloat16 instead, or set `quant_enabled=False` and manage unscaling externally.

### Parallelism strategy

MoQAdam targets **Data Parallel (DDP)** training. It does not currently support:

- **FSDP** (Fully Sharded Data Parallel): FSDP uses reduce-scatter / all-gather, not DDP buckets, so `register_comm_hook` never fires and the per-worker norm accumulation only captures local shards.
- **DeepSpeed ZeRO**: same issue — parameter and optimizer state sharding breaks the per-parameter hook model.

Adapting VNQ and SGRF to sharded parallelism is left as future work.

### Routing models (`find_unused_parameters=True`)

Supported. When a worker skips a layer (e.g. Mixture of Experts routing), the math hook never fires for that parameter. `step()` detects the missing `_sigma` key and applies a zero-gradient `v_t` decay to synthesize the correct denormalization scale, ensuring the AllReduced signal is correctly re-scaled on all workers. Cluster state remains synchronized.

---

## Citation

```bibtex
@article{guenchi2025moqadam,
  title   = {MoQAdam: Moment-Normalized Quantization Adam via
             SNR-Gated Residual Feedback and Decoupled Moment Updates},
  author  = {Guenchi, Samir},#
  year    = {2026},
}
```
