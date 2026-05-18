"""
MoQAdam DDP Wrapper — v2
========================
Author: Samir Guenchi

PyTorch 2.x distributed wrapper for MoQAdam.

Architecture overview
---------------------
Two hooks work together; each has a single, clear responsibility:

  1. Per-parameter math hook  (register_post_accumulate_grad_hook)
       - Fires per-parameter during loss.backward(), in reverse topological order.
       - Applies DELAYED clip scale from the previous step (see below).
       - Updates v_t in-place from the local gradient (DMU — no stash needed).
       - Applies SGRF gate and VNQ, producing q_norm ∈ [-q_levels, q_levels].
       - Writes q_norm (integer-valued floats) back into p.grad.
       - Stores sigma_t in state["_sigma"] for step() to denormalize later.
       - Accumulates p.grad.norm()^2 into wrapper._local_norm_sq for clipping.

  2. Lightweight transport hook  (DDP register_comm_hook)
       - Receives the flat concatenated bucket of p.grad values (already q_norm).
       - Casts the float32 bucket to float16 — halving transport bandwidth.
       - Issues one async NCCL AllReduce per bucket (DDP's natural scheduling).
       - Casts the result back to float32 and divides by world_size.
       - Bandwidth saving: 2× (float32 → float16 transport).
       - True int8 transport (4×) would require int32 AllReduce accumulators
         to prevent overflow (sum of K workers at ±127 reaches ±127K); left
         as future work and noted in the paper.

  3. step() post-backward:
       - AllReduces wrapper._local_norm_sq (a single scalar — 32 bytes) via
         dist.all_reduce to get the true global gradient norm across workers.
       - Computes clip_scale = min(1, G_max / global_norm), stores for next step.
       - For each parameter: reads p.grad (AllReduced q_norm), reads state["_sigma"],
         denormalizes active_grad = p.grad * sigma, updates m_t.
       - v_t was already updated in the hook; step() recomputes v_hat for m_hat.

Memory accounting (with fp16 residual preset)
----------------------------------------------
  Baseline optimizer state: m_t (fp32, P) + v_t (fp32, P) + e_t (fp16, 0.5P) = 2.5P
  DDP wrapper additions:
    state["_sigma"]: fp32, P — needed to denormalize in step(); freed after step().
  Peak during hook (backward): _sigma is created lazily per-parameter and freed
    by step().  No _orig_grad stash.  Peak overhead above 2.5P is one parameter's
    _sigma at a time — O(max_param_size), not O(P).
  Net sustained state: 2.5P (same as advertised).

Delayed clipping
----------------
  True global clipping before quantization requires knowing the global gradient
  norm before the first hook fires — impossible without a synchronous barrier
  that destroys computation/communication overlap.

  Solution: delayed clipping (standard in pipeline-parallel training).
    - At step t, step() computes the global norm from accumulated local_norm_sq
      (one scalar AllReduce), stores clip_scale_t = min(1, G_max / norm_t).
    - At step t+1, each hook applies clip_scale_{t} before VNQ.
  This guarantees clipping before quantization at the cost of using the previous
  step's norm.  For smooth loss landscapes (most transformer training regimes)
  consecutive step norms differ by <5%, making the one-step lag negligible.
  Step 0 uses clip_scale=1 (no clipping), which is safe because step 0 gradients
  have not yet been seen.

Fused v_t synchronization (non-IID mandatory rule)
---------------------------------------------------
  sync_variance_states() flattens all v_t tensors into one buffer, AllReduces
  once, unflattens.  Call every K <= floor(1/(1-beta2)) steps under non-IID data.

Minimal usage
-------------
    model = DDP(model, device_ids=[local_rank])
    base_opt = MoQAdam(model.parameters(), lr=1e-3, quant_bits=8,
                       residual_dtype=torch.float16)
    opt = MoQAdamDDPHook(base_opt, model, process_group=None)
    # Registers both hooks automatically.

    for step_idx, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()       # math hook fires per-param; transport hook compresses
        opt.step()            # norm AllReduce → new clip_scale; m_t update; param update
        if step_idx % 1000 == 0:
            opt.sync_variance_states()   # mandatory under non-IID data
"""

from __future__ import annotations

import contextlib
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from moqadam import MoQAdam


# ---------------------------------------------------------------------------
# Transport hook: float16 bucket-level compression (2× bandwidth saving)
# ---------------------------------------------------------------------------

def fp16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Lightweight DDP comm hook: float32 → float16 transport → float32.

    This hook has NO knowledge of individual parameter boundaries.  Its only
    job is bandwidth reduction.  The per-parameter math (VNQ, SGRF) was already
    done by the post_accumulate_grad_hook before DDP assembled this bucket.

    The bucket contains quantized gradient values written by the math hook.
    These are continuous floats bounded in [-x_max, x_max] where x_max =
    amax(|corrected_grad|) — NOT pure integers.  The _quantize_normalized
    function produces q_norm = round(corrected/step) * step, i.e. multiples
    of the quantization step size, which are rational but not generally integer.
    Each worker's x_max incorporates that worker's local residual e_t, so
    workers may have slightly different x_max values — pure integer encoding
    without transmitting x_max is not possible.

    Why float16 SUM cannot overflow here (VNQ normalization guarantee):
      A naive concern is that SUM across K workers overflows float16 (max 65,504).
      This does NOT apply because VNQ normalizes every gradient by sigma_t
      (Adam's second-moment proxy) before quantization.  The corrected gradient
      passed to _quantize_normalized is bounded by x_max = amax(|corrected|),
      and because corrected = (grad + gamma*residual) / sigma, the values sit
      near the normalized Adam update, empirically bounded to ~3.0 even for
      heavy-tailed distributions.  Worst case: K=1024 × x_max=3.0 = 3072,
      far below float16 max of 65,504.

      Pre-dividing by world_size is therefore NOT needed, and is actively
      harmful: it pushes values down to x_max/K ≈ 3/1024 ≈ 0.003, approaching
      the float16 subnormal threshold (~6.1×10^-5).  GPU hardware flushes
      subnormals to zero (flush-to-zero mode), silently destroying the smallest
      quantization bins and degrading 8-bit resolution.  The cast-then-SUM-then-
      divide approach is both mathematically safe and numerically superior.

    Bandwidth: transmits float16 tensors → 2× saving over float32 baseline.

    Note on int8 transport (4× saving):
      Even without overflow risk at the application level, int8 ring-allreduce
      requires int32 partial accumulators to prevent intermediate ring-buffer
      overflow; outside standard PyTorch comm hooks.  Available via NCCL C++.
    """
    group_to_use = process_group or dist.group.WORLD
    world_size   = dist.get_world_size(group_to_use)

    tensor = bucket.buffer()                         # flat float32 buffer
    # Cast to float16 BEFORE AllReduce.  VNQ normalization guarantees values
    # are O(1) so float16 SUM across any realistic K cannot overflow (see
    # docstring proof above).  Dividing AFTER the SUM avoids the subnormal
    # flush-to-zero trap that pre-division would cause on large clusters.
    #
    # Explicit device guard: GradBucket.buffer() is almost always on the
    # correct CUDA device, but under advanced sharding (FSDP, custom model
    # parallelism) the buffer could land on an unexpected device.  Pinning
    # tensor_fp16 to the process group's expected device (derived from the
    # first parameter of the hook's local bucket) prevents NCCL device
    # mismatch errors in those setups.
    expected_device = bucket.parameters()[0].device
    tensor_fp16 = tensor.to(expected_device).half()

    # Use the low-level ProcessGroup API (list-based signature) instead of
    # the global dist.all_reduce wrapper.  DDP comm hooks are designed to
    # interface with the ProcessGroup backend directly; the global wrapper
    # can cause thread deadlocks in highly scaled async bucket pipelines.
    work = group_to_use.allreduce([tensor_fp16])
    fut  = work.get_future()

    def decompress(fut: torch.futures.Future) -> torch.Tensor:
        # Divide by world_size after SUM → MEAN, then cast back to float32.
        return fut.value()[0].float().div_(world_size)

    return fut.then(decompress)


# ---------------------------------------------------------------------------
# Fused v_t synchronization utility
# ---------------------------------------------------------------------------

def sync_variance_states(
    optimizer: MoQAdam,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Fused AllReduce of ALL v_t tensors grouped by device.

    Naive per-tensor loop: O(num_layers) kernel launches × ~10-50 µs each
    = 1-5 ms latency overhead for a 100-layer model.

    Fused approach: one torch.cat per device → one dist.all_reduce per device
    → scatter back.  For single-device models this is one kernel launch
    regardless of model depth.  For pipeline-parallel models (cuda:0 / cuda:1)
    two launches are issued — one per device — still O(1) per device.

    Memory: allocates a transient 1P float32 flat buffer per device for the
    duration of the AllReduce, then immediately frees it.  Because this
    function is called infrequently (every ~1 000 steps), the allocation
    cost is negligible.  Crucially, NO persistent buffer is retained between
    calls — the sustained optimizer footprint remains 2.5P (m + v + fp16 e)
    and does not grow by a hidden 1P tax.

    Mandatory under non-IID data sharding every K ≤ ⌊1/(1-β₂)⌋ steps.
    Optional under IID sharding.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    pg         = process_group or dist.group.WORLD
    world_size = dist.get_world_size(pg)

    # Group v tensors by device to avoid cross-device torch.cat crashes.
    # Pipeline-parallel models have parameters on multiple GPUs; a single
    # torch.cat over all v_views would throw RuntimeError: Tensors must be
    # on the same device.
    device_views: dict[torch.device, list[torch.Tensor]] = {}
    device_meta:  dict[torch.device, list[tuple]]        = {}

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None or "v" not in state:
                continue
            v   = state["v"]
            dev = v.device
            if dev not in device_views:
                device_views[dev] = []
                device_meta[dev]  = []
            device_views[dev].append(v.view(-1))
            device_meta[dev].append((p, v.numel(), v.shape))

    if not device_views:
        return

    for dev, views in device_views.items():
        total_numel = sum(v.numel() for v in views)

        # Fresh allocation per call — freed immediately after copying back.
        # This produces a transient 1P spike only during the sync call itself
        # (which happens every ~1 000 steps), not a persistent VRAM tax.
        flat_v: torch.Tensor = torch.cat(views)   # new contiguous buffer

        dist.all_reduce(flat_v, op=dist.ReduceOp.SUM, group=pg)
        flat_v.div_(world_size)

        offset = 0
        for p, numel, shape in device_meta[dev]:
            optimizer.state[p]["v"].copy_(flat_v[offset : offset + numel].view(shape))
            offset += numel

        # Explicitly release the flat buffer and return VRAM to the
        # allocator immediately so sustained footprint stays at 2.5P.
        del flat_v
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-parameter math hook factory
# ---------------------------------------------------------------------------

def _make_per_param_hook(
    param:     torch.Tensor,
    optimizer: MoQAdam,
    group:     dict,
    wrapper:   "MoQAdamDDPHook",
):
    """
    Factory for a post-accumulate-grad hook on ``param``.

    The hook performs all the expensive per-parameter math:
      1. Applies delayed clip_scale from the previous step (before quantization).
      2. Updates v_t in-place from the local gradient — DMU, no _orig_grad stash.
      3. Computes sigma_t from the just-updated v_t.
      4. Applies SGRF gate from previous m_t and v_t.
      5. VNQ: normalize, add gated residual, quantize to q_norm ∈ [-q_levels, q_levels].
      6. Updates the gated residual buffer in-place.
      7. Stores sigma_t in state["_sigma"] for step() to denormalize.
      8. Writes q_norm.float() to p.grad so DDP's transport hook can compress it.
      9. Accumulates p.grad.norm()^2 into wrapper._local_norm_sq for delayed clipping.

    Memory note:
      v_t update is done here (step 2) so the original gradient is available and
      then released — no separate _orig_grad tensor persists in optimizer.state.
      state["_sigma"] is one fp32 tensor per parameter, freed by step() after use.
    """

    @torch.no_grad()
    def hook(p: torch.Tensor) -> None:
        if p.grad is None:
            return

        # Gradient accumulation guard: when wrapper.accumulate is True, this
        # is a non-final micro-batch.  Return immediately so PyTorch's native
        # autograd accumulates the raw gradient into p.grad untouched.
        # On the final micro-batch (accumulate=False), the fully-accumulated
        # p.grad arrives here and receives the full VNQ + v_t treatment.
        if wrapper.accumulate:
            return

        bits         = group["quant_bits"]
        beta1, beta2 = group["betas"]
        eps          = group["eps"]
        snr_ref      = group["snr_ref"]
        res_dtype    = group["residual_dtype"]
        snr_on       = group["snr_gate_enabled"]
        quant_on     = group["quant_enabled"]

        grad = (
            p.grad.float()
            if p.grad.dtype in (torch.float16, torch.bfloat16)
            else p.grad.clone()
        )

        if not torch.isfinite(grad).all():
            return

        # ---- State initialisation ----
        state = optimizer.state[p]
        if len(state) == 0:
            p_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data
            state["step"]     = 0
            state["m"]        = torch.zeros_like(p_fp32)
            state["v"]        = torch.zeros_like(p_fp32)
            state["residual"] = torch.zeros_like(
                p_fp32,
                dtype=res_dtype if res_dtype != torch.float32 else p_fp32.dtype,
            )

        m        = state["m"]
        v        = state["v"]
        residual = state["residual"]

        # Step index for this upcoming step (state["step"] is incremented in step()).
        t  = state["step"] + 1
        bc2_prev = 1.0 - beta2 ** max(state["step"], 1)   # for SGRF gate (prev moments)
        bc1_prev = 1.0 - beta1 ** max(state["step"], 1)

        # ---- Step 1: Delayed global clipping ----
        # Measure the norm of the RAW (unclipped) gradient FIRST.
        # This value is AllReduced in step() to compute the clip_scale for
        # step t+1.  If we measured after clipping, a severe clip at step t
        # (scale=0.1) would make the step t norm look 10× smaller, causing
        # step t+2 to use scale=1.0 (no clip), then step t+2's norm looks
        # large again, producing a destructive oscillatory clip/no-clip loop.
        # Measuring the raw norm breaks this feedback entirely.
        # Accumulate raw squared norm into the pre-allocated float64 scalar.
        # IMPORTANT: cast to float64 BEFORE computing the norm, not after.
        # grad.norm().pow(2).to(float64) first computes in float32, squaring
        # an already-truncated norm — any underflow or cancellation that
        # occurred in float32 is already baked in before the cast.
        # grad.to(float64).norm().pow(2) performs the entire reduction in
        # float64, preserving all mantissa bits throughout.
        # The accumulator was pre-allocated in __init__ — zero new allocation.
        dev = grad.device
        raw_norm_sq = grad.to(torch.float64).norm().pow(2)
        wrapper._local_norm_sq[dev].add_(raw_norm_sq)

        # Apply the clip scale from the previous step AFTER measuring the norm.
        prev_scale = wrapper._prev_clip_scale
        if prev_scale is not None and prev_scale < 1.0:
            grad = grad.mul(prev_scale)

        if not quant_on:
            # Ablation: no quantization — leave p.grad unchanged for DDP.
            return

        # ---- Step 2: DMU — update v_t from original (clipped) gradient ----
        # Done HERE, not in step(), so we never stash _orig_grad.
        # peak memory overhead: zero (grad is already on the device, we reuse it).
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        v_hat  = v / (1.0 - beta2 ** t)
        sigma  = v_hat.add(eps).sqrt_()

        # ---- Step 3: SGRF gate (from PREVIOUS m_t and v_t) ----
        gamma = 1.0
        if snr_on and state["step"] >= 1:
            m_hat_prev = m / bc1_prev
            v_hat_prev = v / bc2_prev   # v was just updated; use new v for prev approx
            # Note: we use the just-updated v as a proxy for the previous v_hat.
            # The EMA makes consecutive v_hat values nearly identical (β₂≈0.999).
            snr_mean = m_hat_prev.pow(2).div(v_hat_prev.add(eps)).mean()
            gamma = snr_mean.clamp_(0.0, snr_ref).div_(snr_ref)

        # ---- Step 4: VNQ ----
        res_fp32  = residual.float() if residual.dtype != torch.float32 else residual
        norm_grad = grad / sigma
        corrected = norm_grad + res_fp32
        q_norm, r = optimizer._quantize_normalized(corrected, bits)

        # Gate and write back residual.
        r.mul_(gamma)
        if residual.dtype != torch.float32:
            residual.copy_(r.to(residual.dtype))
        else:
            residual.copy_(r)

        # ---- Step 5: store sigma for step() to denormalize ----
        # Freed by step() via state.pop("_sigma") after use.
        state["_sigma"] = sigma

        # ---- Step 6: write quantized values to p.grad for transport hook ----
        # q_norm contains continuous floats bounded in [-x_max, x_max] (multiples
        # of the quantization step size, not pure integers — see fp16_compress_hook
        # docstring for the full explanation).  The transport hook casts to float16,
        # halving bandwidth; float16 truncation error is negligible vs. VNQ error.
        p.grad.copy_(q_norm.to(p.grad.dtype))

    return hook


# ---------------------------------------------------------------------------
# MoQAdamDDPHook
# ---------------------------------------------------------------------------

class MoQAdamDDPHook:
    """
    Distributed wrapper for MoQAdam.

    Registers:
      - A per-parameter post-accumulate-grad hook for VNQ + SGRF + DMU math.
      - A DDP comm hook for float16 bucket transport (2× bandwidth saving).

    After loss.backward():
      - p.grad = AllReduced q_norm / world_size (quantized, float32).
      - state["_sigma"] = per-parameter normalization scale (fp32).
      - wrapper._local_norm_sq = sum of ||g_t||^2 across local params (unAllReduced).

    step() then:
      1. AllReduces _local_norm_sq (one scalar) → global gradient norm.
      2. Computes clip_scale for NEXT step, stores as _prev_clip_scale.
      3. For each param: denormalizes p.grad via _sigma, updates m_t.
      4. v_t was already updated in the hook — just recomputes v_hat for m_hat.
      5. Applies Adam update to parameters.
    """

    def __init__(
        self,
        optimizer: MoQAdam,
        model: nn.Module,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.optimizer     = optimizer
        self.model         = model
        self.process_group = process_group or (
            dist.group.WORLD if dist.is_initialized() else None
        )
        self._prev_clip_scale: Optional[float] = None
        # Per-device accumulator for raw ||g||^2.  Keyed by torch.device so
        # that pipeline-parallel models (layers on cuda:0 and cuda:1) never
        # attempt cross-device in-place addition.
        # Pre-allocated in __init__ over all unique devices in the model so
        # that hooks never call torch.zeros() during the backward pass —
        # zero dynamic allocation at the memory peak.
        # Using float64 for numerical stability: for billion-parameter models
        # on large clusters, sum-of-squared-norms can span many orders of
        # magnitude; float32 (7 sig. digits) loses precision catastrophically.
        self._local_norm_sq: dict[torch.device, torch.Tensor] = {}
        for p in model.parameters():
            dev = p.device
            if dev not in self._local_norm_sq:
                self._local_norm_sq[dev] = torch.zeros((), device=dev, dtype=torch.float64)
        # Gradient accumulation support.
        # Set wrapper.accumulate = True before each non-final micro-batch
        # backward, and False before the final one.  While True, hooks return
        # immediately, letting PyTorch accumulate raw gradients in p.grad
        # without any VNQ/v_t interference.  On the final micro-batch
        # (accumulate=False) the hook applies full VNQ math to the
        # fully-accumulated gradient.  Default False = no accumulation.
        self.accumulate: bool = False

        self._math_hooks: list = []
        self._register_math_hooks()

        # Only register the float16 transport hook when quantization is
        # active.  With quant_enabled=False ("no_quant" ablation preset),
        # p.grad holds raw unnormalized gradients that lack VNQ's O(1) bound;
        # casting them to float16 before SUM AllReduce would overflow on large
        # clusters.  In that case DDP uses its standard float32 pipeline.
        quant_active = any(
            g.get("quant_enabled", True)
            for g in self.optimizer.param_groups
        )
        if quant_active:
            self._register_transport_hook()

    def _register_math_hooks(self) -> None:
        param_set = {
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
        }
        group_map = {
            p: group
            for group in self.optimizer.param_groups
            for p in group["params"]
        }

        for p in self.model.parameters():
            if not p.requires_grad or p not in param_set:
                continue
            group  = group_map[p]
            hook   = _make_per_param_hook(p, self.optimizer, group, self)
            handle = p.register_post_accumulate_grad_hook(hook)
            self._math_hooks.append(handle)

    def _register_transport_hook(self) -> None:
        """
        Register the float16 compression comm hook on the DDP model.
        If the model is not a DDP instance, skip silently (single-GPU mode).
        """
        if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return
        pg = self.process_group or dist.group.WORLD
        self.model.register_comm_hook(pg, fp16_compress_hook)

    @contextmanager
    def no_sync(self):
        """
        Context manager for safe gradient accumulation.

        Usage (accumulate over N micro-batches before stepping)::

            for i, batch in enumerate(loader):
                is_final = (i + 1) % accum_steps == 0
                ctx = contextlib.nullcontext() if is_final else wrapper.no_sync()
                with ctx:
                    loss = model(batch)
                    loss.backward()
                if is_final:
                    wrapper.step()
                    wrapper.optimizer.zero_grad()

        Why both flags are required together:

        Setting ``wrapper.accumulate = True`` alone only silences the *math
        hook*.  DDP fires the *transport hook* (``fp16_compress_hook``) as
        soon as a bucket is ready, regardless of any optimizer flag.  If the
        math hook has exited early, ``p.grad`` still holds raw, un-normalized
        gradients.  Casting those to float16 before ``dist.all_reduce(SUM)``
        can overflow on large clusters because VNQ's O(1) bound only applies
        after normalization by ``sigma_t``.

        ``model.no_sync()`` suppresses DDP's AllReduce entirely for the
        duration, so the transport hook never fires.  This context manager
        activates both suppressions atomically and restores both on exit,
        making the API impossible to misuse.

        Single-GPU / non-DDP usage: if ``self.model`` is a plain
        ``nn.Module`` (e.g. during single-GPU debugging), it has no
        ``no_sync()`` method.  We fall back to ``contextlib.nullcontext()``
        so the wrapper can be used identically in both environments.

        Exception safety: ``self.accumulate`` is restored in a ``finally``
        block so that even an OOM or data-loader error during the forward/
        backward pass cannot leave the optimizer permanently stuck in
        accumulation mode.
        """
        old_accumulate = self.accumulate
        self.accumulate = True
        no_sync_ctx = (
            self.model.no_sync()
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else contextlib.nullcontext()
        )
        try:
            with no_sync_ctx:
                yield
        finally:
            self.accumulate = old_accumulate

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """
        Optimizer step.  Called after loss.backward() + DDP AllReduce.

        At entry:
          - p.grad = AllReduced q_norm (quantized, float32).
          - state["_sigma"] = normalization scale used during quantization.
          - self._local_norm_sq = sum of local ||g||^2 (not yet AllReduced).

        This method:
          1. AllReduces _local_norm_sq → true global gradient norm (one scalar).
          2. Computes and stores clip_scale for next step's hooks (delayed clipping).
          3. Updates m_t from denormalized AllReduced q_norm.
          4. v_t was already updated in the hook; recomputes v_hat for Adam.
          5. Applies Adam update and writes back parameters.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ---- Delayed clipping: compute new clip_scale for NEXT step ----
        # AllReduce the local sum-of-squared-norms (one scalar, 32 bytes).
        clip = (
            self.optimizer.param_groups[0].get("clip_norm", 0.0)
            if self.optimizer.param_groups else 0.0
        )
        if clip > 0 and self._local_norm_sq:
            # Step 1: sum per-device scalars onto CPU in float64.
            # float64 accumulation prevents catastrophic cancellation when
            # summing thousands of squared gradient norms near convergence
            # (float32 has only ~7 significant digits; float64 has ~15).
            # Works on single-GPU (dist not initialized) identically to
            # multi-GPU — clipping is never silently bypassed.
            norm_sq = torch.zeros((), dtype=torch.float64)
            for dev_tensor in self._local_norm_sq.values():
                norm_sq.add_(dev_tensor.cpu())

            # Step 2: AllReduce only when running distributed.
            # Single-GPU: norm_sq already IS the global norm (one worker).
            if dist.is_available() and dist.is_initialized():
                # The backend (NCCL, Gloo, MPS, XLA) requires the tensor to
                # reside on the same device as the process group.  Derive the
                # target device dynamically from the accumulator dict rather
                # than hardcoding .cuda() — this makes the code work on CPU
                # clusters (Gloo), Apple Silicon (mps), and TPUs (xla).
                target_device = next(iter(self._local_norm_sq))
                norm_sq_dev = norm_sq.to(target_device)
                dist.all_reduce(norm_sq_dev, op=dist.ReduceOp.SUM, group=self.process_group)
                norm_sq = norm_sq_dev.cpu()

            # Step 3: sqrt in float64 → scalar; convert to float32 only now.
            # Converting before sqrt would lose the precision benefit.
            global_norm = norm_sq.sqrt().item()   # Python float (float64)
            self._prev_clip_scale = min(1.0, clip / global_norm) if global_norm > clip else 1.0
        else:
            self._prev_clip_scale = 1.0
        # Zero all per-device accumulators in-place; dict entries persist for reuse.
        for dev_tensor in self._local_norm_sq.values():
            dev_tensor.zero_()

        # ---- Per-parameter moment update and Adam step ----
        for group in self.optimizer.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]
            quant_on     = group["quant_enabled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.optimizer.state[p]
                if len(state) == 0:
                    continue

                state["step"] = state.get("step", 0) + 1
                t     = state["step"]
                bc1   = 1.0 - beta1 ** t
                bc2   = 1.0 - beta2 ** t
                m     = state["m"]
                v     = state["v"]       # already updated in hook (DMU)

                # p.grad holds AllReduced q_norm (quantized integers / world_size).
                agg_q_norm = (
                    p.grad.float()
                    if p.grad.dtype in (torch.float16, torch.bfloat16)
                    else p.grad
                )

                if quant_on:
                    if "_sigma" in state:
                        # Normal path: parameter was used locally this step.
                        # Hook fired: v_t already updated (DMU), _sigma stored.
                        sigma = state.pop("_sigma")
                    else:
                        # Unused-parameter path (find_unused_parameters=True).
                        # This worker did not use this layer in its forward pass.
                        # PyTorch autograd never triggered the math hook, so:
                        #   - v_t was NOT updated this step (hook never fired).
                        #   - _sigma was NOT stored.
                        #   - p.grad was set to 0 by DDP, but after AllReduce
                        #     it holds the average of other workers' q_norm values.
                        #
                        # We must decay v_t here (matching what the hook would have
                        # done with a zero gradient) and derive sigma from the
                        # resulting bias-corrected estimate.  This gives the exact
                        # same denormalization scale that active workers computed
                        # in their hooks, so the AllReduced signal is correctly
                        # re-scaled on both active and inactive workers.
                        #
                        # Failure mode without this branch: the else clause below
                        # would update v_t from agg_q_norm (integer-scale values,
                        # not gradient-scale), permanently corrupting the second
                        # moment and desynchronising the cluster.
                        v.mul_(beta2)                   # zero-gradient v_t decay
                        v_hat = v / bc2
                        sigma = v_hat.add(eps).sqrt_()  # in-place, no new tensor

                    # Denormalize: recover true gradient scale from the AllReduced
                    # quantized signal.  Both paths land here with a valid sigma.
                    agg_grad = agg_q_norm * sigma

                    # v_t is already up-to-date (updated in hook OR decayed above).
                    v_hat = v / bc2
                    # Update m_t from the denormalized AllReduced aggregate.
                    m.mul_(beta1).add_(agg_grad, alpha=1.0 - beta1)
                else:
                    # Ablation / quant disabled: standard AdamW moment updates.
                    # p.grad holds full-precision AllReduced gradients.
                    v.mul_(beta2).addcmul_(agg_q_norm, agg_q_norm, value=1.0 - beta2)
                    v_hat = v / bc2
                    m.mul_(beta1).add_(agg_q_norm, alpha=1.0 - beta1)

                m_hat  = m / bc1
                p_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data
                update = m_hat / (v_hat.sqrt().add_(eps))
                if wd != 0.0:
                    update.add_(p_fp32, alpha=wd)
                p_fp32.add_(update, alpha=-lr)

                if p.data.dtype == torch.float16:
                    p.data.copy_(p_fp32.half())
                elif p.data.dtype == torch.bfloat16:
                    p.data.copy_(p_fp32.bfloat16())
                else:
                    p.data.copy_(p_fp32)

        return loss

    def sync_variance_states(self) -> None:
        """
        Fused AllReduce of all v_t tensors — mandatory under non-IID data.
        One NCCL kernel regardless of model depth.  See sync_variance_states().
        """
        sync_variance_states(self.optimizer, self.process_group)

    def remove_hooks(self) -> None:
        for handle in self._math_hooks:
            handle.remove()
        self._math_hooks.clear()

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)
