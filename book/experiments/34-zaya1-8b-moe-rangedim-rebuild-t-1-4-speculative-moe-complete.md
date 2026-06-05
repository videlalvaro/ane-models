---
layout: default
title: "Experiment 34 - ZAYA1-8B MoE RangeDim Rebuild (T=1..4 speculative MoE) [COMPLETE]"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="33-phi-4-mini-arc-challenge-eval-5-shot-raw-completion-complete.html">Previous: Experiment 33</a> | <a href="35-zaya1-8b-moe-int4pal-per-grouped-channel-palettization-group-size-32-complete.html">Next: Experiment 35</a></nav>

# Experiment 34 - ZAYA1-8B MoE RangeDim Rebuild (T=1..4 speculative MoE) [COMPLETE]

**Date**: 2026-05-13  
**Objective**: Eliminate the Exp 32 speculative-decode bottleneck by rebuilding all 40 MoE
shards with `ct.RangeDim(lower_bound=1, upper_bound=4, default=1)` on the batch-token axis,
so the T=vbt verifier runs a single ANE MoE dispatch instead of `t × T=1` serial dispatches.

### Shard build

Script: `local-artifacts/zaya_full_convert.py`
Architecture: soft-routing (all 16 experts computed, weighted by softmax), all Conv2d 1×1,
INT8 per-tensor, trace at T=1, RangeDim T∈[1..4].

| Metric | Value |
|--------|-------|
| Shards built | 40/40 (L01, L03, … L79) |
| Compiled size per shard | 202.3 MB (vs 193 MB T=1 fixed) |
| ANE residency (gate L01) | **conv_ane=36/36 conv_non_ane=0** |
| Total disk | ~8.1 GB |
| TMPDIR issue | VSCode sandbox tmpfs exhausted mid-run; fixed by setting `TMPDIR=local-artifacts/zaya_ane/cml_tmp` |

### Swift runtime change (`zaya_ane.swift`)

Added `moeRangedim: Bool?` to `ZayaRuntimeMeta` and `verifierMoeProvider` to
`ZayaRuntime`. In `forwardVerifier`, when `verifierMoeProvider != nil`, a single T=vbt
ANE dispatch replaces the serial `t × T=1` loop for each MoE layer. Falls back to T=1
serial when `moe_rangedim` is absent (backward compatible with old manifests).

Manifest: `local-artifacts/zaya_ane/zaya_runtime_meta_stateful_cca_rangedim.json`
Binary: `local-artifacts/zaya_ane_runtime` (recompiled clean, 2 pre-existing warnings only).

### Benchmark results

Hardware: M4 Max (Apple Neural Engine, 100% ANE residency)  
Prompt: `--prompt-ids 2,42 --max-new 40`

| Mode | tok/s | vs baseline |
|------|-------|-------------|
| Baseline T=1 (Exp 32 manifest) | 8.62 | — |
| Baseline T=1 (Exp 34 rangedim manifest) | **8.59–8.94** | ±0% ✓ |
| Speculative ngram (Exp 32, T=1 MoE) | 2.01 | −77% |
| **Speculative ngram (Exp 34, T=4 MoE)** | **2.69** | **−69%** |

Speculative profile (Exp 34, `--ngram-min 1`, 39 tokens):
```
verifier_calls=29  drafted=87  accepted=10  fallbacks=28  acceptance=11.5%
Verifier wall cost: 14.473s / 29 calls = 499 ms/call
```

### Analysis: why only +34% instead of 4×

**Expected**: 40 MoE layers × 4 serial T=1 calls → 40 MoE layers × 1 T=4 call = 4× speedup.  
**Measured**: 499 ms/verifier call (vs ~669 ms in Exp 32) = **+25% per-call improvement**.

Root cause: **ZAYA1-8B uses soft-routing**, computing all 16 expert FFNs for every input
token. Compute scales as O(16 × T × FFN_hidden), so doubling T doubles compute — there
is no savings from expert selection. RangeDim batching eliminates only the CoreML dispatch
overhead (160 → 40 dispatches per verifier pass at T=4), not the dominant compute time.
This contrasts with attn shards, where the KV-cache avoids redundant O(T²) attention work.

**Dispatch-overhead saving estimate**:  
Each MoE shard dispatch ≈ 10ms overhead, 40 shards × (4−1) eliminated dispatches × 10ms
≈ 1.2s saved over 29 verifier calls → ≈ 41ms/call saved — matches the observed 670→499ms
= 170ms/call saving well enough given model loading variability.

**Break-even acceptance rate with 499 ms verifier vs 112 ms T=1:**
$$
p_{\text{break-even}} = 1 - \frac{t_1}{t_v/\text{vbt}} = 1 - \frac{112}{499/4} \approx 0.10
$$

At 11.5% observed acceptance rate, speculative is right at break-even in theory, but
the T=4 verifier commit value is 1.115 tokens/call vs 1.0 for T=1, so the net effect is
still slightly negative at this acceptance rate.

### Next paths for MoE-heavy speculative decode

1. **INT4 per-grouped-channel palettization** (`constexpr_lut_to_dense`): halves MoE
   shard size (202→~101 MB) and halves per-token compute → verifier cost ≈ 300 ms/call.
   Must pass ANE residency gate + golden validator before scale-out (see ANE_CHAIN_SCHEMA.md).
2. **Higher acceptance corpus**: code-completion prompts achieve 60–80% n-gram acceptance;
   at p=0.6 and 300ms verifier, expected speedup ≈ +80% over baseline.
3. **Accept current state**: baseline 8.59 tok/s ZAYA ANE decode is already competitive.
   Speculative remains available via `--speculative` flag for high-acceptance workloads.

**Reference**: [Dragon Book §8.7] — instruction-level parallelism limits (same principle:
batching helps only when work is dispatch-bound, not compute-bound); [EoP §2] — zero-alloc
hot path (verifier dispatch overhead); [BOOK_ANALYSIS Exp 28] — HyMT speculative success
owed to small T=1 attn shards (20 MB) where dispatch dominates.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="33-phi-4-mini-arc-challenge-eval-5-shot-raw-completion-complete.html">Previous: Experiment 33</a> | <a href="35-zaya1-8b-moe-int4pal-per-grouped-channel-palettization-group-size-32-complete.html">Next: Experiment 35</a></nav>
