---
layout: default
title: "Experiment 32 - ZAYA1-8B Speculative Decode (T=4 Verifier + n-gram) [IMPLEMENTED; BOTTLENECKED]"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="31-zaya1-8b-cca-conv-qk-gates-wired-into-40-stateful-attn-shards-2025-07-14.html">Previous: Experiment 31</a> | <a href="33-phi-4-mini-arc-challenge-eval-5-shot-raw-completion-complete.html">Next: Experiment 33</a></nav>

# Experiment 32 - ZAYA1-8B Speculative Decode (T=4 Verifier + n-gram) [IMPLEMENTED; BOTTLENECKED]

**Date**: 2025-05  
**Objective**: Port n-gram speculative decode from HyMT (Exp 28) to ZAYA1-8B using the
Exp 31 CCA stateful shards which already carry `rangedim_t_max: 4`.

**Key finding**: ZAYA's MoE-dominated compute makes T=4 batch decode ineffective without
T=4 MoE shards.  The attn layers (40 shards, T=4 enabled) represent only **~15%** of
wall-clock time; MoE layers (40 shards, T=1 fixed) represent **~85%**.

### Architecture analysis

| Compute | Per decode step | T=4 batch behaviour |
|---------|-----------------|---------------------|
| Attn (40 shards, RangeDim T=1..4) | ~15 ms | ~15 ms for 4 tokens (4× cheaper) |
| MoE (40 shards, T=1 fixed) | ~110 ms | 4 × 110 ms = 440 ms (not cheaper) |
| LM head (3 shards, T=1) | ~5 ms | 4 × 5 ms = 20 ms |
| **T=1 total** | **~130 ms/tok = 7.7 tok/s** | — |
| **T=4 verifier total** | — | **475 ms for 4-token batch** |

**Break-even equation** — need:

$$
\frac{1 + 3p}{475\ \text{ms}} > \frac{1}{130\ \text{ms}}
$$

which means:

$$
p > 0.883
$$

That is an **88.3% n-gram acceptance rate required for any speedup**.

Measured at 1.8% acceptance on synthetic prompts.  Even with perfect acceptance
(p=1.0, all 3 draft tokens accepted every call) speedup would only be:

$$
\frac{(1 + 3) \times 130\ \text{ms}}{475\ \text{ms}} = 1.09\times
$$

That is only a 9% improvement.

### Implementation status

`local-artifacts/zaya_ane.swift` — **complete and correct**:
- `--speculative` / `--ngram-min` / `--ngram-max` CLI flags wired
- `forwardVerifier(tokens:posStart:cacheSeqLen:)` — T=vbt attn + t×T=1 MoE interleave
- `speculativeDraft(history:firstToken:)` — n-gram longest-suffix lookup (from HyMT)
- `predictSlotsWithT1Head(count:)` — 3-shard head, slot-by-slot
- `runGenerationSpeculative` — T=vbt chunked prefill + spec decode loop
- Verifier buffers allocated once: `verifierXArr[1,d,4,1]`, `verifierCosArr[4,32]`, etc.

The implementation routes through `runGeneration` when `--speculative` is passed; the
infrastructure is fully in place for when T=4 MoE shards are available.

### Benchmark results

| Mode | Prompt | max_new | Decode tok/s | vs Baseline |
|------|--------|---------|--------------|-------------|
| T=1 baseline | 41-tok | 40 | **7.66** | — |
| `--speculative --ngram-min 1` | 41-tok | 40 | 2.01 | −74% (MoE bottleneck) |

**Acceptance rate**: 1.8% (synthetic prompt; real code prompts may reach 60–80%
but break-even is still 88.3%).

### Conclusion and next step

The `--speculative` flag is implemented and correct.  **Real speedup requires T=4 MoE
shards** (Exp 33).  The ZAYA MoE shard exporter (`local-artifacts/zaya_full_convert.py`) would need
`ct.RangeDim(lower_bound=1, upper_bound=4, default=1)` added to the batch-token axis
and shards recompiled (~40 shards × 193 MB compiled = ~7.7 GB).  With T=4 MoE, the
verifier cost drops from 475 ms → ~130 ms and the break-even acceptance rate falls to
`p > 0` (any n-gram hit is beneficial), matching the HyMT Exp 28 result (+62%).

**Reference**: [EoP §2] — zero-alloc hot path; [Concrete Math Ch.9] — n-gram cost;
[Dragon Book §8] — prefill head-skip optimisation.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="31-zaya1-8b-cca-conv-qk-gates-wired-into-40-stateful-attn-shards-2025-07-14.html">Previous: Experiment 31</a> | <a href="33-phi-4-mini-arc-challenge-eval-5-shot-raw-completion-complete.html">Next: Experiment 33</a></nav>
