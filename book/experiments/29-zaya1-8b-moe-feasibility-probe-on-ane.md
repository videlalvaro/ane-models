---
layout: default
title: "Experiment 29 - ZAYA1-8B MoE Feasibility Probe on ANE"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="28-hymt-1-8b-rangedim-t-1-4-n-gram-speculative-decode.html">Previous: Experiment 28</a> | <a href="30-zaya1-8b-stateful-attn-shards-kv-cache-on-ane.html">Next: Experiment 30</a></nav>

# Experiment 29 - ZAYA1-8B MoE Feasibility Probe on ANE

**Date**: 2026-05-12

**Sources**: Sakarovitch *Elements of Automata Theory* — weighted automaton layer
partition (each of 80 layers is a state transition; the feasibility question is
whether all transitions remain on ANE). Dragon Book §8.7 (peephole): skip LM
head during prefill since those logits are discarded anyway.

**Context**: ZAYA1-8B (Zyphra) is a 80-layer MoE transformer with alternating
attention (even) and MoE-FFN (odd) layers. Architecture:
`d_model=2048`, `n_attn_heads=16`, `n_kv_heads=2`, `d_head=128`,
`n_experts=16`, `vocab_size=262272`.
The model is unusual: despite 8B total parameters, the activated path per token
is smaller than a dense 8B (top-2 expert routing). This makes it a good ANE
target because the per-shard weight size stays manageable.

**Probe design**:
Each of the 80 layers is exported as a separate `.mlmodelc` shard:
- Even layers (0,2,...,78) → attn shard: simplified `Q→O` projection, no KV
  cache (probe only validates that the weight pattern runs on ANE).
- Odd layers (1,3,...,79) → MoE shard: full routing (16 experts, top-2
  selection) + expert FFN, INT8 symmetric quantisation.
- LM head: 3 shards covering vocab [0,87424), [87424,174848), [174848,262272).

**ANE residency**:
| Shard type | conv_ane/total | PASS |
|------------|---------------|------|
| MoE (L01) | 36/36 | ✓ |
| Attn (L00) | 2/2 | ✓ |
| All 80 layers | all PASS | ✓ |

**End-to-end probe result** (M4 Max, warm JIT cache, 20 decode tokens):

| Metric | Value |
|--------|-------|
| Decode throughput | **9.27 tok/s** |
| Total fwd throughput | 9.73 tok/s |
| Layers (80) total | 1.735s / 20 calls = 86.75ms/token |
| LM head (3 shards) | 0.094s / 20 calls = 4.7ms/token |
| Avg cost per layer | **1.09ms** |

Load time is fast (JIT already cached from prior build session): all 80 shards
load in ≈13s total warm, with MoE shards taking ~0.7-1.0s each (weight mmap +
first ANE dispatch) vs attn shards at ~0.03-0.06s.

**Key finding — MoE dominates attn cost**:
Each MoE shard costs ~0.7ms vs ~0.03ms for attn. With 40 of each, layers break
down as ~28ms MoE + ~1.2ms attn per forward call. The expert routing (top-2 of
16 experts) runs entirely on ANE — the `constexpr_lut` selection pattern stays
ANE-resident. This validates the path for stateful KV-cache shards.

**Shard sizes**: MoE shards are 193MB compiled each; attn shards 4MB each.
Total probe artifact set: 9.2GB on disk (80 individual `.mlmodelc` shards).

**Limitations of probe shards**:
The attn shards implement simplified attention (Q→O only, no KV state, no RoPE)
to isolate the weight residency question from the stateful engineering question.
Generated token IDs are therefore not meaningful as text. The probe result
establishes: (a) all ops run on ANE, (b) MoE routing stays ANE-resident,
(c) 9.27 tok/s is the simplified-attn floor. Real stateful attention will add
KV scatter overhead (same pattern validated in Exp 26 for Phi).

**Next step**: Build stateful attn shards with `max_seq_len=2048`, RoPE, and
KV cache scatter. With d_model=2048, n_kv_heads=2, d_head=128, seq_len=2048 the
KV state per attention layer is:

$$
2 \times 2048 \times 128 \times 2\ \text{bytes} = 1\ \text{MB}
$$

Across 40 layers, that is \(40\ \text{MB}\) total, which is well within ANE DRAM
budget. The RangeDim T=1..4 pattern from Exp 28 applies directly:
`ct.RangeDim(lower_bound=1, upper_bound=4, default=1)`.

**Artifacts**:
- `local-artifacts/zaya_ane/attn/zaya_attn_L{00,02,...,78}.mlmodelc` — 40 simplified attn shards
- `local-artifacts/zaya_ane/moe/zaya_moe_L{01,03,...,79}.mlmodelc` — 40 MoE shards
- `local-artifacts/zaya_ane/lm_head/zaya_lm_head_s{0,1,2}.mlmodelc` — 3 LM head shards
- `local-artifacts/zaya_ane/zaya_runtime_meta.json` — runtime manifest
- `local-artifacts/zaya_ane/zaya_embed.bin` — 1.07 GB fp16 embedding table
- `local-artifacts/zaya_ane.swift` / `zaya_ane_runtime` — probe runtime

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="28-hymt-1-8b-rangedim-t-1-4-n-gram-speculative-decode.html">Previous: Experiment 28</a> | <a href="30-zaya1-8b-stateful-attn-shards-kv-cache-on-ane.html">Next: Experiment 30</a></nav>
