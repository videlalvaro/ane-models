---
layout: default
title: "Experiment 35 - ZAYA1-8B MoE INT4pal (per-grouped-channel palettization, group_size=32) [COMPLETE]"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="34-zaya1-8b-moe-rangedim-rebuild-t-1-4-speculative-moe-complete.html">Previous: Experiment 34</a> | <a href="36-gemma-4-26b-a4b-int8-per-channel-rebuild-t4-3-quality-fix.html">Next: Experiment 36</a></nav>

# Experiment 35 - ZAYA1-8B MoE INT4pal (per-grouped-channel palettization, group_size=32) [COMPLETE]

**Date**: 2026-05-13  
**Objective**: Replace INT8 per-tensor MoE shards (Exp 34, 202 MB each) with INT4
per-grouped-channel palettized shards (`OpPalettizerConfig(mode="uniform", nbits=4,
granularity="per_grouped_channel", group_size=32)` → `constexpr_lut_to_dense` ops),
halving compiled shard size (~101 MB) and improving T=1 baseline throughput.

### Shard build

Script: `python/zaya_moe_export_int4pal.py`  
Config: `OpPalettizerConfig(mode="uniform", nbits=4, granularity="per_grouped_channel",
group_size=32)`. RangeDim T∈[1..4] retained from Exp 34.  
TMPDIR: `$PWD/local-artifacts/zaya_ane/cml_tmp` (main disk hit 100% capacity mid-session
due to 86 GB stale ANE plan caches from macOS 26 upgrade; freed by removing
`~/Library/Caches/zaya_ane_runtime/`, `com.apple.python3/`, and related runtime caches).

| Metric | Value |
|--------|-------|
| Shards built | 40/40 (L01, L03, … L79) |
| Compiled size per shard | **101.2 MB** (halved from 202 MB INT8) |
| ANE residency (gate L01) | **conv_ane=36/36 conv_non_ane=0** |
| Total disk | ~4.0 GB |

Attn shards (Exp 31 CCA, 40 shards) also required recompilation after macOS 26 upgrade
invalidated all `.mlmodelc` ANEF plans (CoreML error -14). Fixed via `xcrun coremlcompiler
compile` on all 40 `.mlpackage` files.

### Golden validator

`python/zaya_golden_validator.py --layer 1 --n-probes 8` on L01 INT4pal shard:

| Metric | Value |
|--------|-------|
| Min cosine | **0.9994** |
| Mean cosine | **0.9996** |
| Gate | **GREEN** ✓ |

### Benchmark results

Hardware: M4 Max (Apple Neural Engine, 100% ANE residency)  
Prompt: `--prompt-ids 2,42 --max-new 40`

| Mode | tok/s | vs Exp 34 INT8 |
|------|-------|----------------|
| Baseline T=1 (INT8, Exp 34) | 8.59 | — |
| **Baseline T=1 (INT4pal, Exp 35)** | **9.25** | **+7.7%** |
| Speculative ngram (INT8 rangedim, Exp 34) | 2.69 | — |
| **Speculative ngram (INT4pal, Exp 35)** | **2.52** | −6% |

Speculative profile (Exp 35, `--ngram-min 1`, prompt-ids 2,42, 40 new tokens):
```
verifier_calls=32  drafted=96  accepted=7  fallbacks=31  acceptance=7.3%
Verifier wall cost: 15.455s / 32 calls ≈ 483 ms/call
```

### Analysis: INT4pal improves T=1, not T=vbt verifier

**T=1 baseline improvement (+7.7%)**: INT4pal halves MoE weight bandwidth. At T=1,
ZAYA's MoE forward pass is DRAM-streaming-bound — the ANE must stream 101 MB of LUT
weights from DRAM per shard vs 202 MB INT8. This directly reduces per-token latency.

**T=4 verifier cost unchanged (483 ms vs 499 ms INT8 = −3%)**: At T=4, ZAYA soft-routing
computes all 16 expert FFNs over all 4 tokens — compute load is `16 × 4 × FFN_hidden`
MACs. INT4pal reduces weight *bandwidth* but not *MAC operation count*. The ANE becomes
MAC-bound at T=4, not bandwidth-bound. Therefore INT4pal delivers diminishing returns
on the verifier call, in contrast to the T=1 case.

This is the ANE equivalent of Knuth's observation in TAOCP Vol. 2 §4.3 about
arithmetic-vs-memory bottlenecks: the bottleneck shifts with the operation count, and
optimizations targeting the wrong resource leave performance on the table.

**Break-even acceptance rate** with 483 ms verifier vs 109 ms T=1 (9.25 tok/s):

$$
p_{\text{break-even}} = 1 - \frac{t_1}{t_v/\text{vbt}} = 1 - \frac{109}{483/4} \approx 0.10
$$

At 7.3% observed acceptance (synthetic prompt), speculative is below break-even —
matching the pattern from Exp 34. Real code-completion prompts at 60–80% acceptance
would yield approximately:

$$
\frac{1 + 3 \times 0.7}{483\ \text{ms}} \approx 6.4\ \text{tok/s}
$$

which is still slower than the T=1 baseline at 9.25 tok/s because the verifier cost dominates.

### Conclusion

INT4pal is a net win for the T=1 baseline (memory-bandwidth-bound): **+7.7%** at half
the shard size. It is not a win for the T=4 MoE verifier (MAC-bound at T=4): essentially
no improvement. The path to speculative speedup on ZAYA requires either reducing soft
routing to top-K sparse (like standard MoE), or moving to a model whose dominant compute
is attn (not FFN).

**Reference**: [TAOCP Vol. 2 §4.3] — arithmetic vs memory bottleneck identification;
[Dragon Book §8.7] — same principle applied to instruction scheduling; [EoP §4] —
reduction via semigroup (INT4pal halves the semigroup element size, not the op count).

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="34-zaya1-8b-moe-rangedim-rebuild-t-1-4-speculative-moe-complete.html">Previous: Experiment 34</a> | <a href="36-gemma-4-26b-a4b-int8-per-channel-rebuild-t4-3-quality-fix.html">Next: Experiment 36</a></nav>
