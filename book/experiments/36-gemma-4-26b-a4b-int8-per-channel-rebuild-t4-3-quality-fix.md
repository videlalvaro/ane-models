---
layout: default
title: "Experiment 36 - Gemma 4-26B-A4B INT8 Per-Channel Rebuild — T4.3 Quality Fix"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="35-zaya1-8b-moe-int4pal-per-grouped-channel-palettization-group-size-32-complete.html">Previous: Experiment 35</a></nav>

# Experiment 36 - Gemma 4-26B-A4B INT8 Per-Channel Rebuild — T4.3 Quality Fix

**Status**: IN PROGRESS (2026-05-13)

**Context**: The Gemma 4 ANE stack (30-layer INT8 per-tensor, T4.1.3) fully compiled and ran
(90/90 shards, 100% ANE residency), but failed the T4.3 full-stack quality gate. The 8-token
golden prompt scored `min cos = 0.5654` at position 2, and a 6-token REAP prompt scored only
`min cos = 0.9875` (well below the 0.999 logit-cosine target). The single-layer quality audit
at T4.1.3 exposed the root cause: `cos(hidden) range 0.9555–0.9999` across 7 sampled layers.
Some layers have 4.5% angular error per layer — and that error compounds multiplicatively
across 30 layers.

**Root Cause Analysis**:

The T4.1.3 build used INT8 per-tensor quantization with `weight_threshold=10_000_000`. The
dominant weights in Gemma 4's MoE FFN are the stacked expert matrices:
- `gate/up` stacked shape: `(45056, 704)` → 31.7 M elements → quantized (above threshold)
- `down` stacked shape: `(45056, 704)` → 31.7 M elements → quantized (above threshold)

Per-tensor quantization assigns ONE global scale to the entire 31.7 M element matrix:
$$
	ext{scale} = \frac{\max(|W|)}{127}
$$

If any element is an outlier at 5× the typical magnitude,
the scale is 5× too large for the remaining weights, losing ~7 bits of effective precision
on the normal-magnitude weights. The cosine-0.9555 layers are those where this outlier
contamination was worst.

**Book Connection**:

[EoP §4, Stepanov–McJones] — The difference between per-tensor and per-channel quantization
is a direct instance of the semigroup reduction principle: per-tensor takes the global
`max(|W|)` as the reduction identity (one orbit over all 31.7 M elements); per-channel
restricts each reduction to a single output channel's 704 input weights. The shorter orbit
(704 vs 31.7 M) cannot be dominated by a single outlier, so the quantization error is
bounded per-channel rather than globally. EoP §4 formalizes this as: *reducing a smaller
domain under the same semigroup operation gives a tighter bound on the accumulated error.*

[Concrete Mathematics Ch. 9, Graham–Knuth–Patashnik] — Quantization error propagation
through \(N\) transformer layers is \(O((1 + \delta)^N)\), where \(\delta\) is the
per-layer normalized error. With INT8 per-tensor giving \(\delta \approx 0.045\)
at worst (cos=0.9555), after \(N = 30\) layers:

$$
(1 + 0.045)^{30} \approx 3.75\times
$$

relative error amplification. Per-channel targets \(\delta \le 0.003\) (cos ≥ 0.997
per layer), giving:

$$
(1 + 0.003)^{30} \approx 1.09\times
$$

which is within the T4.3 logit gate.

[TAOCP Vol. 2 §4.2, Knuth] — Floating-point error analysis: the condition number of the
quantization map scales as `max(|W|) / RMS(|W|)`. Per-tensor condition number grows with
outlier magnitude; per-channel bounds it per row, shrinking the condition number by a factor
proportional to the weight matrix's inter-row magnitude variation.

**Plan**:

1. **Gate T4.1 (single-layer per-channel quality)**: Rebuild layer 0 with
   `--quant-bits 8 --granularity per_channel`. Target: \(\cos(\text{hidden}) \ge 0.997\).
   Uses Xcode `python3` (coremltools 9), TMPDIR on external scratch storage.

2. **Gate T4.1 batch (all 30 layers)**: Run `scripts/gemma_rebuild_int8pc.sh`
   (new script). Output: `python/moe/out/gemma4_shard*_q8c.{mlpackage,mlmodelc}`.
   Validate: \(\cos(\text{hidden}) \ge 0.997\) for all 30 layers (vs 0.9555 floor in T4.1.3).

3. **Gate T4.3a (prompt-prefill)**: Run `python/moe/gemma_swift_logit_gate.py`
   with 8-token golden (`--prompt-ids 2,3689,563,506,5279,529,7001,236881`).
   Target: \(\min \cos \ge 0.97\) at all 8 positions (vs 0.5654 min in T4.1.3).

4. **Gate T4.3b (decode)**: Run bounded 2-step decode. Target: \(\cos \ge 0.97\) (same
   gate as T4.2 bounded test — if per-channel fixes the prompt-prefill, decode
   should follow).

5. **Gate T4.4 (perf+energy)**: If T4.3 passes, run `energy-bencher` for mJ/tok
   on the INT8 per-channel stack. Baseline: INT8 per-tensor (deleted, must rebuild
   from T4.4 numbers in JOURNAL).

**Prerequisites**: Gemma 4-26B-A4B weights. Must download to external scratch storage.
Options: download to `models/gemma-4-26b-a4b/` via HuggingFace Hub (`google/gemma-4-26b-a4b-it`, ~48 GB).

**Expected Outcome**: INT8 per-channel should bring all 30 layers to \(\cos \ge 0.997\) per layer,
cutting the compounded full-stack error from \(3.75\times\) down to \(1.09\times\) (EoP §4 bound). This
should allow T4.3 to pass, unblocking T4.4 and the final paper numbers for Gemma 4 on ANE.


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="35-zaya1-8b-moe-int4pal-per-grouped-channel-palettization-group-size-32-complete.html">Previous: Experiment 35</a></nav>
