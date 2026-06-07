---
layout: default
title: "Experiment 27 - MicroGPT on ANE — Minimum Size Constraint Discovery"
---

<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="26-multi-token-verifier-feasibility.html">Previous: Experiment 26</a> | <a href="28-hymt-1-8b-rangedim-t-1-4-n-gram-speculative-decode.html">Next: Experiment 28</a></nav>

# Experiment 27 - MicroGPT on ANE — Minimum Size Constraint Discovery

**Date**: 2026-05-03

**Sources**: Dragon Book §8.7 (Peephole Optimization) + Knuth TAOCP Vol. 2 Ch. 4
(arithmetic: numerics, overflow avoidance)

**Context**: Karpathy's MicroGPT (gist / blog post 2026-02-12) is a 200-line
educational GPT with scalar autograd. No pre-trained checkpoint exists — it is a
training script. This experiment builds the full ANE pipeline: train from scratch,
export weights, CoreML conv shard, Swift + Python chat runtime.

**Problem discovered**: The original MicroGPT architecture (`n_embd=16`, `n_head=4`,
`block_size=16`, `n_layer=1`) converted to a `0.03 MB` compiled INT8 shard. Every
op fell to CPU — `conv_ane=0/47`, `compute_ane=0/47`. No error is raised; the ANE
cost model simply refuses sub-threshold graphs.

**Root cause (empirical ANE law)**:
The ANE conv scheduler has a minimum compiled-shard size of approximately **14 MB**
for transformer 1×1-conv graphs. Below this floor the cost model prefers CPU
scheduling regardless of op type. This threshold had been documented in
`ANE_CHAIN_SCHEMA.md` but was never triggered by the Hy-MT or Phi-4 shards
(both well above floor). MicroGPT's toy architecture hit it for the first time.

**Fix — scaling to clear the ANE floor**:

The correct response per project policy is to move compute *onto* ANE, never to
optimise a CPU fallback. The model was scaled to:

| Parameter | Original | Scaled |
|-----------|----------|--------|
| `n_embd` | 16 | 512 |
| `n_head` | 4 | 8 |
| `head_dim` | 4 | 64 |
| `n_layer` | 1 | 6 |
| `block_size` | 16 | 64 |
| Params | ~4,192 | ~18.9 M |
| Compiled INT8 size | 0.03 MB | 19.07 MB |

With `n_embd=512` the shard is comfortably above the 14 MB floor.

**Safe-norm peephole (Dragon Book §8.7)**:
The original RMSNorm implementation accumulates `x²` directly in fp16, which
overflows for large channels. The peephole fix divides by `√d` before squaring,
matching the pattern in `gguf_to_ane.py`:

```python
K   = x.shape[1] ** 0.5          # √d, scalar
xs  = x * (1.0 / K)              # x / √d  — keeps fp16 in range
rms = (xs.pow(2).mean(dim=1, keepdim=True) + eps/(K*K)).rsqrt()
return (xs * rms).half()
```

This is a textbook peephole. The unstable pattern:

$$
\left(\frac{\sum x^2}{d} + \varepsilon\right)^{-1/2}
$$

is rewritten as:

$$
\left(\sum \left(\frac{x}{\sqrt{d}}\right)^2 + \frac{\varepsilon}{d}\right)^{-1/2}
$$

The two forms are mathematically identical, but the second is numerically safe
and preferred by the ANE cost model for norm ops.

**Results**:

- Training: 18.9 M params, 5000 steps, Adam (β=(0.85, 0.99)), linear LR decay,
  dataset = 32,033 baby names (character-level), final loss 1.60.
- CoreML shard: `local-artifacts/microgpt_shards/MicroGPT.mlpackage` + `.mlmodelc` (19.07 MB).
- ANE residency: `conv_ane=37/37`, `compute_ane=260/260`, **PASS=True**, 100% ANE.
- Swift runtime: `local-artifacts/microgpt_ane_runtime`, stateful KV cache
  (`MLState` API), FLOAT16 conv shard, host-side embedding lookup + argmax.
- Benchmark: **~1535 tok/s** warm (500 names, 3352 tokens in 2.18 s).
- Sample output: karrin, avian, ana, alina, jelah, dari — plausible name-like forms.

**Artifacts**:

- `local-artifacts/microgpt_train.py` — PyTorch training script (`.venv`)
- `local-artifacts/microgpt_to_ane.py` — CoreML conversion + compile (Xcode python3)
- `local-artifacts/microgpt_export_runtime.py` — wte/wpe fp16 bin export (`.venv`)
- `local-artifacts/microgpt_ane.swift` / `microgpt_ane_runtime` — Swift CLI
- `python/microgpt_ane_chat.py` — Python wrapper
- `local-artifacts/microgpt_ane/` — weights, vocab JSON, fp16 bins, manifest

**Key empirical law confirmed**: Transformer 1×1-conv shards require **≥14 MB
compiled INT8** for ANE placement. Shards below this threshold fall silently to
CPU. The fix is always to scale the model, not to optimise the CPU path.

---


<nav class="experiment-nav"><a href="../08-experiments.html">Back to Experiment Index</a> | <a href="26-multi-token-verifier-feasibility.html">Previous: Experiment 26</a> | <a href="28-hymt-1-8b-rangedim-t-1-4-n-gram-speculative-decode.html">Next: Experiment 28</a></nav>
