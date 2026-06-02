---
layout: default
title: The Apple Neural Engine Inference Book
---

# The Apple Neural Engine Inference Book

A practitioner's guide to production inference on the Apple Neural Engine
with CoreML, Swift runtimes, ANE-only residency checks, and validated model
manifests.

By Alvaro Videla - [@old_sound](https://x.com/old_sound)

## Chapters

| Chapter | Topic |
|---------|-------|
| [00 - Modern Inference](00-why-ane.html) | Tokens, prefill/decode, KV cache, ANE vs GPU vs CPU, the Conv2d trick |
| [01 - ANE Laws](01-ane-laws.html) | Empirical rules: shard limits, quantization, residency |
| [02 - Porting Recipe](02-porting-recipe.html) | GGUF to CoreML, step by step |
| [03 - Quantization](03-quantization.html) | INT8 production, INT4 tradeoffs, the silent CPU fallback |
| [04 - Shard Sizing](04-shard-sizing.html) | Layer count vs size, 250 MB limit, LM-head splits |
| [05 - Stateful KV Cache](05-stateful-kv-cache.html) | MLState, Swift daemon design, decode loop |
| [06 - RangeDim + Speculative](06-rangedim-speculative.html) | Variable T, n-gram acceptance |
| [07 - MoE on ANE](07-moe-on-ane.html) | Soft routing, per-expert dispatch, ZAYA and Privacy Filter |
| [08 - Swift Runtime](08-swift-runtime.html) | Cache-friendly CoreML orchestration, state, buffers, and serving |
| [09 - Experiment Index](08-experiments.html) | Searchable index of experiment writeups |
| [10 - Decision Journal](09-journal.html) | The thinking behind the hard calls |
| [11 - ONNX Bundles to ANE](10-onnx-to-ane.html) | ONNX Runtime contrib ops, local weight materialization, CoreML rebuilds |
| [Glossary](glossary.html) | Definitions for inference, CoreML, ANE, and validation terms |

## Repository

The source code, converters, Swift runtimes, validators, and model manifests live
in the [ane-book repository](https://github.com/videlalvaro/ane-book).