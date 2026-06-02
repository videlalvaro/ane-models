---
layout: default
title: "Book README"
---

# The Apple Neural Engine Inference Book

A practitioner's guide to production inference on the Apple Neural Engine.
The book documents the practical path from model weights to ANE-resident CoreML
graphs, Swift runtimes, validation gates, and the engineering tradeoffs found
while porting real LLMs.

## Read Online

This folder is configured as the source for the repository's GitHub Pages site.
When Pages is enabled for the repository, the rendered book is available at:

<https://videlalvaro.github.io/ane-book/>

## Chapters

| Chapter | Topic |
|---------|-------|
| [00 - Modern Inference](00-why-ane.md) | Tokens, prefill/decode, KV cache, ANE vs GPU vs CPU, the Conv2d trick |
| [01 - ANE Laws](01-ane-laws.md) | Empirical rules: shard limits, quantization, residency |
| [02 - Porting Recipe](02-porting-recipe.md) | GGUF to CoreML, step by step |
| [03 - Quantization](03-quantization.md) | INT8 production, INT4 tradeoffs, the silent CPU fallback |
| [04 - Shard Sizing](04-shard-sizing.md) | Layer count vs size, compiler limits, LM-head splits |
| [05 - Stateful KV Cache](05-stateful-kv-cache.md) | MLState, Swift daemon design, decode loop |
| [06 - RangeDim + Speculative](06-rangedim-speculative.md) | Variable token axes, prefill batching, n-gram speculation |
| [07 - MoE on ANE](07-moe-on-ane.md) | Soft routing, expert shards, ZAYA and Privacy Filter |
| [08 - Swift Runtime](08-swift-runtime.md) | Cache-friendly CoreML orchestration, state, buffers, and serving |
| [09 - Experiment Index](08-experiments.md) | Searchable index of experiment writeups |
| [10 - Decision Journal](09-journal.md) | Design decisions and the reasoning behind them |
| [11 - ONNX Bundles to ANE](10-onnx-to-ane.md) | ONNX Runtime contrib ops, local weight materialization, CoreML rebuilds |
| [Glossary](glossary.md) | Definitions for inference, CoreML, ANE, and validation terms |

## What This Book Covers

- CoreML graph shapes that keep transformer compute on the Apple Neural Engine.
- The modern inference loop: tokens, prefill, decode, logits, sampling, and KV cache.
- Quantization choices that preserve quality without triggering CPU fallback.
- Shard sizing rules for compiler reliability and ANE residency.
- Stateful KV-cache runtimes using public `MLState` APIs.
- Cache-friendly Swift runtime design for warm decode and serving.
- RangeDim and speculative decoding patterns for better throughput.
- MoE-specific lessons from ZAYA and the Privacy Filter runtime.
- ONNX bundle conversion through ONNX Runtime contrib ops and CoreML rebuilds.

## Evidence Standard

The strongest claims in this book should be traceable to one of three evidence
levels:

1. **Checked-in artifact evidence**: converter, runtime, validator, manifest, or
   research note in this repository.
2. **Experiment journal evidence**: an audit-trail entry recording a measurement
   or decision that may not include every generated artifact.
3. **External/author observation**: a useful lesson from adjacent experiments;
   these should be labeled when the supporting artifact is not checked in.

When a model family has multiple artifact variants, keep dimensions, manifests,
converters, and runtime notes from the same variant together.

## Repository Context

The surrounding repository contains the converters, validators, Swift runtimes,
model manifests, and demos referenced by the book. Start from the top-level
[README](../README.md) for setup instructions and model-specific entry points.

## License

The book and repository are released under the [MIT License](../LICENSE).