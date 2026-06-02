# ane-book

**Production LLM inference on the Apple Neural Engine** — a practitioner's guide,
complete with converters, Swift runtimes, and validated model manifests.

Every model in this repo runs **100% on the Neural Engine** (verified with
`MLComputePlan`). No GPU fallback. No CPU matmuls.

---

## Models

| Model | Type | Params | ANE tok/s | Status |
|-------|------|--------|-----------|--------|
| [Phi-4-mini-instruct](models/phi4-mini/) | Dense LLM | 3.8B | ~17 | ✅ v1.0 |
| [Hy-MT 1.5](models/hymt/) | Dense translation | 1.8B | ~34 | ✅ v1.0 |
| [ZAYA1-8B](models/zaya/) | MoE LLM | 8B | ~9 | ✅ v1.0 |
| [Privacy Filter](models/privacy-filter/) | MoE NER / PII | ~1.5B | ~24.6 sent/s | ✅ v1.0 |

Hardware: Apple M4 Max, 48 GB unified memory, macOS 15, Xcode 16.

---

## Quick Start

### Prerequisites
- macOS 15+ (Sequoia), Apple Silicon
- Xcode 16+ (for `xcrun coremlcompiler` and coremltools 9)
- Python 3.11+ via Xcode tools

### Run the Privacy Filter demo

```bash
# Build once (downloads weights from HuggingFace, ~3 GB)
/usr/bin/python3 models/privacy-filter/build_scripts/build_pf_packed_alllayers.py

# Extract Swift weights
/usr/bin/python3 models/privacy-filter/build_scripts/extract_pf_swift_weights.py

# Redact a file
bash demo/demo_redact.sh demo/pii_examples.txt
```

### Convert Phi-4-mini from GGUF

```bash
# Download GGUF (requires HuggingFace account for Phi-4)
# Place at: models/phi4-mini/Phi-4-mini-instruct.Q8_0.gguf

# Convert all shards (Xcode python3 only)
/usr/bin/python3 converters/phi4_mini_rangedim_export_shard.py --all

# Convert LM head shards
/usr/bin/python3 converters/phi4_mini_lm_head_shards.py

# Check ANE residency
/usr/bin/python3 validators/phi4_mini_residency_check.py
```

### Convert Aion from Edge Canary's ONNX bundle

```bash
# Use your local downloaded model bundle directory
/usr/bin/python3 converters/aion3_onnx_to_ane.py \
   --source-bundle PATH_TO_MODEL_BUNDLE \
   --out-dir models/aion/ane \
   --max-seq-len 2048
```

This script copies the tokenizer/config files from the Edge bundle, rebuilds the
Qwen3-shaped model as a stateful CoreML package, and compiles the result for
ANE-targeted execution.

The output directory is local build state. The repository contains the converter
and runtime, not Microsoft's downloaded model bundle, tokenizer, embedding
binary, or compiled CoreML package.

For fastest correctness validation while iterating, pass `--no-int8` to skip
extra post-conversion quantization. The Edge bundle's ONNX weights are already
block-quantized and dequantized by the converter.

For runtime-oriented builds that do not need full logits on the host, add
`--argmax-output`. This keeps the same model weights and quantization, but
returns a scalar `next_token` from CoreML instead of the full vocab logits.

Short-context packages are the most effective no-quantization speed lever because
the current stateful graph attends over its compiled KV-cache length. Build them
into separate directories so you can pick the latency/context tradeoff per use
case:

```bash
/usr/bin/python3 converters/aion3_onnx_to_ane.py \
   --source-bundle PATH_TO_MODEL_BUNDLE \
   --out-dir models/aion/ane-256 \
   --max-seq-len 256 \
   --no-int8 \
   --argmax-output
```

```bash
# Validate Torch/CoreML parity for one token
/usr/bin/python3 validators/aion_torch_coreml_parity.py --token-id 1 --max-seq-len 2048

# Build the Swift host runtime
swiftc -O runtime/aion3_ane.swift -framework CoreML -framework Foundation -o runtime/aion3_ane_runtime

# Run a text prompt through the tokenizer and ANE runtime
/usr/bin/python3 runtime/aion3_prompt.py "Hello, who are you?" --max-new 16 --warmup 1

# Run against a short-context optimized package
/usr/bin/python3 runtime/aion3_prompt.py "Hello, who are you?" \
   --meta models/aion/ane-256/aion_runtime_meta.json \
   --max-new 16 \
   --warmup 1

# Recording-friendly local proof demo with transparent runtime evidence
./demo/aion_ane_demo.py --verbose --raw
```

The demo uses greedy argmax decoding plus a repeated n-gram stop. Keep the
default short proof prompt for a reliable recording, or test custom prompts with
`--delay 0 --verbose --raw` before capturing a polished take.

To regenerate the local terminal GIF:

```bash
vhs demo/aion_ane_demo.tape
```

Current smoke test on the corrected stateful graph: Torch/CoreML argmax both
select token `3575` with cosine `0.997848`; the Swift runtime generates coherent
text at about `16 tok/s` decode for a short prompt on ANE.

Measured no-quantization runtime tiers for the same short prompt, using argmax
output and a 64-token decode smoke:

| Package | Max context | Decode speed |
|---------|-------------|--------------|
| `models/aion/ane` | 2048 | ~15 tok/s |
| `models/aion/ane-512` | 512 | ~21 tok/s |
| `models/aion/ane-256` | 256 | ~22.4 tok/s |
| `models/aion/ane-128` | 128 | ~22.8 tok/s |

---

## The Apple Neural Engine Inference Book

The Apple Neural Engine Inference Book in `book/` is a chapter-by-chapter porting guide for
practitioners who want to port their own models to ANE:

| Chapter | Topic |
|---------|-------|
| [00 — Modern Inference](book/00-why-ane.md) | Tokens, prefill/decode, KV cache, ANE vs GPU vs CPU, the Conv2d trick |
| [01 — ANE Laws](book/01-ane-laws.md) | Empirical rules: shard limits, quantization, residency |
| [02 — Porting Recipe](book/02-porting-recipe.md) | GGUF → CoreML, step by step |
| [03 — Quantization](book/03-quantization.md) | INT8 production, INT4 tradeoffs, the silent CPU fallback |
| [04 — Shard Sizing](book/04-shard-sizing.md) | Layer count vs size, 250 MB limit, LM-head splits |
| [05 — Stateful KV Cache](book/05-stateful-kv-cache.md) | MLState, Swift daemon design, decode loop |
| [06 — RangeDim + Speculative](book/06-rangedim-speculative.md) | Variable T, n-gram acceptance |
| [07 — MoE on ANE](book/07-moe-on-ane.md) | Soft routing, per-expert dispatch, ZAYA & Privacy Filter |
| [08 — Swift Runtime](book/08-swift-runtime.md) | Cache-friendly CoreML orchestration, state, buffers, and serving |
| [09 — Experiment Index](book/08-experiments.md) | Searchable index of experiment writeups |
| [10 — Decision Journal](book/09-journal.md) | The thinking behind the hard calls |
| [Glossary](book/glossary.md) | Definitions for inference, CoreML, ANE, and validation terms |

---

## Repository Structure

```
ane-book/
├── book/           ← the porting guide (chapters 00–10)
├── converters/     ← Python scripts for GGUF → CoreML (Xcode python3)
├── runtime/        ← Swift inference runtimes
├── models/         ← per-model manifests, goldens, build scripts
│   ├── phi4-mini/
│   ├── hymt/
│   ├── zaya/
│   └── privacy-filter/build_scripts/
├── validators/     ← residency checks + quality gates
├── demo/           ← end-user demos
├── research/       ← findings, negative results, ANE internals
└── blogposts/      ← published and draft blog posts
```

---

## Key Invariants

1. **ANE-only**: every matmul, norm, and activation runs on the Neural Engine.
   `MLComputePlan` must show 100% `ios18.conv` ops on ANE before any benchmark.

2. **Quality before perf**: cosine similarity ≥ 0.97 vs FP16 golden before
   any benchmarking or model shipping.

3. **INT8 per-tensor is the production baseline**. INT4 per-block silently falls
   to CPU on small shards — see [research/INT4_SHARD_ANE_BUG.md](research/INT4_SHARD_ANE_BUG.md).

4. **Shard size ≤ 250 MB**. Above this, ANEF compiler emits error -14.

---

## Research

`research/` contains findings that don't fit in the how-to chapters:

- [ANE_CHAIN_SCHEMA.md](research/ANE_CHAIN_SCHEMA.md) — public-safe ANE execution-model notes from black-box CoreML experiments
- [ANE_SCALING_FINDINGS.md](research/ANE_SCALING_FINDINGS.md) — 0.5B → 3B scaling limits
- [INT4_SHARD_ANE_BUG.md](research/INT4_SHARD_ANE_BUG.md) — The silent CPU fallback with INT4 per-block

The runnable code in this repository uses public CoreML APIs only. Research
notes may summarize black-box observations of CoreML and ANE behavior, but the
converters, validators, and Swift runtimes do not require unsupported Apple
frameworks, unsupported entitlements, or direct ANE driver access.

---

## License

See [LICENSE](LICENSE).
