---
layout: default
title: "Chapter 11 - ONNX Bundles to ANE"
---

# Chapter 11 - ONNX Bundles to ANE

Not every model arrives as GGUF. Some local inference stacks ship ONNX bundles:
an ONNX graph, external tensor data, tokenizer files, and runtime metadata. If the
bundle is already optimized for local execution, the graph may also contain
runtime-specific quantization operators instead of plain `MatMul` nodes.

This chapter describes how to convert that kind of local ONNX bundle into a
CoreML model that runs on the Apple Neural Engine. The worked code path in this
repository is `converters/aion3_onnx_to_ane.py`. The code path is concrete; the
lesson is the conversion pattern.

The shape of the workflow is simple:

- start from an ONNX bundle already on disk;
- read the graph structure, metadata, and weight initializers;
- materialize any packed weights needed by the CoreML graph;
- rebuild the decoder in ANE-friendly Torch/CoreML form;
- keep generated model outputs local.

---

## Why ONNX Is Different From GGUF

The GGUF path in Chapter 2 starts from a checkpoint format. The converter owns
almost everything: metadata parsing, weight dequantization, layer construction,
tokenizer export, and CoreML conversion.

An ONNX bundle is different. It already contains an execution graph. That is
useful, because the graph tells us names, shapes, and operator structure. It is
also limiting, because CoreML cannot run every ONNX operator directly, and it
does not understand runtime-specific contrib operators from other runtimes.

So the job changes from "read a checkpoint" to "recover the model behind the
graph." The converter needs to answer a few questions:

1. What architecture is represented by the ONNX graph?
2. How are its weights stored?
3. Which ops are ordinary ONNX, and which are runtime-specific?
4. Which parts should become CoreML compute, and which parts should remain host
   runtime work?

For transformer decoders, the target shape is still familiar: Conv2d projections,
RMSNorm, RoPE, attention, MLP, LM head, and stateful KV cache.

---

## Start With Operator Boundaries

Some ONNX models use Microsoft-domain ONNX Runtime contrib operators for compact
local inference. Two examples are:

```text
com.microsoft::MatMulNBits
com.microsoft::GatherBlockQuantized
```

These are ONNX Runtime contrib operators. CoreML does not compile them as-is, so
the converter uses their attributes and inputs to reconstruct equivalent
CoreML-friendly layers.

At conversion time, the important part is what each op says about the stored
weights:

- `MatMulNBits` represents a matrix multiply whose weights are stored in an
  N-bit block-quantized layout.
- `GatherBlockQuantized` represents a gather from a block-quantized table, often
  useful around embeddings or shared embedding/output weights.

Treat them as compact weight formats with explicit attributes and scale tensors:
recover the dense weight matrix, then build the CoreML-friendly equivalent.

In other words:

```text
ONNX Runtime bundle:
  compressed weights + contrib quantization ops

CoreML/ANE converter:
  unpack weights locally + rebuild decoder with Conv2d projections
```

The generated CoreML graph no longer needs those ONNX Runtime contrib ops.

---

## How the Code Is Organized

The converter is split into a few practical pieces:

```text
OnnxWeightStore
  loads ONNX initializers, records quantized weight nodes, and resolves names

QuantSpec + _dequantize_blockwise
  interpret block-quantization metadata and materialize fp16 weights

Stateful decoder layer/model modules
  rebuild the transformer decoder with CoreML-friendly Torch modules

_build_coreml_model
  traces the Torch module and declares CoreML inputs, outputs, and states

_write_runtime_metadata
  writes local metadata consumed by the Swift runtime

runtime/aion3_ane.swift
  owns MLModel, MLState, token buffers, RoPE tables, masks, and decode loop
```

The reusable idea is broader than any one implementation class name: read a local
ONNX transformer graph, recover its weights from operator metadata, and emit a
stateful CoreML decoder.

---

## Resolve the Local Bundle

The converter accepts a bundle path. It may point directly at a directory
containing `model.onnx`, or at a parent directory containing versioned
subdirectories. Example:

```bash
/usr/bin/python3 converters/aion3_onnx_to_ane.py \
  --source-bundle PATH_TO_MODEL_BUNDLE \
  --out-dir models/aion/ane \
  --max-seq-len 2048
```

The converter then checks for the required local files:

```text
model.onnx
external tensor data, if the ONNX model uses it
runtime/model config JSON
tokenizer files, if the runtime needs text input
```

Only the converter logic is source code. The bundle and all generated outputs are
local build state.

---

## Infer the Decoder Configuration

Before building CoreML, recover the dimensions that define the decoder:

```text
hidden size
layer count
attention heads
KV heads
head size
RoPE dimension
context length
vocabulary size
EOS/BOS token IDs
```

Prefer structured metadata when the bundle provides it. If not, these values can
often be cross-checked from ONNX tensor shapes and node attributes. Do not guess
silently. A wrong head count or KV-head count may still produce a graph that
compiles, but it will not be the same model.

Grouped-query attention is a common source of shape mistakes. The query projection
width may be `num_attention_heads * head_size`, while key and value projection
widths are `num_key_value_heads * head_size`. The output projection must match
the query width, not merely the hidden size.

---

## Materialize Quantized Weights

For ordinary ONNX initializers, loading the tensor is enough. For packed
quantized weights, the converter has to read the operator attributes and matching
scale tensors.

A block-quantized matmul usually provides enough information to reconstruct the
matrix:

```text
bits
block_size
K
N
packed uint8 weight data
per-block scales
optional zero-points
```

The exact packing is determined by the op attributes and tensor layout. For a
4-bit layout, each byte stores two quantized values. For an 8-bit layout, each
byte stores one. The converter expands those values, subtracts the zero-point
convention used by the source graph, multiplies by the corresponding scale, and
reshapes the result to `[out_channels, in_channels]`.

Once materialized, the CoreML path is ordinary:

```python
conv = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
conv.weight = torch.nn.Parameter(weight.reshape(out_channels, in_channels, 1, 1))
```

This is not a quality improvement step. It is a representation change. The goal
is to preserve the source model's math while expressing it in operations CoreML
can lower to ANE.

---

## Rebuild the Decoder as CoreML-Friendly Torch

The converter reconstructs each decoder layer as a small PyTorch module, then
traces it into CoreML. The ANE-friendly rules are the same ones used throughout
this book:

- linear projections become `Conv2d(..., kernel_size=1)`;
- tensors use 4D layouts where practical;
- KV cache is CoreML state, not a host tensor copied every token;
- RoPE and masks are explicit inputs;
- large host allocations are avoided in the runtime.

The layer shape is conventional:

```text
input RMSNorm
q/k/v projection
optional q/k head norms, if present in the source graph
RoPE
grouped-query attention
output projection
post-attention RMSNorm
gated MLP
residual output
```

Two details are easy to miss.

First, do not drop per-head `q_norm` and `k_norm` if the graph has them. A model
can pass shape checks and still produce poor text if those norms are missing.

Second, RMSNorm must be CoreML-safe. In traced PyTorch, casting to float may look
enough, but CoreML lowering can still overflow if the square is effectively done
in fp16. A scale-invariant RMSNorm implementation can avoid this by scaling the
input before squaring and scaling epsilon by the same factor squared.

---

## Make KV Cache a CoreML State

For decode, each layer owns two state tensors:

```text
k_cache_i: [1, num_kv_heads, max_seq_len, head_size]
v_cache_i: [1, num_kv_heads, max_seq_len, head_size]
```

CoreML exposes these through `ct.StateType` during conversion and `MLState` at
runtime. The graph receives a one-position write mask, updates the current cache
slot, and attends over the compiled cache length.

This shape is simple and reliable, but it has a tradeoff: a graph compiled for a
long context may do more fixed-size work per token than a short interactive use
case needs. For that reason, it is useful to build multiple local variants:

```text
2048-token context for longer prompts
512-token context for interactive use
256-token context for demos and short prompts
```

The weights are the same. Only the compiled state length changes.

---

## Decide What the Model Outputs

A validation build should return full logits. That lets you compare Torch and
CoreML directly:

```text
argmax token
top-k tokens
cosine similarity
mean absolute error
max absolute error
```

A demo or runtime-oriented build may return only `next_token`, with argmax inside
CoreML. This reduces host-side output handling, but it does not remove the LM head
compute itself. Treat it as a runtime convenience, not a substitute for proper
sampling.

The converter supports both shapes because they answer different questions:

```text
logits output:   validate math
argmax output:   simple local demo/runtime path
```

---

## Validate in Layers

Do not start with a full generation demo. Start with parity.

The minimum validation ladder is:

1. **Converter Torch smoke**: materialized weights produce finite logits.
2. **Stateless CoreML probe**: token-0 graph matches Torch without CoreML state.
3. **Stateful CoreML parity**: `MLState` decode step matches Torch.
4. **Swift runtime smoke**: prompt IDs produce nonzero, plausible output.
5. **Text prompt demo**: tokenizer + Swift runtime work together.

This order catches different bugs. A stateless probe isolates CoreML math
lowering. A stateful parity test isolates KV cache behavior. The Swift smoke test
catches runtime buffer, mask, RoPE, and metadata mistakes.

When logits collapse to zero, do not blame quantization first. Check norm
conventions, dtype behavior, residual ranges, and whether source graph operations
were omitted. A one-line RMSNorm or head-norm mismatch can look like model
quality loss.

---

## Source Inputs and Build Outputs

The converter and runtime are source files. The ONNX bundle and generated CoreML
artifacts are build inputs and outputs.

Keep in source control:

```text
converter source
Swift runtime source
validators
demo scripts
book notes
```

Keep out of source control:

```text
source ONNX files
external tensor data
tokenizer files copied into the output directory
embedding binaries
CoreML .mlpackage or .mlmodelc outputs
generated demo recordings
compiled Swift binaries
```

Use `PATH_TO_MODEL_BUNDLE` for the local source path and keep the generated
outputs under the chosen `--out-dir`.

---

## What This Gives You

The end state is a local runtime that looks like the rest of the ANE stack in
this book:

```text
local ONNX bundle
    -> converter materializes weights and rebuilds CoreML graph
    -> CoreML package with stateful KV cache
    -> Swift runtime with reusable buffers
    -> local token generation on Apple Neural Engine
```

This does not make the model smarter. It changes where and how the model runs.
For the right class of small local models, that is enough to change the product
shape: lower latency, no network dependency for the local path, and inference
that can live next to the user's data instead of across the wire.
