#!/usr/bin/env python3
"""Local Aion ONNX bundle → CoreML / ANE.

This converter targets a local Aion ONNX model bundle. The source bundle contains:

- model.onnx + model.onnx.data
- genai_config.json
- tokenizer.json / tokenizer_config.json
- manifest.json

The exported CoreML model follows the same ANE-friendly pattern used in this
repository:

- every projection is Conv2d(1×1)
- KV cache is CoreML state
- inputs are hidden states, not token ids
- output is logits

The result is a `.mlpackage` suitable for `xcrun coremlcompiler compile`.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_model_bundle(base_dir: Path) -> Path:
    if (base_dir / "model.onnx").exists():
        return base_dir
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no model.onnx bundle found under {base_dir}")
    return sorted(candidates)[-1]


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)


def _to_f16(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float16:
        return array
    if array.dtype.kind in {"f", "i", "u", "b"}:
        return array.astype(np.float16)
    raise TypeError(f"unsupported tensor dtype: {array.dtype}")


def _candidate_names(*parts: str) -> list[str]:
    return list(parts)


@dataclass
class DecoderConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_size: int
    context_length: int
    vocab_size: int
    rope_theta: float
    rope_dim: int
    eos_token_ids: list[int]
    bos_token_id: int | None

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_size


@dataclass
class QuantSpec:
    bits: int
    block_size: int
    k: int
    n: int


class OnnxWeightStore:
    """Load ONNX initializers and resolve common transformer-style names."""

    def __init__(self, onnx_path: Path):
        try:
            import onnx
            from onnx import numpy_helper
        except Exception as exc:  # pragma: no cover - import-time dependency check
            raise SystemExit(
                "This converter needs `onnx`. Install it in the Python environment "
                "used to run the script, then retry."
            ) from exc

        self._weights: dict[str, np.ndarray] = {}
        self._quant_specs: dict[str, QuantSpec] = {}
        model = onnx.load(str(onnx_path), load_external_data=True)
        for initializer in model.graph.initializer:
            self._weights[initializer.name] = numpy_helper.to_array(initializer)
        for node in model.graph.node:
            if node.op_type != "MatMulNBits" or len(node.input) < 2:
                continue
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            weight_name = node.input[1]
            if {"bits", "block_size", "K", "N"}.issubset(attrs):
                self._quant_specs[weight_name] = QuantSpec(
                    bits=int(attrs["bits"]),
                    block_size=int(attrs["block_size"]),
                    k=int(attrs["K"]),
                    n=int(attrs["N"]),
                )

    @staticmethod
    def _expand_candidate(candidate: str) -> list[str]:
        expanded = [candidate]
        if candidate.endswith(".weight"):
            expanded.append(candidate.replace(".weight", ".matmul.backbone.weight_quantized"))
            expanded.append(candidate.replace(".weight", ".weight_quantized"))
        return expanded

    @staticmethod
    def _dequantize_blockwise(packed: np.ndarray, scales: np.ndarray, spec: QuantSpec) -> np.ndarray:
        if packed.dtype != np.uint8:
            raise TypeError(f"expected uint8 packed weights, got {packed.dtype}")
        if packed.ndim != 3:
            raise ValueError(f"expected [N, blocks, blob] packed tensor, got shape={packed.shape}")

        out_dim, blocks, blob = packed.shape
        if out_dim != spec.n:
            raise ValueError(f"packed N mismatch: tensor has {out_dim}, spec has {spec.n}")
        expected_blocks = spec.k // spec.block_size
        if blocks != expected_blocks:
            raise ValueError(f"packed block count mismatch: tensor has {blocks}, spec expects {expected_blocks}")

        if scales.ndim == 1:
            if scales.size != spec.n * expected_blocks:
                raise ValueError(
                    f"scale size mismatch: expected {spec.n * expected_blocks}, got {scales.size}"
                )
            scales = scales.reshape(spec.n, expected_blocks)
        elif scales.shape != (spec.n, expected_blocks):
            raise ValueError(
                f"unexpected scale shape {scales.shape}; expected {(spec.n, expected_blocks)}"
            )

        if spec.bits == 8:
            if blob != spec.block_size:
                raise ValueError(f"blob size mismatch for int8: got {blob}, expected {spec.block_size}")
            q = packed.astype(np.int16)
        elif spec.bits == 4:
            expected_blob = spec.block_size // 2
            if blob != expected_blob:
                raise ValueError(f"blob size mismatch for int4: got {blob}, expected {expected_blob}")
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            q = np.stack([low, high], axis=-1).reshape(spec.n, expected_blocks, spec.block_size).astype(np.int16)
        elif spec.bits == 2:
            expected_blob = spec.block_size // 4
            if blob != expected_blob:
                raise ValueError(f"blob size mismatch for int2: got {blob}, expected {expected_blob}")
            q0 = packed & 0x03
            q1 = (packed >> 2) & 0x03
            q2 = (packed >> 4) & 0x03
            q3 = (packed >> 6) & 0x03
            q = np.stack([q0, q1, q2, q3], axis=-1).reshape(spec.n, expected_blocks, spec.block_size).astype(np.int16)
        else:
            raise ValueError(f"unsupported MatMulNBits width: {spec.bits}")

        zp = 1 << (spec.bits - 1)
        deq = (q - zp).astype(np.float32) * scales.astype(np.float32)[..., None]
        return deq.reshape(spec.n, spec.k).astype(np.float16)

    def _materialize(self, name: str, value: np.ndarray) -> np.ndarray:
        if name.endswith("weight_quantized") and value.dtype == np.uint8:
            spec = self._quant_specs.get(name)
            if ".matmul.backbone.weight_quantized" in name:
                scale_name = name.replace(
                    ".matmul.backbone.weight_quantized",
                    ".matmul.quantizers.weight.scale_for_export",
                )
            else:
                scale_name = name.replace(".weight_quantized", ".quantizers.weight.scale_for_export")
            scale = self._weights.get(scale_name)
            if scale is None:
                raise KeyError(f"missing quantization scale tensor: {scale_name}")
            if spec is None:
                raise KeyError(f"missing MatMulNBits quantization spec for: {name}")
            return self._dequantize_blockwise(value, scale, spec)
        return _to_f16(value)

    def get_any(self, candidates: Sequence[str], *, required: bool = True) -> np.ndarray | None:
        for candidate in candidates:
            for expanded in self._expand_candidate(candidate):
                value = self._weights.get(expanded)
                if value is not None:
                    return self._materialize(expanded, value)
        for candidate in candidates:
            suffixes = self._expand_candidate(candidate)
            matches = [
                (key, value)
                for key, value in self._weights.items()
                if any(key.endswith(suffix) for suffix in suffixes)
            ]
            if len(matches) == 1:
                key, value = matches[0]
                return self._materialize(key, value)
        if required:
            raise KeyError(f"unable to resolve tensor from candidates: {list(candidates)}")
        return None

    def export_embed_bin(self, out_path: Path) -> tuple[int, int]:
        emb = self.get_any([
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "token_embd.weight",
            "transformer.wte.weight",
            "lm_head.weight",
        ])
        if emb.ndim != 2:
            raise ValueError(f"embedding tensor must be 2D, got {emb.shape}")
        emb.astype(np.float16).tofile(out_path)
        return int(emb.shape[0]), int(emb.shape[1])

    def export_rope_bins(self, out_dir: Path, max_seq_len: int) -> tuple[str, str, tuple[int, int]]:
        cos = self.get_any([
            "model.layers.0.self_attn.cos_cached_export",
            "layers.0.self_attn.cos_cached_export",
        ])
        sin = self.get_any([
            "model.layers.0.self_attn.sin_cached_export",
            "layers.0.self_attn.sin_cached_export",
        ])
        if cos.shape != sin.shape or cos.ndim != 2:
            raise ValueError(f"unexpected RoPE cache shapes: cos={cos.shape}, sin={sin.shape}")
        if cos.shape[0] < max_seq_len:
            raise ValueError(f"RoPE cache length {cos.shape[0]} is shorter than max_seq_len={max_seq_len}")
        cos_name = "aion_rope_cos.bin"
        sin_name = "aion_rope_sin.bin"
        cos[:max_seq_len].astype(np.float16).tofile(out_dir / cos_name)
        sin[:max_seq_len].astype(np.float16).tofile(out_dir / sin_name)
        return cos_name, sin_name, (int(max_seq_len), int(cos.shape[1]))


def infer_config(bundle: Path) -> DecoderConfig:
    genai = _read_json(bundle / "genai_config.json")
    model = genai["model"]
    decoder = model["decoder"]
    return DecoderConfig(
        hidden_size=int(decoder["hidden_size"]),
        num_hidden_layers=int(decoder["num_hidden_layers"]),
        num_attention_heads=int(decoder["num_attention_heads"]),
        num_key_value_heads=int(decoder["num_key_value_heads"]),
        head_size=int(decoder["head_size"]),
        context_length=int(model["context_length"]),
        vocab_size=int(model["vocab_size"]),
        rope_theta=float(genai.get("rope_theta", genai.get("rope_freq_base", 1_000_000.0))),
        rope_dim=int(genai.get("rope_dim", decoder["head_size"])),
        eos_token_ids=[int(x) for x in model.get("eos_token_id", [])],
        bos_token_id=(int(model["bos_token_id"]) if model.get("bos_token_id") is not None else None),
    )


def _reshape_conv_weight(weight: np.ndarray) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight matrix, got shape={weight.shape}")
    return weight.astype(np.float16).reshape(weight.shape[0], weight.shape[1], 1, 1)


class StatefulDecoderLayer:
    def __init__(self, layer_idx: int, weights: OnnxWeightStore, cfg: DecoderConfig, d_ff: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.torch = torch
        self.nn = nn
        self.F = F
        self.d = cfg.hidden_size
        self.dff = d_ff
        self.q_dim = cfg.num_attention_heads * cfg.head_size
        self.kv_dim = cfg.kv_dim
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.dh = cfg.head_size
        self.rope_dim = cfg.rope_dim
        self.hpk = self.nh // self.nkv
        self.scale = 1.0 / math.sqrt(self.dh)

        layer_prefixes = [
            f"model.layers.{layer_idx}.",
            f"layers.{layer_idx}.",
            f"transformer.h.{layer_idx}.",
        ]

        self.attn_norm = nn.Module()
        self.attn_norm = self._make_rms_norm(weights.get_any(_candidate_names(
            *(f"{p}input_layernorm.weight" for p in layer_prefixes),
            *(f"{p}attn_norm.weight" for p in layer_prefixes),
            *(f"{p}self_attn_norm.weight" for p in layer_prefixes),
        )))

        q_w = weights.get_any(_candidate_names(
            *(f"{p}self_attn.q_proj.weight" for p in layer_prefixes),
            *(f"{p}attn.q_proj.weight" for p in layer_prefixes),
            *(f"{p}q_proj.weight" for p in layer_prefixes),
        ))
        k_w = weights.get_any(_candidate_names(
            *(f"{p}self_attn.k_proj.weight" for p in layer_prefixes),
            *(f"{p}attn.k_proj.weight" for p in layer_prefixes),
            *(f"{p}k_proj.weight" for p in layer_prefixes),
        ))
        v_w = weights.get_any(_candidate_names(
            *(f"{p}self_attn.v_proj.weight" for p in layer_prefixes),
            *(f"{p}attn.v_proj.weight" for p in layer_prefixes),
            *(f"{p}v_proj.weight" for p in layer_prefixes),
        ))
        qkv_w = np.concatenate([q_w, k_w, v_w], axis=0)
        self.q_dim = int(q_w.shape[0])
        self.qkv_conv = nn.Conv2d(self.d, self.q_dim + 2 * self.kv_dim, 1, bias=False)
        self.qkv_conv.weight = nn.Parameter(torch.tensor(_reshape_conv_weight(qkv_w)), requires_grad=False)

        self.q_head_norm = self._make_head_rms_norm(weights.get_any(_candidate_names(
            *(f"{p}self_attn.q_norm.weight" for p in layer_prefixes),
            *(f"{p}q_norm.weight" for p in layer_prefixes),
        ), required=False))
        self.k_head_norm = self._make_head_rms_norm(weights.get_any(_candidate_names(
            *(f"{p}self_attn.k_norm.weight" for p in layer_prefixes),
            *(f"{p}k_norm.weight" for p in layer_prefixes),
        ), required=False))

        o_w = weights.get_any(_candidate_names(
            *(f"{p}self_attn.o_proj.weight" for p in layer_prefixes),
            *(f"{p}attn_output.weight" for p in layer_prefixes),
            *(f"{p}o_proj.weight" for p in layer_prefixes),
        ))
        self.out_conv = nn.Conv2d(self.q_dim, self.d, 1, bias=False)
        self.out_conv.weight = nn.Parameter(torch.tensor(_reshape_conv_weight(o_w)), requires_grad=False)

        self.ffn_norm = self._make_rms_norm(weights.get_any(_candidate_names(
            *(f"{p}post_attention_layernorm.weight" for p in layer_prefixes),
            *(f"{p}ffn_norm.weight" for p in layer_prefixes),
            *(f"{p}mlp_norm.weight" for p in layer_prefixes),
        )))

        gate_w = weights.get_any(_candidate_names(
            *(f"{p}mlp.gate_proj.weight" for p in layer_prefixes),
            *(f"{p}gate_proj.weight" for p in layer_prefixes),
            *(f"{p}ffn_gate.weight" for p in layer_prefixes),
        ))
        up_w = weights.get_any(_candidate_names(
            *(f"{p}mlp.up_proj.weight" for p in layer_prefixes),
            *(f"{p}up_proj.weight" for p in layer_prefixes),
            *(f"{p}ffn_up.weight" for p in layer_prefixes),
        ))
        gate_up_w = np.concatenate([gate_w, up_w], axis=0)
        self.gate_up_conv = nn.Conv2d(self.d, 2 * self.dff, 1, bias=False)
        self.gate_up_conv.weight = nn.Parameter(torch.tensor(_reshape_conv_weight(gate_up_w)), requires_grad=False)

        down_w = weights.get_any(_candidate_names(
            *(f"{p}mlp.down_proj.weight" for p in layer_prefixes),
            *(f"{p}down_proj.weight" for p in layer_prefixes),
            *(f"{p}ffn_down.weight" for p in layer_prefixes),
        ))
        self.down_conv = nn.Conv2d(self.dff, self.d, 1, bias=False)
        self.down_conv.weight = nn.Parameter(torch.tensor(_reshape_conv_weight(down_w)), requires_grad=False)

    def _make_rms_norm(self, weight: np.ndarray):
        import torch
        import torch.nn as nn

        class _RMS(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(w.astype(np.float16)), requires_grad=False)
                self.hidden_size = w.shape[0]

            def forward(self, x):
                norm_scale = 1.0 / 16.0
                x_scaled = x * norm_scale
                x_f = x_scaled.float()
                variance = (x_f * x_f).mean(dim=1, keepdim=True)
                x_normed = (x_f * torch.rsqrt(variance + (1e-5 * norm_scale * norm_scale))).to(x.dtype)
                return x_normed * self.weight.view(1, self.hidden_size, 1, 1)

        return _RMS(weight)

    def _make_head_rms_norm(self, weight: np.ndarray | None):
        import torch
        import torch.nn as nn

        class _Identity(nn.Module):
            def forward(self, x):
                return x

        if weight is None:
            return _Identity()

        class _HeadRMS(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(w.astype(np.float16)), requires_grad=False)
                self.head_size = w.shape[0]

            def forward(self, x):
                norm_scale = 1.0 / 16.0
                x_scaled = x * norm_scale
                x_f = x_scaled.float()
                variance = (x_f * x_f).mean(dim=-1, keepdim=True)
                x_normed = (x_f * torch.rsqrt(variance + (1e-5 * norm_scale * norm_scale))).to(x.dtype)
                return x_normed * self.weight.view(1, 1, self.head_size)

        return _HeadRMS(weight)

    def _apply_rope(self, x_flat, rope_cos_pos, rope_sin_pos, n_heads: int):
        x_r = x_flat.reshape(1, n_heads, self.dh)
        x_rot = x_r[:, :, :self.rope_dim]
        x_pass = x_r[:, :, self.rope_dim:]
        rope_half = self.rope_dim // 2
        x_lo = x_rot[:, :, :rope_half]
        x_hi = x_rot[:, :, rope_half:]
        cos_b = rope_cos_pos.unsqueeze(1)
        sin_b = rope_sin_pos.unsqueeze(1)
        r_lo = x_lo * cos_b - x_hi * sin_b
        r_hi = x_lo * sin_b + x_hi * cos_b
        return self.torch.cat([r_lo, r_hi, x_pass], dim=-1).reshape(1, n_heads * self.dh)

    def forward(self, x, k_cache, v_cache, rope_cos_pos, rope_sin_pos, attn_mask, kv_write_mask):
        residual = x
        normed = self.attn_norm(x)
        qkv = self.qkv_conv(normed).squeeze(-1).squeeze(-1)
        q = qkv[:, :self.q_dim]
        k = qkv[:, self.q_dim:self.q_dim + self.kv_dim]
        v = qkv[:, self.q_dim + self.kv_dim:]

        q = self.q_head_norm(q.reshape(1, self.nh, self.dh)).reshape(1, self.q_dim)
        k = self.k_head_norm(k.reshape(1, self.nkv, self.dh)).reshape(1, self.kv_dim)

        q = self._apply_rope(q, rope_cos_pos, rope_sin_pos, self.nh)
        k = self._apply_rope(k, rope_cos_pos, rope_sin_pos, self.nkv)

        new_k = k.reshape(1, self.nkv, 1, self.dh)
        new_v = v.reshape(1, self.nkv, 1, self.dh)

        k_updated = k_cache * (1.0 - kv_write_mask) + new_k * kv_write_mask
        v_updated = v_cache * (1.0 - kv_write_mask) + new_v * kv_write_mask
        k_cache[:] = k_updated
        v_cache[:] = v_updated

        q_heads = q.reshape(1, self.nh, self.dh)
        attn_parts = []
        for kv_idx in range(self.nkv):
            q_group = q_heads[:, kv_idx * self.hpk:(kv_idx + 1) * self.hpk, :]
            k_head = k_updated[:, kv_idx:kv_idx + 1, :, :]
            v_head = v_updated[:, kv_idx:kv_idx + 1, :, :]
            q_g = q_group.unsqueeze(2)
            k_t = k_head.transpose(-2, -1)
            scores = self.torch.matmul(q_g, k_t) * self.scale
            scores = scores + attn_mask
            attn_w = self.torch.softmax(scores.float(), dim=-1).to(q_g.dtype)
            head_out = self.torch.matmul(attn_w, v_head)
            attn_parts.append(head_out.squeeze(2))

        attn_out = self.torch.cat(attn_parts, dim=1).reshape(1, self.q_dim, 1, 1)
        attn_out = self.out_conv(attn_out)
        x = residual + attn_out

        residual2 = x
        normed2 = self.ffn_norm(x)
        gate_up = self.gate_up_conv(normed2)
        gate = gate_up[:, :self.dff, :, :]
        up = gate_up[:, self.dff:, :, :]
        hidden = self.F.silu(gate.float()).to(gate.dtype) * up
        ffn_out = self.down_conv(hidden)
        return residual2 + ffn_out


class StatefulDecoderModel:
    def __init__(self, weights: OnnxWeightStore, cfg: DecoderConfig, max_seq_len: int):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.d = cfg.hidden_size
        self.rope_half = cfg.rope_dim // 2

        first_gate = weights.get_any([
            "model.layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "transformer.h.0.mlp.gate_proj.weight",
        ])
        self.dff = int(first_gate.shape[0])

        self.layers = [StatefulDecoderLayer(i, weights, cfg, self.dff) for i in range(cfg.num_hidden_layers)]
        self.output_norm = self.layers[0]._make_rms_norm(weights.get_any([
            "model.norm.weight",
            "output_norm.weight",
            "transformer.ln_f.weight",
        ]))

        lm_w = weights.get_any([
            "lm_head.weight",
            "output.weight",
            "model.embed_tokens.weight",
        ])
        self.lm_head = nn.Conv2d(self.d, cfg.vocab_size, 1, bias=False)
        self.lm_head.weight = nn.Parameter(torch.tensor(_reshape_conv_weight(lm_w)), requires_grad=False)

    def module(self, output_kind: str = "logits"):
        import torch
        import torch.nn as nn

        if output_kind not in {"logits", "argmax"}:
            raise ValueError(f"unsupported output_kind: {output_kind}")

        cfg = self.cfg
        layers = self.layers
        output_norm = self.output_norm
        lm_head = self.lm_head
        max_seq_len = self.max_seq_len

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                for layer in layers:
                    self.layers.append(layer.attn_norm)
                    self.layers.append(layer.qkv_conv)
                    self.layers.append(layer.q_head_norm)
                    self.layers.append(layer.k_head_norm)
                    self.layers.append(layer.out_conv)
                    self.layers.append(layer.ffn_norm)
                    self.layers.append(layer.gate_up_conv)
                    self.layers.append(layer.down_conv)
                self.output_norm = output_norm
                self.lm_head = lm_head
                for i in range(cfg.num_hidden_layers):
                    self.register_buffer(
                        f"k_cache_{i}",
                        torch.zeros(1, cfg.num_key_value_heads, max_seq_len, cfg.head_size, dtype=torch.float16),
                    )
                    self.register_buffer(
                        f"v_cache_{i}",
                        torch.zeros(1, cfg.num_key_value_heads, max_seq_len, cfg.head_size, dtype=torch.float16),
                    )

            def forward(self, x, rope_cos_pos, rope_sin_pos, attn_mask, kv_write_mask):
                for i, layer in enumerate(layers):
                    k_cache = getattr(self, f"k_cache_{i}")
                    v_cache = getattr(self, f"v_cache_{i}")
                    x = layer.forward(x, k_cache, v_cache, rope_cos_pos, rope_sin_pos, attn_mask, kv_write_mask)
                x = output_norm(x)
                logits = self.lm_head(x).squeeze(-1).squeeze(-1)
                if output_kind == "argmax":
                    return torch.argmax(logits.float(), dim=1).to(torch.float32)
                return logits

        return _Model()


def _build_coreml_model(bundle: Path, out_dir: Path, max_seq_len: int, quantize_int8: bool, output_kind: str) -> tuple[Path, tuple[int, int]]:
    try:
        import coremltools as ct
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )
        import torch
    except Exception as exc:  # pragma: no cover - import-time dependency check
        raise SystemExit(
            "This converter needs `coremltools` and `torch` from a Core ML-capable Python."
        ) from exc

    onnx_path = bundle / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    cfg = infer_config(bundle)
    weights = OnnxWeightStore(onnx_path)
    embed_shape = weights.export_embed_bin(out_dir / "aion_embed.bin")
    weights.export_rope_bins(out_dir, max_seq_len)
    model = StatefulDecoderModel(weights, cfg, max_seq_len).module(output_kind=output_kind)
    model.half().eval()

    rope_half = cfg.rope_dim // 2
    x_ex = torch.randn(1, cfg.hidden_size, 1, 1, dtype=torch.float16)
    cos_ex = torch.randn(1, rope_half, dtype=torch.float16)
    sin_ex = torch.randn(1, rope_half, dtype=torch.float16)
    attn_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float16)
    attn_mask[0, 0, 0, 0] = 0.0
    kv_write_mask = torch.zeros(1, 1, max_seq_len, 1, dtype=torch.float16)
    kv_write_mask[0, 0, 0, 0] = 1.0

    ct_inputs = [
        ct.TensorType(name="x", shape=(1, cfg.hidden_size, 1, 1), dtype=np.float16),
        ct.TensorType(name="rope_cos", shape=(1, rope_half), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=(1, rope_half), dtype=np.float16),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, max_seq_len, 1), dtype=np.float16),
    ]
    if output_kind == "argmax":
        ct_outputs = [ct.TensorType(name="next_token", dtype=np.float32)]
    else:
        ct_outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    ct_states = []
    for i in range(cfg.num_hidden_layers):
        ct_states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, cfg.num_key_value_heads, max_seq_len, cfg.head_size), dtype=np.float16), name=f"k_cache_{i}"))
        ct_states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, cfg.num_key_value_heads, max_seq_len, cfg.head_size), dtype=np.float16), name=f"v_cache_{i}"))

    with torch.no_grad():
        model(x_ex, cos_ex, sin_ex, attn_mask, kv_write_mask)

    print(f"Converting to CoreML (stateful, max_seq_len={max_seq_len}, output={output_kind}) …")
    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, attn_mask, kv_write_mask))
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if quantize_int8:
        print("Quantizing weights INT8 …")
        mlmodel = linear_quantize_weights(
            mlmodel,
            config=OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "Aion3_Stateful_Argmax" if output_kind == "argmax" else "Aion3_Stateful"
    pkg_path = out_dir / f"{stem}.mlpackage"
    mlmodel.save(str(pkg_path))

    compile_result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg_path.resolve()), str(out_dir.resolve())],
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        raise SystemExit(f"coremlcompiler failed:\n{compile_result.stderr[:1200]}")

    return out_dir / f"{stem}.mlmodelc", embed_shape


def _write_runtime_metadata(bundle: Path, out_dir: Path, max_seq_len: int, embed_shape: tuple[int, int], mlmodelc: Path, output_kind: str) -> None:
    cfg = infer_config(bundle)
    weights = OnnxWeightStore(bundle / "model.onnx")
    rope_cos_bin, rope_sin_bin, rope_cache_shape = weights.export_rope_bins(out_dir, max_seq_len)
    manifest = _read_json(bundle / "manifest.json")
    vocab_from_embed, d_from_embed = embed_shape
    metadata = {
        "model_family": "Aion-1.0-Instruct",
        "source_bundle": str(bundle),
        "source_manifest": manifest,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_size": cfg.head_size,
        "rope_dim": cfg.rope_dim,
        "rope_theta": cfg.rope_theta,
        "context_length": cfg.context_length,
        "vocab_size": vocab_from_embed,
        "d_model": d_from_embed,
        "max_seq_len": max_seq_len,
        "eos_token_ids": cfg.eos_token_ids,
        "bos_token_id": cfg.bos_token_id,
        "tokenizer": "tokenizer.json",
        "tokenizer_config": "tokenizer_config.json",
        "embed_bin": "aion_embed.bin",
        "rope_cos_bin": rope_cos_bin,
        "rope_sin_bin": rope_sin_bin,
        "rope_cache_shape": list(rope_cache_shape),
        "coreml_package": mlmodelc.with_suffix(".mlpackage").name,
        "coreml_compiled": mlmodelc.name,
        "output_kind": output_kind,
    }
    (out_dir / "aion_runtime_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
        help="Path to PATH_TO_MODEL_BUNDLE or a parent directory containing a versioned model.onnx bundle.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models" / "aion" / "ane",
        help="Output directory for the CoreML package and metadata.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Compiled KV cache length (defaults to 2048 to keep the state bundle practical).",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip post-conversion INT8 weight quantization.",
    )
    parser.add_argument(
        "--argmax-output",
        action="store_true",
        help="Return only the sampled next-token id from CoreML instead of full vocab logits.",
    )
    args = parser.parse_args()

    bundle = _latest_model_bundle(args.source_bundle)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source bundle: {bundle}")
    print(f"Output dir:    {out_dir}")

    _copy_if_exists(bundle / "tokenizer.json", out_dir)
    _copy_if_exists(bundle / "tokenizer_config.json", out_dir)
    _copy_if_exists(bundle / "genai_config.json", out_dir)
    _copy_if_exists(bundle / "manifest.json", out_dir)
    _copy_if_exists(bundle / "edge_on_device_model_execution_config.pb", out_dir)

    output_kind = "argmax" if args.argmax_output else "logits"
    mlmodelc, embed_shape = _build_coreml_model(
        bundle,
        out_dir,
        args.max_seq_len,
        quantize_int8=not args.no_int8,
        output_kind=output_kind,
    )
    _write_runtime_metadata(bundle, out_dir, args.max_seq_len, embed_shape, mlmodelc, output_kind)
    print(f"Saved: {mlmodelc}")
    print(f"Tokenizer/config copied into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())