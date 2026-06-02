#!/usr/bin/env python3
"""Convert a stateless token-0 Aion probe to CoreML and compare against Torch.

This isolates CoreML math lowering from CoreML state handling by testing the
single-token, no-past-cache path only.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converters.aion3_onnx_to_ane import OnnxWeightStore, Qwen3StatefulModel, _latest_model_bundle, infer_config


class StatelessToken0Model(torch.nn.Module):
    def __init__(self, stateful: Qwen3StatefulModel):
        super().__init__()
        self.layers = stateful.layers
        self.output_norm = stateful.output_norm
        self.lm_head = stateful.lm_head
        self.cfg = stateful.cfg

    def forward(self, x, rope_cos_pos, rope_sin_pos):
        for layer in self.layers:
            residual = x
            normed = layer.attn_norm(x)
            qkv = layer.qkv_conv(normed).squeeze(-1).squeeze(-1)
            q = qkv[:, :layer.q_dim]
            k = qkv[:, layer.q_dim:layer.q_dim + layer.kv_dim]
            v = qkv[:, layer.q_dim + layer.kv_dim:]

            q = layer.q_head_norm(q.reshape(1, layer.nh, layer.dh)).reshape(1, layer.q_dim)
            k = layer.k_head_norm(k.reshape(1, layer.nkv, layer.dh)).reshape(1, layer.kv_dim)

            q = layer._apply_rope(q, rope_cos_pos, rope_sin_pos, layer.nh)
            k = layer._apply_rope(k, rope_cos_pos, rope_sin_pos, layer.nkv)

            q_heads = q.reshape(1, layer.nh, layer.dh)
            k_heads = k.reshape(1, layer.nkv, layer.dh)
            v_heads = v.reshape(1, layer.nkv, layer.dh)

            attn_parts = []
            for kv_idx in range(layer.nkv):
                q_group = q_heads[:, kv_idx * layer.hpk:(kv_idx + 1) * layer.hpk, :]
                k_head = k_heads[:, kv_idx:kv_idx + 1, :]
                v_head = v_heads[:, kv_idx:kv_idx + 1, :]
                scores = torch.matmul(q_group, k_head.transpose(-2, -1)) * layer.scale
                attn_w = torch.softmax(scores.float(), dim=-1).to(q_group.dtype)
                head_out = torch.matmul(attn_w, v_head)
                attn_parts.append(head_out)

            attn_out = torch.cat(attn_parts, dim=1).reshape(1, layer.q_dim, 1, 1)
            attn_out = layer.out_conv(attn_out)
            x = residual + attn_out

            residual2 = x
            normed2 = layer.ffn_norm(x)
            gate_up = layer.gate_up_conv(normed2)
            gate = gate_up[:, :layer.dff, :, :]
            up = gate_up[:, layer.dff:, :, :]
            hidden = torch.nn.functional.silu(gate.float()).to(gate.dtype) * up
            x = residual2 + layer.down_conv(hidden)

        x = self.output_norm(x)
        return self.lm_head(x).squeeze(-1).squeeze(-1)


def fill_rope(pos: int, rope_half: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    cos = np.empty((1, rope_half), dtype=np.float16)
    sin = np.empty((1, rope_half), dtype=np.float16)
    for j in range(rope_half):
        inv = 1.0 / (theta ** (float(j) / float(rope_half)))
        angle = float(pos) * inv
        cos[0, j] = np.float16(math.cos(angle))
        sin[0, j] = np.float16(math.sin(angle))
    return cos, sin


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def run(args: argparse.Namespace) -> int:
    bundle = _latest_model_bundle(args.bundle)
    cfg = infer_config(bundle)
    weights = OnnxWeightStore(bundle / "model.onnx")
    emb = weights.get_any([
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "token_embd.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    ])
    stateful = Qwen3StatefulModel(weights, cfg, max_seq_len=2)
    model = StatelessToken0Model(stateful).half().eval()

    token_id = int(args.token_id)
    x_np = emb[token_id].astype(np.float16).reshape(1, cfg.hidden_size, 1, 1)
    cos_np, sin_np = fill_rope(0, cfg.rope_dim // 2, cfg.rope_theta)

    x = torch.from_numpy(x_np)
    rope_cos = torch.from_numpy(cos_np)
    rope_sin = torch.from_numpy(sin_np)
    with torch.no_grad():
        logits_t = model(x, rope_cos, rope_sin).float().numpy()[0]

    with tempfile.TemporaryDirectory(prefix="aion_stateless_probe_") as tmpdir:
        tmp_path = Path(tmpdir) / "AionStatelessProbe.mlpackage"
        traced = torch.jit.trace(model, (x, rope_cos, rope_sin))
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x", shape=x_np.shape, dtype=np.float16),
                ct.TensorType(name="rope_cos", shape=cos_np.shape, dtype=np.float16),
                ct.TensorType(name="rope_sin", shape=sin_np.shape, dtype=np.float16),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(str(tmp_path))
        loaded = ct.models.MLModel(str(tmp_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        out = loaded.predict({"x": x_np, "rope_cos": cos_np, "rope_sin": sin_np})
        logits_c = np.asarray(out["logits"], dtype=np.float32).reshape(-1)

    logits_t = logits_t.astype(np.float32)
    diff = logits_t - logits_c
    print(f"token_id={token_id}")
    print(f"torch_argmax={int(np.argmax(logits_t))}")
    print(f"coreml_argmax={int(np.argmax(logits_c))}")
    print(f"top5_torch={np.argsort(-logits_t)[:5].tolist()}")
    print(f"top5_coreml={np.argsort(-logits_c)[:5].tolist()}")
    print(f"cosine={cosine(logits_t, logits_c):.6f}")
    print(f"mae={float(np.mean(np.abs(diff))):.6f}")
    print(f"max_abs={float(np.max(np.abs(diff))):.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to PATH_TO_MODEL_BUNDLE or a parent directory containing a versioned model.onnx bundle.",
    )
    parser.add_argument("--token-id", type=int, default=1)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
