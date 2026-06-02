#!/usr/bin/env python3
"""Aion ONNX vs converter Torch parity check (single token, decode step 0)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converters.aion3_onnx_to_ane import OnnxWeightStore, StatefulDecoderModel, _latest_model_bundle, infer_config


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
    onnx_path = bundle / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    cfg = infer_config(bundle)
    max_seq_len = int(args.max_seq_len)
    token_id = int(args.token_id)

    weights = OnnxWeightStore(onnx_path)
    emb = weights.get_any([
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "token_embd.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    ])

    model = StatefulDecoderModel(weights, cfg, max_seq_len=max_seq_len).module()
    model.half().eval()

    x = torch.tensor(emb[token_id], dtype=torch.float16).reshape(1, cfg.hidden_size, 1, 1)
    rope_half = cfg.rope_dim // 2
    cos_np, sin_np = fill_rope(pos=0, rope_half=rope_half, theta=cfg.rope_theta)
    cos_t = torch.from_numpy(cos_np)
    sin_t = torch.from_numpy(sin_np)

    attn_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.float16)
    attn_mask[0, 0, 0, 0] = 0
    kv_write_mask = torch.zeros((1, 1, max_seq_len, 1), dtype=torch.float16)
    kv_write_mask[0, 0, 0, 0] = 1

    with torch.no_grad():
        logits_t = model(x, cos_t, sin_t, attn_mask, kv_write_mask).float().numpy()[0]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs: dict[str, np.ndarray] = {
        "input_ids": np.array([[token_id]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }
    for i in range(cfg.num_hidden_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros((1, cfg.num_key_value_heads, 0, cfg.head_size), dtype=np.float16)
        inputs[f"past_key_values.{i}.value"] = np.zeros((1, cfg.num_key_value_heads, 0, cfg.head_size), dtype=np.float16)

    logits_o = sess.run(["logits"], inputs)[0][0, 0, :].astype(np.float32)

    logits_t32 = logits_t.astype(np.float32)
    diff = logits_t32 - logits_o

    topk_t = np.argsort(-logits_t32)[:5].tolist()
    topk_o = np.argsort(-logits_o)[:5].tolist()

    print(f"token_id={token_id}")
    print(f"torch_argmax={int(np.argmax(logits_t32))}")
    print(f"onnx_argmax={int(np.argmax(logits_o))}")
    print(f"top5_torch={topk_t}")
    print(f"top5_onnx={topk_o}")
    print(f"cosine={cosine(logits_t32, logits_o):.6f}")
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
    parser.add_argument("--token-id", type=int, default=151643)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
