#!/usr/bin/env python3
"""Aion converter Torch vs CoreML parity check (single token, decode step 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converters.aion3_onnx_to_ane import OnnxWeightStore, StatefulDecoderModel, _latest_model_bundle, infer_config


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def to_multiarray(name: str, array: np.ndarray):
    return ct.models.datatypes.Array(*array.shape), {name: array}


def run(args: argparse.Namespace) -> int:
    bundle = _latest_model_bundle(args.bundle)
    onnx_path = bundle / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    coreml_model_path = args.coreml_model
    if not coreml_model_path.exists():
        raise FileNotFoundError(coreml_model_path)

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

    torch_model = StatefulDecoderModel(weights, cfg, max_seq_len=max_seq_len).module()
    torch_model.half().eval()

    x = emb[token_id].astype(np.float16).reshape(1, cfg.hidden_size, 1, 1)
    cos_np = weights.get_any([
        "model.layers.0.self_attn.cos_cached_export",
        "layers.0.self_attn.cos_cached_export",
    ])[:1].astype(np.float16)
    sin_np = weights.get_any([
        "model.layers.0.self_attn.sin_cached_export",
        "layers.0.self_attn.sin_cached_export",
    ])[:1].astype(np.float16)

    attn_mask = np.full((1, 1, 1, max_seq_len), np.float16(-1e4), dtype=np.float16)
    attn_mask[0, 0, 0, 0] = np.float16(0)
    kv_write_mask = np.zeros((1, 1, max_seq_len, 1), dtype=np.float16)
    kv_write_mask[0, 0, 0, 0] = np.float16(1)

    with torch.no_grad():
        logits_t = torch_model(
            torch.from_numpy(x),
            torch.from_numpy(cos_np),
            torch.from_numpy(sin_np),
            torch.from_numpy(attn_mask),
            torch.from_numpy(kv_write_mask),
        ).float().numpy()[0]

    mlmodel = ct.models.MLModel(str(coreml_model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = mlmodel.make_state()
    out = mlmodel.predict(
        {
            "x": x,
            "rope_cos": cos_np,
            "rope_sin": sin_np,
            "attn_mask": attn_mask,
            "kv_write_mask": kv_write_mask,
        },
        state=state,
    )

    logits_c = np.asarray(out["logits"], dtype=np.float32).reshape(-1)
    logits_t32 = logits_t.astype(np.float32)

    diff = logits_t32 - logits_c
    print(f"token_id={token_id}")
    print(f"torch_argmax={int(np.argmax(logits_t32))}")
    print(f"coreml_argmax={int(np.argmax(logits_c))}")
    print(f"top5_torch={np.argsort(-logits_t32)[:5].tolist()}")
    print(f"top5_coreml={np.argsort(-logits_c)[:5].tolist()}")
    print(f"cosine={cosine(logits_t32, logits_c):.6f}")
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
    parser.add_argument(
        "--coreml-model",
        type=Path,
        default=ROOT / "models" / "aion" / "ane" / "Aion3_Stateful.mlpackage",
    )
    parser.add_argument("--token-id", type=int, default=151643)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
