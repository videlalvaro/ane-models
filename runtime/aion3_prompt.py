#!/usr/bin/env python3
"""Tokenize text with Aion's Edge tokenizer and run the Swift ANE runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from converters.aion3_onnx_to_ane import format_aion_chat_prompt


def run(args: argparse.Namespace) -> int:
    meta_path = args.meta
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizer_path = meta_path.parent / meta.get("tokenizer", "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    prompt = args.prompt
    if args.chat:
        prompt = format_aion_chat_prompt(prompt, system_prompt=args.system_prompt)

    encoded = tokenizer.encode(prompt)
    prompt_ids = encoded.ids
    if not prompt_ids:
        raise SystemExit("tokenizer produced no tokens")

    runtime = args.runtime
    cmd = [
        str(runtime),
        "--meta",
        str(meta_path),
        "--prompt-ids",
        ",".join(str(x) for x in prompt_ids),
        "--max-new",
        str(args.max_new),
        "--warmup",
        str(args.warmup),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if proc.stderr:
        print(proc.stderr, end="")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    generated_ids = payload.get("generated_ids") or []
    text = tokenizer.decode(generated_ids)
    print(json.dumps({"prompt_ids": prompt_ids, "generated_ids": generated_ids, "text": text, "timing": payload.get("timing")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--meta", type=Path, default=ROOT / "models" / "aion" / "ane" / "aion_runtime_meta.json")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime" / "aion3_ane_runtime")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--chat", action="store_true", help="Wrap the prompt with Aion's native chat template")
    parser.add_argument("--system-prompt", help="Optional system message used with --chat")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
