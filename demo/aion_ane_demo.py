#!/usr/bin/env python3
"""Recording-friendly local Aion-on-ANE demo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = [
    "Hello, who are you?",
]

BOLD = "\033[1m"
DIM = "\033[90m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"


def default_meta() -> Path:
    short = ROOT / "models" / "aion" / "ane-256" / "aion_runtime_meta.json"
    if short.exists():
        return short
    return ROOT / "models" / "aion" / "ane" / "aion_runtime_meta.json"


def compile_runtime(runtime: Path, force: bool) -> None:
    source = ROOT / "runtime" / "aion3_ane.swift"
    if not force and runtime.exists() and runtime.stat().st_mtime >= source.stat().st_mtime:
        return
    print(f"{DIM}Compiling Swift CoreML runtime...{RESET}")
    subprocess.run(
        [
            "swiftc",
            "-O",
            str(source),
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-o",
            str(runtime),
        ],
        cwd=ROOT,
        check=True,
    )


def load_tokenizer(meta_path: Path) -> tuple[dict, Tokenizer]:
    if not meta_path.exists():
        raise SystemExit(f"missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizer_path = meta_path.parent / meta.get("tokenizer", "tokenizer.json")
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer: {tokenizer_path}")
    return meta, Tokenizer.from_file(str(tokenizer_path))


def run_prompt(runtime: Path, meta_path: Path, tokenizer: Tokenizer, prompt: str, max_new: int, warmup: int, stop_repeat_ngram: int) -> dict:
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise RuntimeError("tokenizer produced no prompt tokens")

    cmd = [
        str(runtime),
        "--meta",
        str(meta_path),
        "--prompt-ids",
        ",".join(str(token_id) for token_id in prompt_ids),
        "--max-new",
        str(max_new),
        "--warmup",
        str(warmup),
    ]
    if stop_repeat_ngram > 0:
        cmd.extend(["--stop-repeat-ngram", str(stop_repeat_ngram)])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    raw_json = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(raw_json)
    generated_ids = payload.get("generated_ids") or []
    payload["prompt_ids"] = prompt_ids
    payload["text"] = tokenizer.decode(generated_ids).strip()
    payload["runtime_stderr"] = proc.stderr.strip()
    payload["runtime_cmd"] = cmd
    payload["raw_json"] = raw_json
    return payload


def print_intro(meta: dict, meta_path: Path, max_new: int, stop_repeat_ngram: int) -> None:
    print(f"{BOLD}Aion-1.0-Instruct on Apple Neural Engine{RESET}")
    print(f"{DIM}model:   {meta.get('model_family', 'Aion')}{RESET}")
    print(f"{DIM}package: {meta_path.parent}{RESET}")
    print(f"{DIM}context: {meta.get('max_seq_len')} tokens | output: {meta.get('output_kind', 'logits')} | max_new: {max_new}{RESET}")
    if stop_repeat_ngram > 0:
        print(f"{DIM}decode:  greedy argmax with repeated {stop_repeat_ngram}-gram stop{RESET}")
    else:
        print(f"{DIM}decode:  greedy argmax, no repetition stop{RESET}")
    print()


def print_result(index: int, prompt: str, payload: dict, delay: float, verbose: bool, raw: bool) -> None:
    timing = payload.get("timing") or {}
    print(f"{CYAN}{BOLD}Prompt {index}{RESET}")
    print(f"{DIM}user>{RESET} {prompt}")
    if verbose:
        print(f"{DIM}runtime>{RESET} {' '.join(payload.get('runtime_cmd', []))}")
        print(f"{DIM}prompt_ids>{RESET} {payload.get('prompt_ids', [])}")
    if delay > 0:
        time.sleep(delay)
    print(f"{GREEN}aion>{RESET} {payload.get('text', '')}")
    if verbose:
        print(f"{DIM}generated_ids>{RESET} {payload.get('generated_ids', [])}")
    print(
        f"{DIM}tokens: prompt={len(payload.get('prompt_ids', []))}, "
        f"generated={len(payload.get('generated_ids', []))} | "
        f"prefill={timing.get('prefill_s', 0):.3f}s | "
        f"decode={timing.get('decode_tok_s', 0):.2f} tok/s{RESET}"
    )
    if raw:
        print(f"{DIM}raw_json>{RESET} {payload.get('raw_json', '')}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=default_meta(), help="Aion runtime metadata JSON")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime" / "aion3_ane_runtime")
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--stop-repeat-ngram", type=int, default=4, help="Stop when the generated suffix repeats an earlier n-gram; set 0 to disable")
    parser.add_argument("--delay", type=float, default=0.8, help="Pause between prompt and answer for video pacing")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to run; repeat for multiple prompts")
    parser.add_argument("--no-build", action="store_true", help="Do not compile the Swift runtime first")
    parser.add_argument("--force-build", action="store_true", help="Always rebuild the Swift runtime")
    parser.add_argument("--verbose", action="store_true", help="Show runtime command, prompt IDs, and generated IDs")
    parser.add_argument("--raw", action="store_true", help="Show raw JSON returned by the Swift runtime")
    args = parser.parse_args()

    meta_path = args.meta.resolve()
    runtime = args.runtime.resolve()
    prompts = args.prompt or DEFAULT_PROMPTS

    if not args.no_build:
        compile_runtime(runtime, force=args.force_build)
    if not runtime.exists():
        raise SystemExit(f"missing runtime: {runtime}")

    meta, tokenizer = load_tokenizer(meta_path)
    print_intro(meta, meta_path, args.max_new, args.stop_repeat_ngram)

    for index, prompt in enumerate(prompts, start=1):
        payload = run_prompt(
            runtime,
            meta_path,
            tokenizer,
            prompt,
            args.max_new,
            args.warmup if index == 1 else 0,
            args.stop_repeat_ngram,
        )
        print_result(index, prompt, payload, args.delay, args.verbose, args.raw)

    print(f"{BOLD}Done.{RESET} {DIM}Run again with --prompt 'your text' to customize the recording.{RESET}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
