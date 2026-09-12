#!/usr/bin/env python3
"""Steady-state batch-1 decode latency, one methodology for every engine.

Per-token cost is the slope between generating a short and a long completion
from the same prompt. The slope cancels prefill, the first-token cost and every
fixed per-request overhead, so engines with very different request paths stay
comparable. Sampling is greedy and detokenization is off on both sides, because
qwc has no detokenizer in the measured loop.

    bench/latency.py --engine qwc
    .venvs/baseline/bin/python bench/latency.py --engine vllm
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4"
# "Hello, world is a test of the system" — a prompt both engines take as IDs,
# so no tokenizer difference enters the measurement.
PROMPT = [9707, 11, 1879, 374, 264, 1273, 315, 279, 1849]


def run_qwc(args) -> dict:
    command = [
        "cargo", "run", "--release", "--quiet", "-p", "qwc-engine",
        "--bin", "latency", "--",
        "--model", str(args.model),
        "--prompt-ids", ",".join(str(token) for token in PROMPT),
        "--short", str(args.short),
        "--long", str(args.long),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--batch", str(args.batch),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"qwc latency runner exited with {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_vllm(args) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        max_model_len=args.context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        disable_log_stats=True,
        # Matched to what qwc is configured for, so neither engine is asked to
        # reserve for a batch shape the benchmark never uses.
        max_num_seqs=max(args.batch, 8),
        max_num_batched_tokens=args.context,
    )
    load_seconds = time.perf_counter() - started

    def generate(tokens: int) -> float:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=tokens,
            min_tokens=tokens,
            ignore_eos=True,
            detokenize=False,
        )
        requests = [{"prompt_token_ids": list(PROMPT)} for _ in range(args.batch)]
        started = time.perf_counter()
        outputs = llm.generate(requests, params, use_tqdm=False)
        elapsed = (time.perf_counter() - started) * 1e3
        for output in outputs:
            produced = len(output.outputs[0].token_ids)
            if produced != tokens:
                raise SystemExit(f"vLLM produced {produced} tokens, expected {tokens}")
        return elapsed

    # Capture CUDA graphs and settle the caches before anything is recorded.
    for _ in range(args.warmup):
        generate(args.long)

    short = statistics.median(generate(args.short) for _ in range(args.repeats))
    long = statistics.median(generate(args.long) for _ in range(args.repeats))
    slope = (long - short) / (args.long - args.short)
    import vllm

    return {
        "engine": "vllm",
        "version": vllm.__version__,
        "batch": args.batch,
        "prompt_tokens": len(PROMPT),
        "short_tokens": args.short,
        "long_tokens": args.long,
        "repeats": args.repeats,
        "load_seconds": round(load_seconds, 3),
        "short_ms": round(short, 4),
        "long_ms": round(long, 4),
        "decode_ms_per_token": round(slope, 4),
        "tokens_per_second": round(args.batch * 1000.0 / slope, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("qwc", "vllm"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--short", type=int, default=8)
    parser.add_argument("--long", type=int, default=72)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = run_qwc(args) if args.engine == "qwc" else run_vllm(args)
    text = json.dumps(record, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
