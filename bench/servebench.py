#!/usr/bin/env python3
"""Serving throughput under concurrency, one methodology for every engine.

All requests are submitted at t=0 so the engine runs saturated; the headline is
steady-state output throughput. TTFT therefore includes queueing for everything
past the first C requests, which is what a real backlog looks like.

    bench/servebench.py --engine qwc --concurrency 32
    .venvs/baseline/bin/python bench/servebench.py --engine vllm --concurrency 32
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


def prompt_ids(count: int, seed: int) -> list[int]:
    """Distinct prompt per request: identical ones let prefix caching skip
    prefill, which reports a prefill rate the hardware cannot reach."""
    return [1000 + ((seed * 7919 + index) % 20000) for index in range(count)]


def run_qwc(args) -> dict:
    command = [
        "cargo", "run", "--release", "--quiet", "-p", "qwc-engine",
        "--bin", "servebench", "--",
        "--model", str(args.model),
        "--requests", str(args.requests),
        "--concurrency", str(args.concurrency),
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--context", str(args.context),
        "--kv-cache-gb", str(args.kv_cache_gb),
        "--memory-limit-gb", str(args.memory_limit_gb),
        "--kv-cache", args.kv_cache_dtype,
        "--delta-state", args.delta_state,
    ]
    # Ширина шага prefill — такой же свипаемый параметр, как
    # max_num_batched_tokens у vLLM: держать его фиксированным значит
    # занижать одну из сторон.
    if args.prefill_chunk:
        command.extend(("--prefill-chunk", str(args.prefill_chunk)))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"qwc servebench exited with {result.returncode}")
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
        # v1 only fills RequestStateStats when stats are enabled, and without
        # them the record has no TTFT/ITL to compare against qwc's.
        disable_log_stats=False,
        # Matched to qwc: the same number of concurrent sequences, and a token
        # budget per step that admits the same prefill chunking.
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_batched_tokens or max(args.context, args.concurrency),
        # "bf16" is vLLM's unquantized KV, which it spells "auto".
        kv_cache_dtype="fp8" if args.kv_cache_dtype == "fp8" else "auto",
    )
    load_seconds = time.perf_counter() - started

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_tokens=args.max_new,
        min_tokens=args.max_new,
        ignore_eos=True,
        detokenize=False,
    )
    requests = [
        {"prompt_token_ids": prompt_ids(args.prompt_tokens, seed)}
        for seed in range(1, args.requests + 1)
    ]

    # Warm up CUDA graphs and the caches on a smaller saturated run. Its seeds
    # are disjoint from the measured ones so it cannot prime the prefix cache.
    warmup = [
        {"prompt_token_ids": prompt_ids(args.prompt_tokens, 10**6 + seed)}
        for seed in range(min(args.concurrency, args.requests))
    ]
    llm.generate(warmup, params, use_tqdm=False)

    started = time.perf_counter()
    outputs = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.perf_counter() - started

    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    expected = args.requests * args.max_new
    if output_tokens != expected:
        raise SystemExit(f"vLLM produced {output_tokens} tokens, expected {expected}")

    # vLLM 0.28 reports RequestStateStats: `first_token_latency` is already
    # wall-clock since arrival, so it includes queueing exactly like qwc's TTFT.
    # The token timestamps are engine-core monotonic and only valid as a span.
    ttft = []
    itl = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics is None:
            continue
        if getattr(metrics, "first_token_latency", 0.0):
            ttft.append(metrics.first_token_latency * 1e3)
        produced = len(output.outputs[0].token_ids)
        span = getattr(metrics, "last_token_ts", 0.0) - getattr(metrics, "first_token_ts", 0.0)
        if produced > 1 and span > 0:
            itl.append(span * 1e3 / (produced - 1))
    import vllm

    record = {
        "engine": "vllm",
        "version": vllm.__version__,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new,
        "context": args.context,
        "kv_cache": args.kv_cache_dtype,
        "max_batched_tokens": args.max_batched_tokens or max(args.context, args.concurrency),
        "load_seconds": round(load_seconds, 3),
        "wall_seconds": round(elapsed, 4),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / elapsed, 2),
        "requests_per_second": round(args.requests / elapsed, 3),
    }
    if ttft:
        ttft.sort()
        record["ttft_ms_p50"] = round(statistics.median(ttft), 2)
        record["ttft_ms_p95"] = round(ttft[int((len(ttft) - 1) * 0.95)], 2)
    if itl:
        itl.sort()
        record["itl_ms_p50"] = round(statistics.median(itl), 3)
        record["itl_ms_p95"] = round(itl[int((len(itl) - 1) * 0.95)], 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("qwc", "vllm"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", choices=("fp8", "bf16"), default="fp8")
    # vLLM only: match qwc's 512-token prefill arena to isolate chunk width.
    parser.add_argument("--max-batched-tokens", type=int, default=0)
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument(
        "--delta-state", choices=("wy", "bf16", "fp32"), default="wy")
    parser.add_argument("--kv-cache-gb", type=float, default=5.0)
    parser.add_argument("--memory-limit-gb", type=float, default=28.0)
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
