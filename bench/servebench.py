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
import math
import os
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


def corpus_prompts(path: Path, count: int, prompt_tokens: int) -> list[list[int]]:
    """Те же промпты, что читает `--corpus` у qwc: один и тот же файл, одна и
    та же нарезка. Синтетический `prompt_ids` — прогрессия, и спекуляция на
    ней угадывает почти всё; корпус нужен, чтобы acceptance был настоящим."""
    prompts = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            ids = json.loads(line)["prompt_token_ids"]
            if len(ids) < prompt_tokens:
                raise SystemExit(f"{path}: промпт короче {prompt_tokens} токенов")
            prompts.append(ids[:prompt_tokens])
            if len(prompts) == count:
                break
    if len(prompts) < count:
        raise SystemExit(f"{path}: {len(prompts)} промптов, а запрошено {count}")
    return prompts


def percentile(values: list[float], fraction: float) -> float:
    """Как `percentile` в servebench.rs: элемент с индексом round((n−1)·q)."""
    ordered = sorted(values)
    return ordered[int(math.floor((len(ordered) - 1) * fraction + 0.5))]


def latency_record(
    started: float,
    emissions: dict[str, list[tuple[float, int]]],
    ttft_service: list[float],
) -> dict:
    """TTFT, TPOT и ITL по моментам выдачи токенов — те же определения, что у
    qwc в servebench.rs. TTFT — от подачи всех запросов (t=0), с очередью;
    `ttft_service` — от взятия запроса в работу, без очереди. TPOT — по
    запросу (последний − первый)/(n − 1), перцентили по запросам. ITL — по
    промежуткам между выдачами; выдача в k токенов (спекуляция) даёт
    промежуток/k."""
    ttft, tpot, itl = [], [], []
    for times in emissions.values():
        ttft.append((times[0][0] - started) * 1e3)
        count = sum(tokens for _, tokens in times)
        if count > 1:
            tpot.append((times[-1][0] - times[0][0]) * 1e3 / (count - 1))
        for (previous, _), (now, tokens) in zip(times, times[1:]):
            itl.append((now - previous) * 1e3 / tokens)
    record = {
        "ttft_ms_p50": round(percentile(ttft, 0.50), 2),
        "ttft_ms_p95": round(percentile(ttft, 0.95), 2),
    }
    if itl:
        record["itl_ms_p50"] = round(percentile(itl, 0.50), 3)
        record["itl_ms_p95"] = round(percentile(itl, 0.95), 3)
    if tpot:
        record["tpot_ms_p50"] = round(percentile(tpot, 0.50), 3)
        record["tpot_ms_p95"] = round(percentile(tpot, 0.95), 3)
    if ttft_service:
        record["ttft_service_ms_p50"] = round(percentile(ttft_service, 0.50), 2)
        record["ttft_service_ms_p95"] = round(percentile(ttft_service, 0.95), 2)
    return record


def total_gpu_memory_gb() -> float:
    """Ёмкость карты в тех же десятичных ГБ, в которых движок задаёт лимит."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.split()[0]) * 1024 * 1024 / 1e9


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
        # При общем бюджете KV берёт остаток, как у vLLM: 0 означает «авто».
        "--kv-cache-gb", str(0 if args.engine_budget_gb else args.kv_cache_gb),
        "--memory-limit-gb", str(args.engine_budget_gb or args.memory_limit_gb),
        "--kv-cache", args.kv_cache_dtype,
        "--delta-state", args.delta_state,
        "--speculative", str(args.qwc_speculative),
        "--shortlist", str(args.qwc_shortlist),
        "--mtp-prime", args.qwc_mtp_prime,
    ]
    # Ширина шага prefill — такой же свипаемый параметр, как
    # max_num_batched_tokens у vLLM: держать его фиксированным значит
    # занижать одну из сторон.
    if args.prefill_chunk:
        command.extend(("--prefill-chunk", str(args.prefill_chunk)))
    if args.corpus:
        command.extend(("--corpus", str(args.corpus)))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"qwc servebench exited with {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run_vllm(args) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    # Один бюджет на оба движка: у нас он абсолютный, у vLLM — доля карты.
    # Считаем долю здесь и кладём в артефакт, чтобы её можно было проверить.
    total_gb = total_gpu_memory_gb()
    utilization = (
        args.engine_budget_gb / total_gb
        if args.engine_budget_gb
        else args.gpu_memory_utilization
    )

    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        max_model_len=args.context,
        gpu_memory_utilization=utilization,
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
        # Промпты и так уникальны, но фиксируем политику явно: результат не
        # должен зависеть от значения default в конкретной версии vLLM.
        enable_prefix_caching=False,
        # Чекпоинт мультимодальный, а стенд текстовый. Без этого флага vLLM
        # поднимает vision tower (0.92 ГБ), и под общим бюджетом мы
        # сравнивались с соперником, который носит лишний гигабайт.
        language_model_only=not args.vllm_keep_vision,
        # Голова MTP лежит в том же чекпоинте, и vLLM умеет её грузить как
        # qwen3_5_mtp. Мерить его без неё — мерить недонастроенного соперника.
        speculative_config=(
            {
                "method": "qwen3_5_mtp",
                "model": str(args.model),
                "num_speculative_tokens": args.vllm_speculative,
            }
            if args.vllm_speculative
            else None
        ),
    )
    load_seconds = time.perf_counter() - started
    # Формат состояния DeltaNet не задаём: "auto" для этой модели — fp32 из
    # `mamba_ssm_dtype` в config.json. В артефакт идёт итог, а не запрос.
    engine_config = getattr(llm.llm_engine, "vllm_config", None)
    cache_config = getattr(engine_config, "cache_config", None)
    mamba_ssm_dtype = str(getattr(cache_config, "mamba_ssm_cache_dtype", "auto"))

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
    if args.corpus:
        requests = [
            {"prompt_token_ids": ids}
            for ids in corpus_prompts(args.corpus, args.requests, args.prompt_tokens)
        ]
    else:
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

    # Тот же цикл, что внутри `LLM.generate`, но с выдачей по шагам (DELTA):
    # generate просит FINAL_ONLY, и у запроса остаются только первый и
    # последний моменты — ITL по промежуткам из них не собрать.
    from vllm.sampling_params import RequestOutputKind

    stepped = params.clone()
    stepped.output_kind = RequestOutputKind.DELTA
    engine = llm.llm_engine
    emissions: dict[str, list[tuple[float, int]]] = {}
    ttft_service: list[float] = []
    started = time.perf_counter()
    for index, request in enumerate(requests):
        engine.add_request(f"bench-{index}", request, stepped)
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        now = time.perf_counter()
        for output in step_outputs:
            tokens = len(output.outputs[0].token_ids)
            if tokens:
                emissions.setdefault(output.request_id, []).append((now, tokens))
            # Моменты ядра движка монотонные и годятся только как промежуток:
            # взят в работу -> первый токен.
            metrics = getattr(output, "metrics", None)
            if output.finished and metrics is not None and metrics.scheduled_ts:
                ttft_service.append((metrics.first_token_ts - metrics.scheduled_ts) * 1e3)
    elapsed = time.perf_counter() - started

    output_tokens = sum(tokens for times in emissions.values() for _, tokens in times)
    expected = args.requests * args.max_new
    if output_tokens != expected or len(emissions) != args.requests:
        raise SystemExit(
            f"vLLM produced {output_tokens} tokens in {len(emissions)} requests, "
            f"expected {expected} in {args.requests}"
        )
    import vllm

    record = {
        "engine": "vllm",
        "version": vllm.__version__,
        "speculative_tokens": args.vllm_speculative,
        "engine_budget_gb": round(args.engine_budget_gb, 2) if args.engine_budget_gb else None,
        "gpu_memory_utilization": round(utilization, 4),
        "gpu_total_gb": round(total_gb, 2),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new,
        "context": args.context,
        "kv_cache": args.kv_cache_dtype,
        "cuda_graphs": True,
        "prefix_caching": False,
        "language_model_only": not args.vllm_keep_vision,
        "mamba_ssm_cache_dtype": mamba_ssm_dtype,
        "prompts": str(args.corpus) if args.corpus else "synthetic",
        "max_batched_tokens": args.max_batched_tokens or max(args.context, args.concurrency),
        "load_seconds": round(load_seconds, 3),
        "wall_seconds": round(elapsed, 4),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / elapsed, 2),
        "requests_per_second": round(args.requests / elapsed, 3),
        **latency_record(started, emissions, ttft_service),
    }
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
    parser.add_argument(
        "--engine-budget-gb",
        type=float,
        default=0.0,
        help="общий потолок видеопамяти на движок; у vLLM пересчитывается в долю карты",
    )
    parser.add_argument(
        "--vllm-speculative",
        type=int,
        default=0,
        help="глубина MTP-спекуляции у vLLM (0 — выключена)",
    )
    parser.add_argument(
        "--qwc-speculative",
        type=int,
        default=0,
        help="глубина MTP-спекуляции QwenCore (0 — выключена)",
    )
    parser.add_argument(
        "--qwc-shortlist",
        type=int,
        default=0,
        help="частотная часть shortlist словаря QwenCore (0 — полный lm_head)",
    )
    parser.add_argument(
        "--qwc-mtp-prime",
        default="2048",
        help="окно прогрева MTP-головы QwenCore: N, all или off; 2048 — как у serve",
    )
    parser.add_argument(
        "--vllm-keep-vision",
        action="store_true",
        help="не передавать language_model_only: vLLM поднимет vision tower "
             "(так шли замеры до 24.09)",
    )
    parser.add_argument("--max-batched-tokens", type=int, default=0)
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument(
        "--delta-state", choices=("wy", "bf16", "fp32"), default="wy")
    parser.add_argument("--kv-cache-gb", type=float, default=5.0)
    parser.add_argument("--memory-limit-gb", type=float, default=28.0)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="jsonl с prompt_token_ids (bench/make_corpus.py); "
             "без него промпты синтетические",
    )
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
