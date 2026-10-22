# qwencore vs vLLM 0.28.0 — 2026-09-12

> **Single points from one session, without confidence intervals.** The comparison with
> vLLM here was taken before the protocol
> [`docs/11-experiments.md`](../../docs/11-experiments.md) existed: one or two runs per
> point, no variant order rotation and no bootstrap CI, on a partially busy
> card. These multipliers are superseded by a comparison suite run and must not
> be quoted as a project result. The conditions of each point are given below.

RTX 5090 (32.6 GB), Qwen3.8-27B-QUASAR-NVFP4, `--context 32768`, greedy, no detokenization.

Conditions the same for both engines: **bf16 KV**, **a unique prompt for every request**
(identical prompts fed vLLM's prefix cache and made the measurement meaningless), the same
`bench/servebench.py`, all requests submitted at t=0.

The memory budget favors vLLM: it got `--gpu-memory-utilization 0.93` (~30.3 GB), qwencore lived
under `--memory-limit-gb 28`. vLLM additionally warms up with a saturated run, qwencore doesn't.

## Summary

| Point (prompt/output, concurrency) | qwencore | vLLM | |
|---|---:|---:|---|
| 256/128, c=80 | **1498** tok/s | 1340 tok/s | qwc +12% |
| 256/512, c=16 | 668 tok/s | **748** tok/s | vLLM +12% |
| 4096/512, c=16 | 265 tok/s | **439** tok/s | vLLM 1.66× |
| 4096/8, c=16 — pure prefill | **3763** tok/s prefill | **14656** tok/s prefill | vLLM 3.9× |

Repeats (spread): qwencore 256/128 — 1496.35 / 1504.34 / 1498.28; vLLM same point with chunk 512 —
1336.76 / 1340.41 / 1368.11.

## Sweep of vLLM's `max_num_batched_tokens`, point 256/128 c=80

| chunk | 512 | 1024 | 2048 | 4096 | 8192 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 1337 | 1190 | 1314 | 1161 | 1152 | 896 |

The old harness hard-set chunk = context (32768) and thereby understated vLLM by a third.
qwencore's step width is pinned to `PREFILL_CHUNK_SIZE = 512` (the activation arena is compiled
for 512 rows), but that's not an obstacle: vLLM squeezed into the same 512 loses nothing — on short
prompts that's its optimum. The prefill gap is in the kernels themselves, not in step width.

## What doesn't reproduce

- vLLM with `kv_cache_dtype=fp8` doesn't start in this venv: it goes to the FlashInfer backend,
  and flashinfer isn't installed. So KV type parity was done with bf16 on both sides.
- At util 0.93 and chunk 32768 on unique 4096-token prompts vLLM crashes with CUDA OOM:
  with the prefix cache it never ran a real large prefill, and the memory profiler
  didn't account for it. The working chunk for long prompts is 4096.
- The harness doesn't record vLLM latencies: in V1 offline `LLM.generate`, `RequestOutput.metrics`
  has no data. There's nothing to compare TTFT/ITL with, only wall-clock.
