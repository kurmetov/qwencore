#!/usr/bin/env python3
"""FlashInfer attention on QwenCore shapes: for comparison with packedbench.

Run only from the baseline venv and only with prebuilt kernels from the JIT cache:

    FLASHINFER_DISABLE_JIT=1 ~/.venvs/baseline/bin/python bench/flashinfer_attention.py

FLASHINFER_DISABLE_JIT is mandatory: without it a cache miss triggers a build, and
FlashInfer's JIT hangs the desktop. Lacking nvcc in PATH isn't enough — FlashInfer takes
$CUDA_HOME/bin/nvcc directly. The flag by itself also forbids loading from cache:
without building, FlashInfer trusts only AOT artifacts and sends a prebuilt .so from the JIT cache
through ninja. So the bench loads it itself.

Shapes are the same as `packedbench`: a segment of one sequence after
`start` of history, KV fp8 e4m3 in the HND layout ([page][KV head][token]
[dimension], as ours), page 64, 16 layers with separate caches. Paths:
- xqa, q_len 1 — decode of one row;
- xqa, q_len 4 — verifying three drafts (how vLLM runs it with MTP);
- batch prefill (FA2) — prompt chunks and tails.
Before measuring, every kernel is checked against a torch reference.
"""

import argparse
import os
import sys

if not os.environ.get("FLASHINFER_DISABLE_JIT"):
    sys.exit("FLASHINFER_DISABLE_JIT=1 is required: otherwise a cache miss triggers a JIT build")

import torch
import flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.core import JitSpecNvcc


def load_built(self, aot=JitSpecNvcc.try_load):
    """The AOT artifact, or if absent — the vLLM-built .so from the JIT cache."""
    module = aot(self)
    if module is None and self.jit_library_path.exists():
        module = self.load()
    return module


JitSpecNvcc.try_load = load_built

HEADS, KV_HEADS, DIM, PAGE, LAYERS = 24, 4, 256, 64, 16
SCALE = DIM ** -0.5
# This is how much vLLM gives the FlashInfer buffer by default; the first 8 MB are
# xqa semaphores. xqa and FA2 need separate buffers: FA2 partial sums land on top of
# the semaphores, while xqa expects them zeroed and then computes garbage.
WORKSPACE_BYTES = 394 * 1024 * 1024


def draft_mask(q_len: int, device) -> torch.Tensor:
    """Causal draft mask in xqa packing, as vLLM builds it."""
    packed = (q_len + 31) // 32
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(packed * 32, device=device).unsqueeze(0)
    bits = 1 << torch.arange(32, device=device, dtype=torch.int64)
    words = ((kv_idx <= q_idx).view(q_len, packed, 32).to(torch.int64) * bits).sum(-1)
    return words.to(torch.uint32).view(torch.uint16).reshape(1, q_len, packed * 2)


def reference(query, key, value, table, start, rows):
    """Causal attention in fp32 over table pages; row r sees start+r+1."""
    context = start + rows
    pages = table[: (context + PAGE - 1) // PAGE].long()
    k = key[pages].float().permute(1, 0, 2, 3).reshape(KV_HEADS, -1, DIM)[:, :context]
    v = value[pages].float().permute(1, 0, 2, 3).reshape(KV_HEADS, -1, DIM)[:, :context]
    k = k.repeat_interleave(HEADS // KV_HEADS, 0)
    v = v.repeat_interleave(HEADS // KV_HEADS, 0)
    q = query.float().transpose(0, 1)  # [head][row][dimension]
    scores = q @ k.transpose(1, 2) * SCALE
    limits = torch.arange(rows, device=q.device) + start + 1
    allowed = torch.arange(context, device=q.device).unsqueeze(0) < limits.unsqueeze(1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1)


def timed(run, iterations=5):
    run()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1e3 / (iterations * LAYERS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", default="8192,30000,60000")
    parser.add_argument("--prefill-rows", default="4,64,128,256,512")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"FlashInfer {flashinfer.__version__}, {torch.cuda.get_device_name()}, "
          f"{LAYERS} layers, KV fp8 e4m3 HND; us per layer")
    print(f"  {'path':>18} | {'rows':>5} | {'start':>6} | {'us':>8} | {'rel.err':>8}")

    xqa_workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    prefill_workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    for start in (int(s) for s in args.starts.split(",")):
        max_rows = 512
        pages = (start + max_rows + PAGE - 1) // PAGE
        caches = []
        for _ in range(LAYERS):
            k = (torch.randn(pages, KV_HEADS, PAGE, DIM, device=device) * 0.5).to(torch.float8_e4m3fn)
            v = (torch.randn(pages, KV_HEADS, PAGE, DIM, device=device) * 0.5).to(torch.float8_e4m3fn)
            caches.append((k, v))
        # Pages shuffled: a test that ignores the table won't pass.
        table = torch.randperm(pages, device=device, dtype=torch.int32)

        def check(name, rows, output, query):
            key, value = caches[0]
            expected = reference(query, key, value, table, start, rows)
            # The output is an average over thousands of keys, its scale ~0.01: error
            # relative to the reference's largest element.
            error = ((output.float() - expected).abs().max() / expected.abs().max()).item()
            if error > 0.02:
                raise SystemExit(f"{name}: {rows} rows at {start} diverge from reference: {error}")
            return error

        # xqa: decode and draft verification.
        for q_len in (1, 4):
            query = (torch.randn(q_len, HEADS, DIM, device=device) * 0.5).to(torch.bfloat16)
            seq_lens = torch.tensor([start + q_len], dtype=torch.int32, device=device)
            block_tables = table[: (start + q_len + PAGE - 1) // PAGE].unsqueeze(0).contiguous()
            mask = draft_mask(q_len, device) if q_len > 1 else None
            output = torch.empty_like(query)

            def xqa_layer(layer):
                trtllm_batch_decode_with_kv_cache(
                    query, caches[layer], xqa_workspace, block_tables, seq_lens,
                    max_seq_len=block_tables.shape[1] * PAGE,
                    bmm1_scale=SCALE, bmm2_scale=1.0, out=output, kv_layout="HND",
                    backend="xqa", q_len_per_req=q_len, mask=mask,
                )

            xqa_layer(0)
            error = check("xqa", q_len, output, query)
            micros = timed(lambda: [xqa_layer(layer) for layer in range(LAYERS)])
            name = "xqa decode" if q_len == 1 else f"xqa spec q_len={q_len}"
            print(f"  {name:>18} | {q_len:>5} | {start:>6} | {micros:>8.1f} | {error:>8.4f}")

        # FA2 batch prefill: a segment of `rows` rows at position start.
        for rows in (int(r) for r in args.prefill_rows.split(",")):
            context = start + rows
            used = (context + PAGE - 1) // PAGE
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(prefill_workspace, "HND")
            wrapper.plan(
                torch.tensor([0, rows], dtype=torch.int32, device=device),
                torch.tensor([0, used], dtype=torch.int32, device=device),
                table[:used].contiguous(),
                torch.tensor([context - (used - 1) * PAGE], dtype=torch.int32, device=device),
                HEADS, KV_HEADS, DIM, PAGE,
                causal=True, sm_scale=SCALE,
                q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
            )
            query = (torch.randn(rows, HEADS, DIM, device=device) * 0.5).to(torch.bfloat16)
            output = torch.empty_like(query)
            wrapper.run(query, caches[0], out=output)
            error = check("prefill", rows, output, query)
            micros = timed(lambda: [wrapper.run(query, caches[layer], out=output)
                                    for layer in range(LAYERS)])
            print(f"  {'batch prefill FA2':>18} | {rows:>5} | {start:>6} | {micros:>8.1f} | {error:>8.4f}")
        del caches
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
