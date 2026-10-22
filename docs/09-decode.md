# Token loop: prefill, CUDA graphs and continuous batching

```bash
IDS=$(scripts/tok.py encode "The capital of France is")
cargo run --release -p qwc-engine --bin generate -- --prompt-ids "$IDS" --max-new 24
scripts/tok.py decode "<returned ids>"
```

## The full path

`Executor` keeps decode buffer addresses constant and captures a separate
CUDA graph for every batch size it encounters. One step goes through all 64
layers:

* the FP8 embedding outputs BF16 directly;
* 48 Gated DeltaNet layers update the conv history and the recurrent state slot;
* 16 full-attention layers write FP8 paged KV and compute gated GQA;
* the MLP runs SwiGLU and the down projection;
* residual-add is fused with the next block's RMSNorm;
* the FP8 `lm_head` produces logits, a two-stage GPU argmax returns only
  a four-byte token ID.

Batch 1..4 uses W4A16 and fused gate/up. Batch 5..32 automatically
switches to the W4A4 tensor-core GEMM; there gate and up are computed by two GEMMs with
a small BF16 SwiGLU kernel between them.

## Causal prefill

The prompt is split into chunks of fixed `M=128`. All NVFP4 projections and the MLP
compute 128 rows in one W4A4 GEMM. In an incomplete last chunk the extra rows
are padding and don't take part in causal operations.

The recurrent axis can't be treated as a batch. So prefill has two
specialized scans:

* one CUDA thread carries the three-element conv history of one channel through the whole
  chunk;
* one warp holds a DeltaNet state row in registers across all tokens of the chunk.
  After every step the state is rounded to BF16, so the result matches
  sequential decode, but the layer's 1.5 MB of state is read and written once
  per chunk, not 128 times.

Full attention first writes the K/V of all chunk tokens in parallel, then
uses the shared physical page table and a separate `context_len` for each
query row. This gives an ordinary causal mask without a separate dense attention
buffer. Logits are computed only for the last live token.

On the RTX 5090 a 601-token prompt ran in **171.9 ms**, or **0.286
ms/token**. The old token-by-token path took about 8 s. After prefill the model
correctly continued the repeated pangram (`…The` → ` quick`) and then started
analyzing it.

## Decode and sampling, batch 1

| | |
|---|---:|
| decode step with CUDA graph, median | **14.17 ms (70.5 tok/s)** |
| GPU argmax | **0.03 ms** |
| full cycle | **14.20 ms (70.4 tok/s)** |
| previous host argmax | 0.38 ms and 1 MB PCIe readback |

The first step of a given batch size captures the graph and so is not part of the
steady-state median. Positions, lengths, state slots and page tables live in
resident device buffers: before a repeated launch their contents change, but
not their addresses.

Contexts of 30 and 601 tokens still have the same ITL: at batch 1 the path
is bound by 15 GB of weights and state per step, not by tens of megabytes of KV.

## W4A4 decode, batch 8

Eight identical prompts after independent prefill gave identical tokens:

```text
 Paris.
The capital of Germany is Berlin.
```

Decode median — **24.90 ms**, i.e. **321.3 tok/s goodput**; together with the GPU
argmax — **319.3 tok/s**. This is an end-to-end check of the dispatcher, not an isolated
projection benchmark.

## Wiring to the runtime

`Executor::execute_layout` takes a `qwc_runtime::BatchLayout`. The decode part
of the batch gets real `state_slots`, positions, context lengths and physical
KV page IDs. Prefill chunks use the same arbitrary slots/pages, after
which the GPU argmax returns `(SeqId, token)` for each batch element.

An example of the full admission → prefill → decode → completion loop:

```bash
cargo run --release -p qwc-engine --bin scheduled -- --requests 8 --max-new 8
```

The check run finished eight requests, all results matched. For a new
prompt with offset 0 the reused DeltaNet/conv slot is zeroed on the GPU;
KV pages need no clearing because visibility is set by `context_len`.

## Verification

```bash
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

68 tests currently pass, including a GPU oracle for the chunked DeltaNet scan and
deterministic argmax. End-to-end runs of the 27B checkpoint were done separately
for batch 1, batch 8 and scheduler-driven page tables. Step-by-step comparison of
token IDs and top-k log-probabilities against external runtimes is described in
[`10-differential-eval.md`](10-differential-eval.md).

## Decode path limitations

* several prefill sequences of a mixed batch run
  sequentially, although each of their chunks uses tensor cores;
* W4A4 gate/up re-read the normalized activation: they have no shared fused mainloop;
* no preemption, prefix snapshots or corpus-level perplexity eval;
* `scripts/tok.py` uses an ASCII approximation of the pre-tokenizer; Unicode
  needs the full tokenizer.
