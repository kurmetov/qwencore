# Runtime: two-resource cache and continuous batching

## Why a separate runtime

In Qwen3.8 one sequence occupies two different resources at once:

* a fixed state slot for the 48 Gated DeltaNet layers;
* a variable number of paged KV blocks for the 16 full-attention layers.

So a single KV counter is not enough. A request is admitted only when
there is both a state slot and all the prompt's KV blocks. `CacheManager` performs this check
transactionally: on refusal no resource is consumed. The scheduler additionally
commits KV capacity for `prompt + max_new_tokens` while there is no preemption: otherwise
several requests could together fill the pool at a page boundary and get stuck forever
before the next decode token.

`CacheManager::from_budget` statically splits the available VRAM for a given
`target_context`. The number of state slots equals `Budget::max_concurrency`, the rest
goes to the KV pool. If real requests are noticeably shorter than the target, state slots
run out first; if longer, KV blocks do.

The GPU executor uses the same shared pool of physical pages as
`CacheManager`. `max_context` sets the width of one sequence's logical block table,
but not the size of the physical allocation. The allocation size is set
independently as `kv_pool_blocks`; every page ID from the scheduler layout is checked
against this capacity before metadata is uploaded. This allows, for example, keeping
`max_context=32768` and 80 state slots without reserving `80 × 32768` KV tokens.

## Request states

```text
waiting
  -> prefill-ready <-> prefill-in-flight
  -> decode-ready  <-> decode-in-flight
  -> completed / cancelled -> release(state + KV)
```

At most one in-flight batch exists at a time. This is not a GPU
limitation: a batch can hold up to `max_num_seqs` sequences. The limitation
is needed at the CPU/GPU boundary so that commit and rollback always refer to
the single known plan.

Executor lifecycle:

1. `submit` puts the request in a FIFO without taking VRAM yet.
2. `next_batch` admits requests that fit and returns a mixed `Batch`.
3. The executor runs the `decode` and `prefill` parts of the batch.
4. On success it calls `complete_batch(stopped)`, on an error before execution —
   `abort_batch()`.

## Scheduling policy

1. Ready decode has priority: this protects the inter-token latency of already
   running requests.
2. The remaining sequence slots and token budget are filled with chunked prefill.
3. Decode is served round-robin.
4. Admission is FIFO, but a long prompt that temporarily doesn't fit
   in free KV doesn't block a smaller request behind it. Lack of a state slot
   stops admission entirely, because the next request wouldn't be able to
   get through either.

`max_num_batched_tokens` is shared: one decode costs one token, prefill costs the chunk
size. So in one step the scheduler can keep decode of active
sequences going and use the rest of the GPU work for a new prompt.

## Reservation before the GPU

Before adding a sequence to the decode batch, the scheduler calls
`CacheManager::append_token`. If the next token crosses a page boundary,
a new KV block is allocated **before** the kernel launches. OOM between planning and
execution is impossible.

The lifetime capacity commit is logical: real pages are still allocated
lazily. So block tables contain only pages already in use, but
admission doesn't promise the free remainder to other requests. The conservatism can
be lifted together with the arrival of preemption, rather than at the cost of a deadlock.

If a batch was not executed, `abort_batch` calls `rollback_token`: the length counter
goes back, and the extra KV page is freed. Prefill is easier to roll back
— the whole prompt cache is reserved at admission, and the committed offset
changes only in `complete_batch`.

Finishing on the token limit, EOS/stop condition and explicit cancellation free both
resources. Cancelling an in-flight sequence is forbidden: the batch must first be
committed or rolled back so the GPU doesn't get a dangling block table.

## Batch metadata for the GPU

`BatchLayout` turns the scheduler batch and current cache state into a flat
structure-of-arrays layout of `u32`. Decode entries come first, then
prefill chunks. For each sequence it passes:

* the `SeqId` and state slot number;
* the start position and visible context length;
* the CSR offset of the input tokens;
* the CSR offset and physical IDs of KV blocks.

For decode the length already includes the reserved next token. For prefill
it ends at the current chunk boundary, even if admission has already
allocated pages for the whole prompt. This layout has no Rust padding and
is copied into preallocated fixed-address device buffers. The CUDA graph
reads new values from the same addresses on every launch.

## Verification

```bash
cargo test -p qwc-runtime
cargo clippy -p qwc-runtime --all-targets -- -D warnings \
  -A clippy::manual-is-multiple-of
```

CPU tests cover both kinds of shortage, lazy KV growth, rollback at
a page boundary, chunked prefill, mixed batch, first-fit admission, the generation
limit, stop condition, safe cancellation and the CSR layout of GPU metadata.

`qwc-engine::Executor::execute_layout` takes this layout directly:
decode uses the scheduler-selected state slots and physical KV pages, and
prefill chunks update the same pools. An example of the full loop is in the binary
`qwc-engine/src/bin/scheduled.rs`. Preemption, swap and prefix snapshots are
intentionally not part of the current runtime.
