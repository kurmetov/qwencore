# Differential eval

Reference: `vllm` 0.28.0

Distribution metrics stop after the first greedy divergence, because later logits use different contexts.

| Candidate | Cases | Exact prefix | Top-1 | Top-k Jaccard | Logprob MAE | Max Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| qwc-b1 | 14/14 | 1.000 | 1.000 | 0.782 | 0.364 | 3.868 | FAIL |
| qwc-b1-fp32-delta-state | 14/14 | 1.000 | 1.000 | 0.781 | 0.380 | 3.315 | FAIL |

Thresholds: top-1 ≥ 0.950, top-k Jaccard ≥ 0.700, logprob MAE ≤ 0.250.

## qwc-b1

| Case | Prefix | Same-context steps | Status |
|---|---:|---:|---|
| en-fact | 16 | 16 | ok |
| code | 16 | 16 | ok |
| ru | 12 | 12 | ok |
| zh | 12 | 12 | ok |
| ar | 12 | 12 | ok |
| math | 12 | 12 | ok |
| whitespace | 8 | 8 | ok |
| page-63 | 4 | 4 | ok |
| page-64 | 4 | 4 | ok |
| page-65 | 4 | 4 | ok |
| chunk-127 | 4 | 4 | ok |
| chunk-128 | 4 | 4 | ok |
| chunk-129 | 4 | 4 | ok |
| long-601 | 8 | 8 | ok |

## qwc-b1-fp32-delta-state

| Case | Prefix | Same-context steps | Status |
|---|---:|---:|---|
| en-fact | 16 | 16 | ok |
| code | 16 | 16 | ok |
| ru | 12 | 12 | ok |
| zh | 12 | 12 | ok |
| ar | 12 | 12 | ok |
| math | 12 | 12 | ok |
| whitespace | 8 | 8 | ok |
| page-63 | 4 | 4 | ok |
| page-64 | 4 | 4 | ok |
| page-65 | 4 | 4 | ok |
| chunk-127 | 4 | 4 | ok |
| chunk-128 | 4 | 4 | ok |
| chunk-129 | 4 | 4 | ok |
| long-601 | 8 | 8 | ok |
