# Differential eval

Reference: `qwc` 0.1.0

Distribution metrics stop after the first greedy divergence, because later logits use different contexts.

| Candidate | Cases | Exact prefix | Top-1 | Top-k Jaccard | Logprob MAE | Max Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| qwc | 14/14 | 0.967 | 0.991 | 0.853 | 0.286 | 5.826 | FAIL |

Thresholds: top-1 ≥ 0.950, top-k Jaccard ≥ 0.700, logprob MAE ≤ 0.250.

## qwc

| Case | Prefix | Same-context steps | Status |
|---|---:|---:|---|
| en-fact | 16 | 16 | ok |
| code | 16 | 16 | ok |
| ru | 12 | 12 | ok |
| zh | 12 | 12 | ok |
| ar | 12 | 12 | ok |
| math | 8 | 9 | ok |
| whitespace | 8 | 8 | ok |
| page-63 | 4 | 4 | ok |
| page-64 | 4 | 4 | ok |
| page-65 | 4 | 4 | ok |
| chunk-127 | 4 | 4 | ok |
| chunk-128 | 4 | 4 | ok |
| chunk-129 | 4 | 4 | ok |
| long-601 | 8 | 8 | ok |
