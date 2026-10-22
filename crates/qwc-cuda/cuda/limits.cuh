#pragma once

// Row caps shared by all step kernels.
//
// One engine step carries at most kMaxStepRows rows: that's the prefill arena
// capacity (`PREFILL_CHUNK_SIZE` in the engine, `MAX_STEP_ROWS` in Rust). Checks
// in kernels catch garbage arguments rather than describe their structure — their grids
// scale linearly in rows.
//
// The row-wise decode path is a separate case: its partition layout
// is sized for the number of sequence slots, not the chunk size.
namespace qwc {

constexpr int kMaxStepRows = 2048;
constexpr int kMaxDecodeRows = 1024;

}  // namespace qwc
