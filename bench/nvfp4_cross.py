#!/usr/bin/env python3
"""Cross-check QWC's NVFP4 quantizer and W4A4 GEMM against vLLM's CUTLASS path.

The reference engine selects CutlassNvFp4LinearKernel for every linear layer,
so both engines should produce the same FP4 codes, the same block scales and
the same GEMM output for one projection on identical activations. Any
difference here is a systematic arithmetic divergence, not a trajectory effect.

    .venvs/baseline/bin/python bench/nvfp4_cross.py --model PATH
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408


def make_activations(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Scaled to the magnitude of a post-RMSNorm hidden state.
    return (torch.randn(rows, HIDDEN_SIZE, generator=generator) * 0.35).to(torch.bfloat16)


def find_shard(model: Path, tensor: str) -> Path:
    index = model / "model.safetensors.index.json"
    if index.exists():
        import json

        mapping = json.loads(index.read_text())["weight_map"]
        return model / mapping[tensor]
    return model / "model.safetensors"


def load_weight(model: Path, prefix: str):
    out = {}
    for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
        name = f"{prefix}.{suffix}"
        shard = find_shard(model, name)
        with safe_open(shard, framework="pt") as handle:
            out[suffix] = handle.get_tensor(name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/qwc-nvfp4-cross"))
    parser.add_argument(
        "--tensor", default="model.language_model.layers.0.mlp.gate_proj"
    )
    parser.add_argument("--out-features", type=int, default=INTERMEDIATE_SIZE)
    args = parser.parse_args()

    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_weight_for_cutlass,
        slice_nvfp4_output,
        swizzle_blockscale,
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    acts = make_activations(args.rows, args.seed)
    acts_path = args.workdir / "acts.bin"
    acts_path.write_bytes(acts.view(torch.uint16).numpy().tobytes())

    prefix = args.workdir / "qwc"
    command = [
        "cargo", "run", "--release", "--quiet", "-p", "qwc-cuda",
        "--example", "nvfp4_cross", "--",
        str(args.model), str(acts_path), str(prefix),
        args.tensor, str(args.out_features),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout + result.stderr)
        return 1
    print(result.stdout.strip())

    tensors = load_weight(args.model, args.tensor)
    input_global_scale = tensors["input_global_scale"].max().to(torch.float32).cuda()
    weight_global_scale = tensors["weight_global_scale"].max().to(torch.float32).cuda()
    alpha = (1.0 / input_global_scale) * (1.0 / weight_global_scale)

    device_acts = acts.cuda()
    ref_codes, ref_scales = scaled_fp4_quant(
        device_acts, input_global_scale, is_sf_swizzled_layout=False
    )

    qwc_codes = np.frombuffer((prefix.with_suffix(prefix.suffix + ".codes")).read_bytes(), dtype=np.uint8)
    qwc_scales = np.frombuffer((prefix.with_suffix(prefix.suffix + ".scales")).read_bytes(), dtype=np.uint8)
    ref_codes_flat = ref_codes.view(torch.uint8).cpu().numpy().reshape(-1)
    ref_scales_flat = ref_scales.view(torch.uint8).cpu().numpy().reshape(-1)[: qwc_scales.size]

    code_mismatch = int((qwc_codes != ref_codes_flat).sum())
    # Each byte holds two FP4 codes; count nibbles for an interpretable rate.
    nibble_mismatch = int(
        ((qwc_codes & 0x0F) != (ref_codes_flat & 0x0F)).sum()
        + ((qwc_codes >> 4) != (ref_codes_flat >> 4)).sum()
    )
    scale_mismatch = int((qwc_scales != ref_scales_flat).sum())
    total_nibbles = qwc_codes.size * 2
    print(f"fp4 codes:    {nibble_mismatch}/{total_nibbles} nibbles differ "
          f"({100.0 * nibble_mismatch / total_nibbles:.4f}%), {code_mismatch} bytes")
    print(f"block scales: {scale_mismatch}/{qwc_scales.size} differ "
          f"({100.0 * scale_mismatch / qwc_scales.size:.4f}%)")

    packed_weight = tensors["weight_packed"].cuda()
    swizzled = swizzle_blockscale(tensors["weight_scale"].cuda())
    padded_weight, padding_cols = pad_nvfp4_weight_for_cutlass(packed_weight)
    x_fp4, x_blockscale = scaled_fp4_quant(
        device_acts,
        input_global_scale,
        is_sf_swizzled_layout=True,
        backend="cutlass",
        padded_n=device_acts.shape[-1] + padding_cols * 2,
    )
    reference_out = cutlass_scaled_fp4_mm(
        x_fp4, padded_weight, x_blockscale, swizzled, alpha, torch.bfloat16
    )
    reference_out = slice_nvfp4_output(reference_out, args.out_features)
    reference = reference_out.to(torch.float32).cpu().numpy()

    # Ground truth: dequantize the weight exactly and matmul in FP32. Neither
    # engine computes this, so it says which of the two is nearer the real
    # value instead of only how far apart they are.
    codes = packed_weight
    low = (codes & 0x0F).to(torch.int32)
    high = (codes >> 4).to(torch.int32)
    nibbles = torch.stack((low, high), dim=-1).reshape(args.out_features, HIDDEN_SIZE)
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=nibbles.device,
        dtype=torch.float32,
    )
    values = lookup[nibbles]
    block_scale = tensors["weight_scale"].cuda().to(torch.float32)
    values = values.reshape(args.out_features, HIDDEN_SIZE // 16, 16)
    dequantized = (values * block_scale.unsqueeze(-1)).reshape(
        args.out_features, HIDDEN_SIZE
    ) / weight_global_scale
    truth = (device_acts.to(torch.float32) @ dequantized.T).cpu().numpy()

    reference_error = np.abs(reference - truth)
    print(f"vllm w4a4 vs ground truth: max |Δ| {reference_error.max():.5f}  "
          f"mean |Δ| {reference_error.mean():.5f}")

    w4a16_rows = min(args.rows, 4)
    for name, live in (("w4a4", args.rows), ("w4a16", w4a16_rows)):
        path = prefix.with_suffix(prefix.suffix + "." + name)
        actual = np.frombuffer(path.read_bytes(), dtype=np.float32).reshape(
            args.rows, args.out_features
        )[:live]
        difference = np.abs(actual - reference[:live])
        spread = float(np.abs(reference[:live]).max())
        truth_difference = np.abs(actual - truth[:live])
        print(f"{name} ({live} rows): vs vllm max |Δ| {difference.max():.5f} "
              f"mean {difference.mean():.5f} | vs truth max |Δ| "
              f"{truth_difference.max():.5f} mean {truth_difference.mean():.5f} "
              f"| range ±{spread:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
