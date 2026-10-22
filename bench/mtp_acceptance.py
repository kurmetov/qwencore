#!/usr/bin/env python3
"""MTP head acceptance rate, offline and without any speculation machinery.

The head takes (h_t, embedding of token t+1) and must predict token t+2.
Hidden states and tokens are dumped by `qwc-engine --bin mtpdump`; here the
head itself is computed (torch, weights as-is from the checkpoint) and its argmax is compared with
what the real model produced.

    python bench/mtp_acceptance.py --dumps /tmp/mtp

The measurement gives an upper bound on acceptance: the head here sees only accepted
tokens, i.e. exactly the history it would see during speculation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4"
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
ROPE_DIMS = 64          # partial RoPE: only the first quarter of dimensions


def read_dump(path: Path):
    with open(path, "rb") as handle:
        blob = handle.read()
    assert blob[:8] == b"QWCMTP01", f"{path}: unknown format"
    hidden, positions, prompt_len, token_count = struct.unpack_from("<IIII", blob, 8)
    offset = 24
    tokens = torch.frombuffer(
        bytearray(blob[offset : offset + 4 * token_count]), dtype=torch.int32
    ).clone()
    offset += 4 * token_count
    raw = torch.frombuffer(
        bytearray(blob[offset : offset + 2 * positions * hidden]), dtype=torch.bfloat16
    ).clone()
    return tokens.long(), raw.view(positions, hidden), int(prompt_len)


def load_weights(model_dir: Path, device: str):
    index = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    wanted = {name: shard for name, shard in index.items()
              if name.startswith("mtp.")
              or name.endswith("embed_tokens.weight")
              or name == "lm_head.weight"}
    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)
    tensors = {}
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name).to(device)
    return tensors


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Gemma variant: scale is (1 + weight), not weight.

    The whole model normalizes this way (`rmsnorm.cu` does the same). The usual
    form gives exactly zero matches at the head output: the vector's direction
    drifts away entirely.
    """
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (1.0 + weight.float())).to(dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float):
    # x: [tokens, heads, HEAD_DIM]; only the first ROPE_DIMS are rotated.
    half = ROPE_DIMS // 2
    inv = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, None] * inv[None, :]
    cos, sin = angles.cos(), angles.sin()
    rotated = x[..., :ROPE_DIMS].float()
    first, second = rotated[..., :half], rotated[..., half:]
    out = torch.cat(
        [first * cos[:, None, :] - second * sin[:, None, :],
         second * cos[:, None, :] + first * sin[:, None, :]],
        dim=-1,
    )
    return torch.cat([out.to(x.dtype), x[..., ROPE_DIMS:]], dim=-1)


def mtp_forward(w: dict, hidden: torch.Tensor, next_tokens: torch.Tensor,
                positions: torch.Tensor, theta: float):
    """One draft-head layer over the whole sequence at once.

    Position t gets h_t and the embedding of token t+1 — the same pairing as in
    speculation; causal attention within the layer makes the history the same as
    it would be with step-by-step KV-cache operation.
    """
    embed = w["model.language_model.embed_tokens.weight"][next_tokens]
    x = torch.cat(
        [rms_norm(embed, w["mtp.pre_fc_norm_embedding.weight"]),
         rms_norm(hidden, w["mtp.pre_fc_norm_hidden.weight"])],
        dim=-1,
    )
    x = x @ w["mtp.fc.weight"].t()

    residual = x
    h = rms_norm(x, w["mtp.layers.0.input_layernorm.weight"])

    # q and gate are interleaved per head: [head][q | gate], not two
    # consecutive halves. Splitting in half along the last axis is a classic
    # trap that gives exactly zero matches.
    qg = h @ w["mtp.layers.0.self_attn.q_proj.weight"].t()
    qg = qg.view(h.shape[0], NUM_Q_HEADS, 2 * HEAD_DIM)
    q, gate = qg.chunk(2, dim=-1)
    q = q.reshape(h.shape[0], -1)
    gate = gate.reshape(h.shape[0], -1)
    k = h @ w["mtp.layers.0.self_attn.k_proj.weight"].t()
    v = h @ w["mtp.layers.0.self_attn.v_proj.weight"].t()
    tokens = h.shape[0]
    q = q.view(tokens, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(tokens, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(tokens, NUM_KV_HEADS, HEAD_DIM)
    q = rms_norm(q, w["mtp.layers.0.self_attn.q_norm.weight"])
    k = rms_norm(k, w["mtp.layers.0.self_attn.k_norm.weight"])
    q = rope(q, positions, theta)
    k = rope(k, positions, theta)

    group = NUM_Q_HEADS // NUM_KV_HEADS
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    attn = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).float(), k.transpose(0, 1).float(),
        v.transpose(0, 1).float(), is_causal=True,
    ).transpose(0, 1)
    attn = (attn.reshape(tokens, NUM_Q_HEADS * HEAD_DIM).to(h.dtype)
            * torch.sigmoid(gate.float()).to(h.dtype))
    x = residual + attn @ w["mtp.layers.0.self_attn.o_proj.weight"].t()

    residual = x
    h = rms_norm(x, w["mtp.layers.0.post_attention_layernorm.weight"])
    gate_proj = h @ w["mtp.layers.0.mlp.gate_proj.weight"].t()
    up = h @ w["mtp.layers.0.mlp.up_proj.weight"].t()
    x = residual + (torch.nn.functional.silu(gate_proj.float()).to(h.dtype) * up) \
        @ w["mtp.layers.0.mlp.down_proj.weight"].t()
    return rms_norm(x, w["mtp.norm.weight"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chain", type=int, default=3,
                        help="draft chain depth: the second and later ones follow "
                             "the head's own output")
    args = parser.parse_args()

    config = json.load(open(args.model / "config.json"))
    text = config.get("text_config", config)
    rope_parameters = text.get("rope_parameters", text)
    theta = float(rope_parameters.get("rope_theta", text.get("rope_theta", 10000.0)))

    w = load_weights(args.model, args.device)
    head = w.get("lm_head.weight")
    if head is None:
        head = w["model.language_model.embed_tokens.weight"]
    print(f"weights loaded on {args.device}, rope_theta={theta:g}\n")

    print(f"{'case':>12} {'positions':>8} {'prompt':>8} {'generation':>10} {'total':>7}")
    print(f"{'':->12} {'':->8} {'':->8} {'':->10} {'':->7}")
    total_hit = total = prompt_hit = prompt_total = gen_hit = gen_total = 0
    # Chain: how often the k-th draft is accepted given that all previous ones
    # were accepted. Only over generated positions — the prompt doesn't take part
    # in speculation.
    chain_hit = [0] * args.chain
    chain_total = [0] * args.chain
    for path in sorted(glob.glob(str(args.dumps / "*.bin"))):
        tokens, hidden, prompt_len = read_dump(Path(path))
        hidden = hidden.to(args.device)
        tokens = tokens.to(args.device)
        positions = torch.arange(hidden.shape[0], device=args.device)
        usable = hidden.shape[0] - 1
        out = mtp_forward(w, hidden[:usable], tokens[1 : usable + 1],
                          positions[:usable], theta)
        predicted = (out.float() @ head.float().t()).argmax(-1)
        target = tokens[2 : usable + 2]
        hit = (predicted == target)
        case = Path(path).stem
        in_prompt = torch.arange(usable, device=args.device) < prompt_len - 1
        p_hit = int(hit[in_prompt].sum()), int(in_prompt.sum())
        g_hit = int(hit[~in_prompt].sum()), int((~in_prompt).sum())
        print(f"{case:>12} {usable:>8} {p_hit[0]/max(p_hit[1],1):>8.3f} "
              f"{g_hit[0]/max(g_hit[1],1):>10.3f} {int(hit.sum())/usable:>7.3f}")
        total_hit += int(hit.sum()); total += usable
        prompt_hit += p_hit[0]; prompt_total += p_hit[1]
        gen_hit += g_hit[0]; gen_total += g_hit[1]

        # Second and later drafts: the head follows its own output, as
        # in real speculation. The one approximation — its own positions don't
        # get into the KV of neighbouring drafts, so the estimate is slightly low.
        alive = ~in_prompt
        state, token_in, offset = out, predicted, 0
        for depth in range(args.chain):
            if depth > 0:
                state = mtp_forward(w, state, token_in, positions[:usable] + depth, theta)
                token_in = (state.float() @ head.float().t()).argmax(-1)
            limit = usable - depth - 2
            if limit <= 0:
                break
            step_hit = token_in[:limit] == tokens[depth + 2 : depth + 2 + limit]
            mask = alive[:limit]
            chain_hit[depth] += int(step_hit[mask].sum())
            chain_total[depth] += int(mask.sum())
            alive = torch.zeros_like(alive)
            alive[:limit] = mask & step_hit

    print(f"\nacceptance on prompts:    {prompt_hit/max(prompt_total,1):.3f} ({prompt_total} positions)")
    print(f"acceptance on generation: {gen_hit/max(gen_total,1):.3f} ({gen_total} positions)")
    print(f"acceptance overall:       {total_hit/max(total,1):.3f} ({total} positions)")

    print("\ndraft chain (conditional probability of accepting the k-th when the previous ones were accepted):")
    expected = 1.0
    survive = 1.0
    for depth in range(args.chain):
        if chain_total[depth] == 0:
            break
        p = chain_hit[depth] / chain_total[depth]
        survive *= p
        expected += survive
        print(f"  k={depth+1}: p={p:.3f} ({chain_total[depth]} cases), "
              f"E[tokens per step] = {expected:.2f}")

if __name__ == "__main__":
    main()
