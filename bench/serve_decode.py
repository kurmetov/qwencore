#!/usr/bin/env python3
"""Время decode-шага у запущенного `serve` на длинном промпте.

Шаг берётся наклоном между запросами в 1 и N+1 токен: префилл, прогрев
головы и накладные расходы запроса сокращаются. Промпты — склейка документов
`bench/corpus/state-length.jsonl`, у каждого одновременного запроса свой
порядок документов. Кэш префиксов на сервере должен быть выключен
(`--prefix-cache 0`), иначе повторный запрос не пересчитывает промпт и наклон
врёт.

  bench/serve_decode.py --lengths 8192,30000 --concurrency 1,2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def documents() -> dict[str, list[int]]:
    rows = {}
    for line in open(ROOT / "bench/corpus/state-length.jsonl"):
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row["prompt_token_ids"]
    return rows


def prompt(rows: dict[str, list[int]], length: int, seed: int) -> list[int]:
    tokens = [token for index in range(3) for token in rows[f"doc{(seed + index) % 3}-16384"]]
    if length > len(tokens):
        raise SystemExit(f"--lengths: at most {len(tokens)} tokens")
    return tokens[:length]


def call(args: argparse.Namespace, tokens: list[int], new_tokens: int) -> None:
    body = json.dumps(
        {"model": args.model, "prompt": tokens, "max_tokens": new_tokens, "ignore_eos": True}
    ).encode()
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/completions", body, {"content-type": "application/json"}
    )
    reply = json.loads(urllib.request.urlopen(request, timeout=600).read())
    if reply["usage"]["completion_tokens"] != new_tokens:
        raise SystemExit(f"server returned {reply['usage']} instead of {new_tokens} tokens")


def concurrent(args: argparse.Namespace, prompts: list[list[int]], new_tokens: int) -> float:
    threads = [
        threading.Thread(target=call, args=(args, tokens, new_tokens)) for tokens in prompts
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-27B-QUASAR-NVFP4")
    parser.add_argument("--lengths", default="8192,30000")
    parser.add_argument("--concurrency", default="1,2")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2, help="minimum over repeats")
    args = parser.parse_args()
    rows = documents()
    for length in map(int, args.lengths.split(",")):
        for streams in map(int, args.concurrency.split(",")):
            prompts = [prompt(rows, length, seed) for seed in range(streams)]
            concurrent(args, prompts, 1)
            short = min(concurrent(args, prompts, 1) for _ in range(args.repeats))
            long = min(
                concurrent(args, prompts, args.new_tokens + 1) for _ in range(args.repeats)
            )
            step_ms = (long - short) / args.new_tokens * 1e3
            print(
                f"ctx {length:>6} c={streams}: {step_ms:6.2f} ms/step, "
                f"{streams * 1e3 / step_ms:6.1f} tok/s total",
                flush=True,
            )


if __name__ == "__main__":
    main()
