#!/usr/bin/env python3
"""Корпус реальных промптов для servebench — вместо арифметической прогрессии.

Синтетический `prompt_ids` — это `1000 + (id*7919 + i) % 20000`, то есть шаг с
постоянным приращением. На нём черновая голова угадывает продолжение почти
всегда (acceptance 0.97), и любая спекуляция выглядит лучше, чем она есть.
Здесь промпты нарезаются из настоящего текста репозитория: проза документации
и исходники — ровно то, что видит агентный сервер.

    python bench/make_corpus.py --requests 200 --prompt-tokens 256

Каждая запись — ровно `--prompt-tokens` токенов, куски не пересекаются, так
что общий префикс не даст движку с prefix-кэшем пропустить префилл.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4"


def sources() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("docs/**/*.md", "crates/**/*.rs", "crates/**/*.cu"):
        paths.extend(p for p in ROOT.glob(pattern) if "target/" not in str(p))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "bench/corpus/serve.jsonl"
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    # Куски режутся внутри файла, а файлы перемешиваются: так соседние запросы
    # в батче приходят из разных мест, как у независимых пользователей.
    chunks: list[tuple[str, list[int]]] = []
    for path in sources():
        text = path.read_text(errors="ignore")
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        name = str(path.relative_to(ROOT))
        for start in range(0, len(ids) - args.prompt_tokens + 1, args.prompt_tokens):
            chunks.append((f"{name}:{start}", ids[start : start + args.prompt_tokens]))

    if len(chunks) < args.requests:
        raise SystemExit(
            f"нарезалось {len(chunks)} кусков по {args.prompt_tokens} токенов, "
            f"а нужно {args.requests}"
        )
    random.Random(args.seed).shuffle(chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for name, ids in chunks[: args.requests]:
            handle.write(
                json.dumps({"id": name, "prompt_token_ids": ids}, ensure_ascii=False)
                + "\n"
            )
    print(
        f"{args.output}: {args.requests} промптов по {args.prompt_tokens} токенов "
        f"(доступно было {len(chunks)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
