#!/usr/bin/env python3
"""A corpus of real prompts for servebench — instead of an arithmetic progression.

Synthetic `prompt_ids` is `1000 + (id*7919 + i) % 20000`, i.e. a step with
a constant increment. On it the draft head guesses the continuation almost
always (acceptance 0.97), and any speculation looks better than it is.
Here prompts are sliced from the repository's real text: documentation prose
and sources — exactly what an agentic server sees.

    python bench/make_corpus.py --requests 200 --prompt-tokens 256

Each record is exactly `--prompt-tokens` tokens, chunks don't overlap, so
a shared prefix won't let an engine with a prefix cache skip prefill.
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

    # Chunks are cut within a file, and files are shuffled: so neighbouring requests
    # in a batch come from different places, like independent users.
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
            f"sliced {len(chunks)} chunks of {args.prompt_tokens} tokens, "
            f"but {args.requests} are needed"
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
        f"{args.output}: {args.requests} prompts of {args.prompt_tokens} tokens "
        f"({len(chunks)} were available)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
