#!/usr/bin/env python3
"""Кодировщик и декодировщик Qwen3.5 BPE на голой стандартной библиотеке.

Нужен только для проверки движка: без него генерация — это список чисел.
Pre-tokenizer приближён ASCII-регуляркой (в стандартном `re` нет \\p{L}),
поэтому для не-ASCII текста лучше брать настоящий tokenizers.

    scripts/tok.py encode "The capital of France is"
    scripts/tok.py decode "785,6722,315"
"""

import functools
import json
import os
import re
import sys

MODEL = os.path.expanduser("~/models/Qwen3.8-27B-QUASAR-NVFP4/tokenizer.json")
# Разбиение как в tokenizer.json, но с ASCII-классами вместо \p{L}/\p{N}.
PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\nA-Za-z0-9]?[A-Za-z]+"
    r"|[0-9]"
    r"| ?[^\sA-Za-z0-9]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def byte_encoder():
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    mapping = {b: chr(b) for b in printable}
    shift = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + shift)
            shift += 1
    return mapping


@functools.lru_cache(maxsize=1)
def tokenizer():
    with open(MODEL, encoding="utf-8") as handle:
        data = json.load(handle)
    vocab = data["model"]["vocab"]
    raw_merges = data["model"]["merges"]
    ranks = {}
    for index, merge in enumerate(raw_merges):
        pair = tuple(merge) if isinstance(merge, list) else tuple(merge.split(" "))
        ranks[pair] = index
    specials = {t["content"]: t["id"] for t in data.get("added_tokens", [])}
    vocab.update(specials)
    inverse = {index: token for token, index in vocab.items()}
    return vocab, ranks, inverse, specials


def bpe(word, ranks):
    symbols = list(word)
    while len(symbols) > 1:
        pairs = {
            (symbols[i], symbols[i + 1]): i
            for i in range(len(symbols) - 1)
        }
        candidates = [pair for pair in pairs if pair in ranks]
        if not candidates:
            break
        best = min(candidates, key=lambda pair: ranks[pair])
        merged = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and (symbols[i], symbols[i + 1]) == best
            ):
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


def encode(text):
    vocab, ranks, _, specials = tokenizer()
    encoder = byte_encoder()
    ids = []
    # Специальные токены вырезаются целиком: они не проходят через BPE.
    parts = re.split("(" + "|".join(re.escape(s) for s in specials) + ")", text) if specials else [text]
    for part in parts:
        if not part:
            continue
        if part in specials:
            ids.append(specials[part])
            continue
        for piece in PATTERN.findall(part):
            mapped = "".join(encoder[b] for b in piece.encode("utf-8"))
            for symbol in bpe(mapped, ranks):
                if symbol not in vocab:
                    raise SystemExit(f"нет токена для {symbol!r}")
                ids.append(vocab[symbol])
    return ids


def decode(ids):
    _, _, inverse, _ = tokenizer()
    encoder = byte_encoder()
    decoder = {char: b for b, char in encoder.items()}
    text = "".join(inverse.get(int(i), "") for i in ids)
    return bytes(decoder.get(char, 0) for char in text).decode("utf-8", errors="replace")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    command, payload = sys.argv[1], sys.argv[2]
    if command == "encode":
        print(",".join(str(i) for i in encode(payload)))
    elif command == "decode":
        print(decode(payload.replace("[", "").replace("]", "").split(",")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
