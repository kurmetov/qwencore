#!/usr/bin/env bash
# Downloads the target checkpoint. Resuming is supported: it can be interrupted.
set -uo pipefail

REPO="${REPO:-QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4}"
DEST="${DEST:-$HOME/models/Qwen3.8-27B-QUASAR-NVFP4}"
BASE="https://huggingface.co/$REPO/resolve/main"

FILES=(
    config.json
    generation_config.json
    model.safetensors.index.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    chat_template.jinja
    preprocessor_config.json
    video_preprocessor_config.json
    model-00001-of-00005.safetensors
    model-00002-of-00005.safetensors
    model-00003-of-00005.safetensors
    model-00004-of-00005.safetensors
    model-00005-of-00005.safetensors
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
    out="$DEST/$f"
    want=$(curl -sIL "$BASE/$f" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
        echo "== $f already present ($(numfmt --to=iec "$have"))"
        continue
    fi
    echo "== $f -> $(numfmt --to=iec "${want:-0}")"
    curl -sL -C - --retry 5 --retry-delay 5 -o "$out" "$BASE/$f" || {
        echo "!! failed: $f"; exit 1; }
done

echo "== done, total $(du -sh "$DEST" | cut -f1)"
