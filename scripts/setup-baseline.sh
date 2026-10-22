#!/usr/bin/env bash
# Environment for the baseline engines. A separate venv via uv: the system Python 3.14
# is newer than vLLM supports, and it shouldn't be touched.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/.venvs/baseline}"
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 "$ENV_DIR"
# sm_120 (consumer Blackwell) requires torch builds with CUDA >= 12.8.
VIRTUAL_ENV="$ENV_DIR" uv pip install --python "$ENV_DIR/bin/python" vllm

"$ENV_DIR/bin/python" - <<'PY'
import torch, vllm
print("torch  ", torch.__version__, "cuda", torch.version.cuda)
print("vllm   ", vllm.__version__)
print("device ", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
