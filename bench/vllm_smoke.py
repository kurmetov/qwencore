"""Check: does Qwen3.8-27B-NVFP4 load in vLLM on a single RTX 5090.

Key questions:
  1. is the hybrid Qwen3_5 architecture supported;
  2. is --enforce-eager needed (i.e. does CUDA graph capture OOM);
  3. how much VRAM the weights actually take.
"""
import os
import subprocess
import sys
import time

MODEL = os.path.expanduser("~/models/Qwen3.8-27B-QUASAR-NVFP4")
EAGER = "--eager" in sys.argv
MAXLEN = int(os.environ.get("MAXLEN", "4096"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.80"))


def vram_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def main():
    # vLLM uses multiprocessing with spawn: creating the LLM must be
    # protected by a main guard, otherwise the child process re-executes the script.
    # Launching via an absolute python path doesn't activate the venv, so
    # otherwise FlashInfer won't find the ninja executable installed next to it.
    venv_bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = os.pathsep.join((venv_bin, os.environ.get("PATH", "")))

    from vllm import LLM, SamplingParams

    print(
        f"mode: {'eager' if EAGER else 'CUDA graphs'}, "
        f"max_model_len={MAXLEN}, gpu_memory_utilization={GPU_UTIL:.2f}"
    )
    print(f"VRAM before load: {vram_mib()} MiB", flush=True)

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager=EAGER,
        disable_log_stats=True,
    )
    load_s = time.time() - t0
    print(f"load: {load_s:.1f} s")
    print(f"VRAM after load: {vram_mib()} MiB", flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=64)
    t0 = time.time()
    out = llm.generate(["Explain what linear attention is."], params)
    gen_s = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"generated {n} tokens in {gen_s:.2f} s -> {n / gen_s:.1f} tok/s")
    print("OK")


if __name__ == "__main__":
    main()
