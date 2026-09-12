"""Проверка: грузится ли Qwen3.8-27B-NVFP4 в vLLM на одной RTX 5090.

Ключевые вопросы:
  1. поддерживается ли гибридная архитектура Qwen3_5;
  2. нужен ли --enforce-eager (то есть падает ли захват CUDA-графов в OOM);
  3. сколько VRAM реально занято весами.
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
    # vLLM использует multiprocessing со spawn: создание LLM обязано быть
    # защищено main-guard, иначе дочерний процесс повторно исполнит скрипт.
    # Запуск через абсолютный путь к python не активирует venv, поэтому
    # FlashInfer иначе не найдёт установленный рядом executable ninja.
    venv_bin = os.path.dirname(sys.executable)
    os.environ["PATH"] = os.pathsep.join((venv_bin, os.environ.get("PATH", "")))

    from vllm import LLM, SamplingParams

    print(
        f"режим: {'eager' if EAGER else 'CUDA graphs'}, "
        f"max_model_len={MAXLEN}, gpu_memory_utilization={GPU_UTIL:.2f}"
    )
    print(f"VRAM до загрузки: {vram_mib()} MiB", flush=True)

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager=EAGER,
        disable_log_stats=True,
    )
    load_s = time.time() - t0
    print(f"загрузка: {load_s:.1f} с")
    print(f"VRAM после загрузки: {vram_mib()} MiB", flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=64)
    t0 = time.time()
    out = llm.generate(["Объясни, что такое linear attention."], params)
    gen_s = time.time() - t0
    n = len(out[0].outputs[0].token_ids)
    print(f"сгенерировано {n} токенов за {gen_s:.2f} с -> {n / gen_s:.1f} tok/s")
    print("ОК")


if __name__ == "__main__":
    main()
