#!/usr/bin/env python3
"""Differential token/log-probability evaluation across LLM runtimes.

The corpus freezes prompt token IDs. Every adapter either consumes those exact
IDs or marks the case unsupported; silently retokenizing is never allowed.

Examples:
  python bench/diff_eval.py validate-corpus
  python bench/diff_eval.py run --engine qwc --output bench/results/qwc-b1.jsonl
  python bench/diff_eval.py run --engine vllm --output bench/results/vllm.jsonl
  python bench/diff_eval.py compare --reference bench/results/vllm.jsonl \
      bench/results/qwc-b1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4"
DEFAULT_CORPUS = ROOT / "bench/corpus/core.jsonl"
SCHEMA_VERSION = 1


class EvalError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise EvalError(f"{path}:{line_number}: {error}") from error
    return records


def expand_case(case: dict[str, Any]) -> dict[str, Any]:
    result = dict(case)
    direct = result.get("prompt_token_ids")
    repeat = result.get("repeat")
    if direct and repeat:
        raise EvalError(f"{result.get('id')}: prompt_token_ids and repeat are exclusive")
    if repeat:
        pattern = repeat.get("token_ids", [])
        length = int(repeat.get("length", 0))
        if not pattern or length <= 0:
            raise EvalError(f"{result.get('id')}: invalid repeat specification")
        direct = [pattern[index % len(pattern)] for index in range(length)]
    if not direct:
        raise EvalError(f"{result.get('id')}: no prompt tokens")
    result["prompt_token_ids"] = [int(token) for token in direct]
    result["max_new_tokens"] = int(result.get("max_new_tokens", 16))
    if result["max_new_tokens"] <= 0:
        raise EvalError(f"{result.get('id')}: max_new_tokens must be positive")
    return result


def load_corpus(path: Path) -> list[dict[str, Any]]:
    cases = [expand_case(case) for case in read_jsonl(path)]
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise EvalError("every corpus case needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise EvalError("corpus case ids must be unique")
    return cases


def corpus_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def display_path(path: Path | str) -> str:
    """Путь для run-header: дом — через `$HOME`. Артефакты уходят в публичный
    репозиторий, и имя пользователя в каждом из них не нужно. Движок при этом
    получает настоящий путь — подменяется только записываемая строка."""
    resolved = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    return "$HOME" if str(relative) == "." else f"$HOME/{relative.as_posix()}"


def run_header(
    engine: str,
    version: str,
    model: Path,
    corpus: Path,
    top_k: int,
    logits: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "record_type": "run",
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version,
        "model": display_path(model),
        "corpus_sha256": corpus_digest(corpus),
        "top_k": top_k,
        "logits": logits,
        **extra,
    }


def write_artifact(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
    temporary.replace(path)


def top_from_tensor(logits: Any, top_k: int) -> list[dict[str, Any]]:
    """Convert one torch logits row to the common top-token schema."""
    import torch

    values, ids = torch.topk(logits.float(), k=top_k)
    logsumexp = torch.logsumexp(logits.float(), dim=-1)
    return [
        {
            "token_id": int(token_id),
            "logit": float(logit),
            "logprob": float(logit - logsumexp),
            "rank": rank,
        }
        for rank, (token_id, logit) in enumerate(zip(ids.tolist(), values.tolist()), 1)
    ]


def run_qwc(args: argparse.Namespace) -> None:
    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "-p",
        "qwc-engine",
        "--bin",
        "eval",
        "--",
        "--model",
        str(args.model),
        "--corpus",
        str(args.corpus),
        "--top-k",
        str(args.top_k),
        "--batch",
        str(args.batch),
        "--embedding",
        args.embedding,
        "--lm-head",
        args.lm_head,
        "--kv-cache",
        args.qwc_kv_cache_dtype,
        "--decode-linear",
        args.qwc_decode_linear,
        "--delta-state",
        args.qwc_delta_state,
    ]
    if args.context:
        command.extend(("--context", str(args.context)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, check=False)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise EvalError(f"qwc runner exited with status {result.returncode}")
    records = read_jsonl(temporary)
    if not records or records[0].get("record_type") != "run":
        temporary.unlink(missing_ok=True)
        raise EvalError("qwc runner did not produce a valid artifact")
    records[0]["corpus_sha256"] = corpus_digest(args.corpus)
    modes = []
    if args.embedding != "fp8":
        modes.append(f"{args.embedding}-embedding")
    if args.lm_head != "fp8":
        modes.append(f"{args.lm_head}-head")
    if args.qwc_kv_cache_dtype != "fp8":
        modes.append(f"{args.qwc_kv_cache_dtype}-kv")
    if args.qwc_decode_linear != "auto":
        modes.append(f"{args.qwc_decode_linear}-decode")
    if args.qwc_delta_state != "bf16":
        modes.append(f"{args.qwc_delta_state}-delta-state")
    suffix = f"-{'-'.join(modes)}" if modes else ""
    records[0]["engine"] = args.name or f"qwc-b{args.batch}{suffix}"
    write_artifact(args.output, records)
    temporary.unlink(missing_ok=True)


def vllm_top_tokens(position: dict[int, Any]) -> list[dict[str, Any]]:
    items = []
    for token_id, value in position.items():
        items.append(
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": int(value.rank) if value.rank is not None else None,
            }
        )
    items.sort(key=lambda item: (item["rank"] is None, item["rank"] or 10**9, item["token_id"]))
    return items


def run_vllm(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("vllm") is None:
        raise EvalError("vLLM is not installed in this Python environment")
    # EngineCore uses multiprocessing spawn. Keep venv executables such as
    # ninja visible in the child even when this script was invoked by an
    # absolute Python path without activating the environment.
    # Do not resolve the Python symlink: its lexical parent is the venv bin,
    # while the target lives in uv's shared interpreter directory.
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = os.pathsep.join((venv_bin, os.environ.get("PATH", "")))
    # Greedy eval does not need FlashInfer's stochastic sampler. Its first-run
    # JIT additionally requires a complete, version-matched CUDA SDK, which a
    # deployment host is not guaranteed to have.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    cases = load_corpus(args.corpus)
    max_context = args.context or max(
        len(case["prompt_token_ids"]) + case["max_new_tokens"] for case in cases
    )
    started = time.perf_counter()
    llm_options = dict(
        model=str(args.model),
        max_model_len=max_context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        disable_log_stats=True,
        max_logprobs=max(args.top_k, 20),
    )
    if args.vllm_kv_cache_dtype != "auto":
        llm_options["kv_cache_dtype"] = args.vllm_kv_cache_dtype
    llm = LLM(**llm_options)
    load_seconds = time.perf_counter() - started
    records = [
        run_header(
            args.name
            or (
                "vllm"
                if args.vllm_kv_cache_dtype == "auto"
                else f"vllm-{args.vllm_kv_cache_dtype}-kv"
            ),
            package_version("vllm"),
            args.model,
            args.corpus,
            args.top_k,
            "raw_logprobs_top_k",
            load_seconds=load_seconds,
            kv_cache_dtype=args.vllm_kv_cache_dtype,
        )
    ]
    for case in cases:
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=case["max_new_tokens"],
            ignore_eos=True,
            detokenize=False,
            logprobs=args.top_k,
        )
        started = time.perf_counter()
        output = llm.generate(
            [{"prompt_token_ids": case["prompt_token_ids"]}], params, use_tqdm=False
        )[0].outputs[0]
        elapsed_ms = (time.perf_counter() - started) * 1e3
        positions = output.logprobs or []
        if len(positions) != len(output.token_ids):
            raise EvalError(
                f"{case['id']}: vLLM returned {len(output.token_ids)} tokens but "
                f"{len(positions)} logprob positions"
            )
        steps = [
            {
                "index": index,
                "token_id": int(token_id),
                "top_tokens": vllm_top_tokens(position),
            }
            for index, (token_id, position) in enumerate(zip(output.token_ids, positions))
        ]
        records.append(case_record(case, steps, elapsed_ms=elapsed_ms))
    write_artifact(args.output, records)


def model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def run_transformers(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("transformers") is None:
        raise EvalError("Transformers is not installed in this Python environment")
    import torch
    from transformers import AutoModelForCausalLM

    cases = load_corpus(args.corpus)
    load_options: dict[str, Any] = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if importlib.util.find_spec("accelerate") is not None:
        load_options["device_map"] = {"": "cuda"}
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_options)
    if "device_map" not in load_options:
        model = model.cuda()
    model.eval()
    load_seconds = time.perf_counter() - started
    records = [
        run_header(
            args.name or "transformers",
            package_version("transformers"),
            args.model,
            args.corpus,
            args.top_k,
            "raw_full_vocab",
            load_seconds=load_seconds,
            torch_version=torch.__version__,
        )
    ]
    device = model_device(model)
    with torch.inference_mode():
        for case in cases:
            started = time.perf_counter()
            input_ids = torch.tensor([case["prompt_token_ids"]], device=device)
            outputs = model(input_ids=input_ids, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[0, -1]
            steps = []
            for index in range(case["max_new_tokens"]):
                top_tokens = top_from_tensor(logits, args.top_k)
                token_id = top_tokens[0]["token_id"]
                steps.append(
                    {"index": index, "token_id": token_id, "top_tokens": top_tokens}
                )
                if index + 1 < case["max_new_tokens"]:
                    next_id = torch.tensor([[token_id]], device=device)
                    outputs = model(input_ids=next_id, past_key_values=past, use_cache=True)
                    past = outputs.past_key_values
                    logits = outputs.logits[0, -1]
            torch.cuda.synchronize()
            records.append(
                case_record(case, steps, elapsed_ms=(time.perf_counter() - started) * 1e3)
            )
    write_artifact(args.output, records)


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        raise EvalError(f"HTTP {error.code} from {url}: {body[:1000]}") from error
    except URLError as error:
        raise EvalError(f"cannot reach {url}: {error}") from error


def normalize_score(item: Any, rank: int | None = None) -> dict[str, Any]:
    if isinstance(item, dict):
        token_id = item.get("token_id", item.get("id"))
        logprob = item.get("logprob")
        item_rank = item.get("rank", rank)
    else:
        logprob, token_id = item[:2]
        item_rank = rank
    return {
        "token_id": int(token_id),
        "logprob": float(logprob),
        "rank": int(item_rank) if item_rank is not None else None,
    }


def run_sglang(args: argparse.Namespace) -> None:
    cases = load_corpus(args.corpus)
    records = [
        run_header(
            args.name or "sglang",
            server_version(args.url, args.timeout),
            args.model,
            args.corpus,
            args.top_k,
            "raw_logprobs_top_k",
            url=args.url,
        )
    ]
    for case in cases:
        payload = {
            "input_ids": case["prompt_token_ids"],
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": True,
                "max_new_tokens": case["max_new_tokens"],
            },
            "return_logprob": True,
            "top_logprobs_num": args.top_k,
            "return_text_in_logprobs": False,
        }
        started = time.perf_counter()
        result = http_json(args.url.rstrip("/") + "/generate", payload, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        meta = result.get("meta_info", {})
        chosen = meta.get("output_token_logprobs", [])
        top = meta.get("output_top_logprobs", [])
        if len(chosen) != len(top):
            raise EvalError(
                f"{case['id']}: SGLang returned {len(chosen)} selected and {len(top)} top-k rows"
            )
        steps = []
        for index, (selected, row) in enumerate(zip(chosen, top)):
            selected_score = normalize_score(selected)
            scores = [normalize_score(item, rank + 1) for rank, item in enumerate(row)]
            if all(item["token_id"] != selected_score["token_id"] for item in scores):
                scores.append(selected_score)
            scores.sort(key=lambda item: (-(item["logprob"]), item["token_id"]))
            for rank, item in enumerate(scores, 1):
                item["rank"] = rank
            steps.append(
                {
                    "index": index,
                    "token_id": selected_score["token_id"],
                    "top_tokens": scores,
                }
            )
        records.append(case_record(case, steps, elapsed_ms=elapsed_ms))
    write_artifact(args.output, records)


def run_llamacpp(args: argparse.Namespace) -> None:
    cases = load_corpus(args.corpus)
    records = [
        run_header(
            args.name or "llama.cpp",
            server_version(args.url, args.timeout),
            args.model,
            args.corpus,
            args.top_k,
            "raw_logprobs_top_k",
            url=args.url,
            note="server must use a tokenizer-compatible GGUF conversion",
        )
    ]
    for case in cases:
        payload = {
            "prompt": case["prompt_token_ids"],
            "n_predict": case["max_new_tokens"],
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "ignore_eos": True,
            "seed": 0,
            "n_probs": args.top_k,
            "return_tokens": True,
            "cache_prompt": False,
        }
        started = time.perf_counter()
        result = http_json(args.url.rstrip("/") + "/completion", payload, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        rows = result.get("completion_probabilities", [])
        steps = []
        for index, row in enumerate(rows):
            token_id = int(row["id"])
            scores = [normalize_score(item, rank + 1) for rank, item in enumerate(row["top_logprobs"])]
            if all(item["token_id"] != token_id for item in scores):
                scores.append(
                    {"token_id": token_id, "logprob": float(row["logprob"]), "rank": None}
                )
            steps.append({"index": index, "token_id": token_id, "top_tokens": scores})
        expected = case["max_new_tokens"]
        if len(steps) != expected:
            raise EvalError(
                f"{case['id']}: llama.cpp returned {len(steps)} probability rows, expected {expected}"
            )
        records.append(case_record(case, steps, elapsed_ms=elapsed_ms))
    write_artifact(args.output, records)


def run_tgi(args: argparse.Namespace) -> None:
    """TGI custom API; exact IDs are verified before sending prompt text."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    cases = load_corpus(args.corpus)
    records = [
        run_header(
            args.name or "tgi",
            server_version(args.url, args.timeout),
            args.model,
            args.corpus,
            args.top_k,
            "raw_logprobs_top_k",
            url=args.url,
        )
    ]
    for case in cases:
        prompt = case.get("prompt")
        if prompt is None or tokenizer.encode(prompt, add_special_tokens=False) != case["prompt_token_ids"]:
            records.append(case_record(case, [], status="unsupported_exact_token_input"))
            continue
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens": case["max_new_tokens"],
                "repetition_penalty": 1.0,
                "return_full_text": False,
                "details": True,
                "top_n_tokens": args.top_k,
                "stop": [],
            },
        }
        started = time.perf_counter()
        result = http_json(args.url.rstrip("/") + "/generate", payload, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        details = result.get("details", {})
        tokens = details.get("tokens", [])
        tops = details.get("top_tokens", [])
        steps = []
        for index, selected in enumerate(tokens):
            row = tops[index] if index < len(tops) else []
            scores = [normalize_score(item, rank + 1) for rank, item in enumerate(row)]
            steps.append(
                {"index": index, "token_id": int(selected["id"]), "top_tokens": scores}
            )
        records.append(case_record(case, steps, elapsed_ms=elapsed_ms))
    write_artifact(args.output, records)


def token_text_id(tokenizer: Any, text: str) -> int | None:
    """Recover an ID only when a server token round-trips unambiguously."""
    # vLLM and qwc serve spell tokens as "token_id:N" when asked to.
    if text.startswith("token_id:") and text[len("token_id:"):].isdigit():
        return int(text[len("token_id:"):])
    token_id = tokenizer.convert_tokens_to_ids(text)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unknown:
        if tokenizer.convert_ids_to_tokens(token_id) == text:
            return int(token_id)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return int(encoded[0]) if len(encoded) == 1 else None


def parse_openai_logprobs(choice: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]] | None:
    logprobs = choice.get("logprobs")
    if not logprobs:
        return None
    steps = []
    if "content" in logprobs:
        for index, row in enumerate(logprobs["content"]):
            token_id = token_text_id(tokenizer, row["token"])
            if token_id is None:
                return None
            scores = []
            for rank, item in enumerate(row.get("top_logprobs", []), 1):
                item_id = token_text_id(tokenizer, item["token"])
                if item_id is None:
                    continue
                scores.append(
                    {"token_id": item_id, "logprob": float(item["logprob"]), "rank": rank}
                )
            if all(item["token_id"] != token_id for item in scores):
                scores.append(
                    {"token_id": token_id, "logprob": float(row["logprob"]), "rank": None}
                )
            steps.append({"index": index, "token_id": token_id, "top_tokens": scores})
        return steps

    tokens = logprobs.get("tokens", [])
    selected_logprobs = logprobs.get("token_logprobs", [])
    top_rows = logprobs.get("top_logprobs", [])
    for index, text in enumerate(tokens):
        token_id = token_text_id(tokenizer, text)
        if token_id is None:
            return None
        row = top_rows[index] if index < len(top_rows) and top_rows[index] else {}
        scores = []
        for rank, (top_text, value) in enumerate(
            sorted(row.items(), key=lambda item: (-item[1], item[0])), 1
        ):
            item_id = token_text_id(tokenizer, top_text)
            if item_id is not None:
                scores.append({"token_id": item_id, "logprob": float(value), "rank": rank})
        if all(item["token_id"] != token_id for item in scores):
            selected = selected_logprobs[index] if index < len(selected_logprobs) else None
            if selected is not None:
                scores.append({"token_id": token_id, "logprob": float(selected), "rank": None})
        steps.append({"index": index, "token_id": token_id, "top_tokens": scores})
    return steps


def run_openai(args: argparse.Namespace) -> None:
    """Fallback for TensorRT-LLM, LMDeploy, MLC, Ollama, and similar servers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    cases = load_corpus(args.corpus)
    records = [
        run_header(
            args.name or "openai-compatible",
            server_version(args.url, args.timeout),
            args.model,
            args.corpus,
            args.top_k,
            "openai_logprobs_top_k",
            url=args.url,
            note="token IDs accepted only when returned token strings round-trip uniquely",
        )
    ]
    for case in cases:
        payload = {
            "model": args.served_model_name or str(args.model),
            "prompt": case["prompt_token_ids"],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": case["max_new_tokens"],
            "logprobs": args.top_k,
            "stream": False,
            "seed": 0,
        }
        if args.openai_token_ids:
            # vLLM extensions: exact IDs instead of text round-trips, and
            # exactly max_tokens steps like the reference adapters.
            payload["return_tokens_as_token_ids"] = True
            payload["ignore_eos"] = True
        started = time.perf_counter()
        result = http_json(args.url.rstrip("/") + "/v1/completions", payload, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        choices = result.get("choices", [])
        if not choices:
            raise EvalError(f"{case['id']}: OpenAI-compatible server returned no choices")
        steps = parse_openai_logprobs(choices[0], tokenizer)
        if steps is None:
            records.append(case_record(case, [], status="unsupported_exact_token_output"))
        else:
            records.append(case_record(case, steps, elapsed_ms=elapsed_ms))
    write_artifact(args.output, records)


def server_version(url: str, timeout: float) -> str:
    for endpoint in ("/get_server_info", "/server_info", "/props"):
        try:
            request = Request(url.rstrip("/") + endpoint, method="GET")
            with urlopen(request, timeout=min(timeout, 5.0)) as response:
                data = json.load(response)
            for key in ("version", "build_info", "system_fingerprint"):
                if key in data:
                    return str(data[key])
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
    return "unknown"


def case_record(
    case: dict[str, Any],
    steps: list[dict[str, Any]],
    status: str = "ok",
    **timing: float,
) -> dict[str, Any]:
    return {
        "record_type": "case",
        "case_id": case["id"],
        "prompt": case.get("prompt"),
        "prompt_token_ids": case["prompt_token_ids"],
        "tags": case.get("tags", []),
        "status": status,
        "steps": steps,
        "timing": timing,
    }


def load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    records = read_jsonl(path)
    if not records or records[0].get("record_type") != "run":
        raise EvalError(f"{path}: first record is not a run header")
    if records[0].get("schema_version") != SCHEMA_VERSION:
        raise EvalError(f"{path}: unsupported schema version")
    cases = {
        record["case_id"]: record
        for record in records[1:]
        if record.get("record_type") == "case"
    }
    return records[0], cases


def score_case(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference.get("prompt_token_ids") != candidate.get("prompt_token_ids"):
        return {"status": "prompt_mismatch"}
    if reference.get("status") != "ok" or candidate.get("status") != "ok":
        return {"status": candidate.get("status", "error")}
    ref_steps = reference.get("steps", [])
    cand_steps = candidate.get("steps", [])
    total = min(len(ref_steps), len(cand_steps))
    exact_prefix = 0
    for left, right in zip(ref_steps, cand_steps):
        if left["token_id"] != right["token_id"]:
            break
        exact_prefix += 1
    same_context_steps = min(total, exact_prefix + (1 if exact_prefix < total else 0))
    top1_equal = 0
    jaccards: list[float] = []
    logprob_errors: list[float] = []
    for index in range(same_context_steps):
        left = ref_steps[index]
        right = cand_steps[index]
        top1_equal += left["token_id"] == right["token_id"]
        left_scores = {int(item["token_id"]): float(item["logprob"]) for item in left["top_tokens"]}
        right_scores = {int(item["token_id"]): float(item["logprob"]) for item in right["top_tokens"]}
        union = left_scores.keys() | right_scores.keys()
        intersection = left_scores.keys() & right_scores.keys()
        if union:
            jaccards.append(len(intersection) / len(union))
        logprob_errors.extend(abs(left_scores[token] - right_scores[token]) for token in intersection)
    return {
        "status": "ok",
        "reference_steps": len(ref_steps),
        "candidate_steps": len(cand_steps),
        "exact_prefix": exact_prefix,
        "same_context_steps": same_context_steps,
        "top1_equal": top1_equal,
        "topk_jaccard_sum": sum(jaccards),
        "topk_jaccard_count": len(jaccards),
        "logprob_abs_error_sum": sum(logprob_errors),
        "logprob_abs_error_count": len(logprob_errors),
        "logprob_abs_error_max": max(logprob_errors, default=None),
    }


def score_artifact(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    ref_header, ref_cases = load_artifact(reference_path)
    cand_header, cand_cases = load_artifact(candidate_path)
    details = {}
    totals = {
        "reference_steps": 0,
        "exact_prefix": 0,
        "same_context_steps": 0,
        "top1_equal": 0,
        "topk_jaccard_sum": 0.0,
        "topk_jaccard_count": 0,
        "logprob_abs_error_sum": 0.0,
        "logprob_abs_error_count": 0,
        "logprob_abs_error_max": None,
    }
    for case_id, reference in ref_cases.items():
        candidate = cand_cases.get(case_id)
        result = {"status": "missing"} if candidate is None else score_case(reference, candidate)
        details[case_id] = result
        if result["status"] != "ok":
            continue
        for key in totals:
            value = result.get(key)
            if value is None:
                continue
            if key == "logprob_abs_error_max":
                totals[key] = value if totals[key] is None else max(totals[key], value)
            else:
                totals[key] += value
    reference_steps = totals["reference_steps"]
    same_context = totals["same_context_steps"]
    aggregate = {
        **totals,
        "exact_prefix_ratio": totals["exact_prefix"] / reference_steps if reference_steps else None,
        "top1_agreement": totals["top1_equal"] / same_context if same_context else None,
        "topk_jaccard": (
            totals["topk_jaccard_sum"] / totals["topk_jaccard_count"]
            if totals["topk_jaccard_count"]
            else None
        ),
        "logprob_mae": (
            totals["logprob_abs_error_sum"] / totals["logprob_abs_error_count"]
            if totals["logprob_abs_error_count"]
            else None
        ),
        "cases_ok": sum(result["status"] == "ok" for result in details.values()),
        "cases_total": len(details),
    }
    return {
        "reference": ref_header,
        "candidate": cand_header,
        "aggregate": aggregate,
        "cases": details,
    }


def metric(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(scores: list[dict[str, Any]], thresholds: dict[str, float]) -> str:
    if not scores:
        raise EvalError("no candidates to compare")
    reference = scores[0]["reference"]
    lines = [
        "# Differential eval",
        "",
        f"Reference: `{reference['engine']}` {reference.get('engine_version', '')}",
        "",
        "Distribution metrics stop after the first greedy divergence, because later logits use different contexts.",
        "",
        "| Candidate | Cases | Exact prefix | Top-1 | Top-k Jaccard | Logprob MAE | Max Δ | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for score in scores:
        aggregate = score["aggregate"]
        passed = (
            aggregate["top1_agreement"] is not None
            and aggregate["top1_agreement"] >= thresholds["min_top1"]
            and aggregate["topk_jaccard"] is not None
            and aggregate["topk_jaccard"] >= thresholds["min_jaccard"]
            and aggregate["logprob_mae"] is not None
            and aggregate["logprob_mae"] <= thresholds["max_logprob_mae"]
        )
        aggregate["passed"] = passed
        lines.append(
            "| {name} | {ok}/{total} | {prefix} | {top1} | {jaccard} | {mae} | {maximum} | {verdict} |".format(
                name=score["candidate"].get("name", score["candidate"]["engine"]),
                ok=aggregate["cases_ok"],
                total=aggregate["cases_total"],
                prefix=metric(aggregate["exact_prefix_ratio"]),
                top1=metric(aggregate["top1_agreement"]),
                jaccard=metric(aggregate["topk_jaccard"]),
                mae=metric(aggregate["logprob_mae"]),
                maximum=metric(aggregate["logprob_abs_error_max"]),
                verdict="PASS" if passed else "FAIL",
            )
        )
    lines.extend(
        (
            "",
            "Thresholds: "
            f"top-1 ≥ {thresholds['min_top1']:.3f}, "
            f"top-k Jaccard ≥ {thresholds['min_jaccard']:.3f}, "
            f"logprob MAE ≤ {thresholds['max_logprob_mae']:.3f}.",
            "",
        )
    )
    for score in scores:
        candidate_name = score["candidate"].get("name", score["candidate"]["engine"])
        lines.extend((f"## {candidate_name}", ""))
        lines.extend(("| Case | Prefix | Same-context steps | Status |", "|---|---:|---:|---|"))
        for case_id, result in score["cases"].items():
            lines.append(
                f"| {case_id} | {result.get('exact_prefix', '—')} | "
                f"{result.get('same_context_steps', '—')} | {result['status']} |"
            )
        lines.append("")
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> None:
    scores = [score_artifact(args.reference, candidate) for candidate in args.candidates]
    thresholds = {
        "min_top1": args.min_top1,
        "min_jaccard": args.min_jaccard,
        "max_logprob_mae": args.max_logprob_mae,
    }
    report = render_report(scores, thresholds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_threshold and not all(score["aggregate"].get("passed") for score in scores):
        raise SystemExit(1)


def validate_corpus(args: argparse.Namespace) -> None:
    cases = load_corpus(args.corpus)
    vocab_size = None
    config_path = args.model / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab_size = config.get("text_config", config).get("vocab_size")
    tokenizer = None
    if importlib.util.find_spec("transformers") is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    errors = []
    for case in cases:
        tokens = case["prompt_token_ids"]
        if vocab_size and any(token < 0 or token >= vocab_size for token in tokens):
            errors.append(f"{case['id']}: token outside vocabulary")
        if case.get("prompt") is not None and tokenizer is not None:
            actual = tokenizer.encode(case["prompt"], add_special_tokens=False)
            if actual != tokens:
                errors.append(f"{case['id']}: frozen IDs differ from AutoTokenizer")
    if errors:
        raise EvalError("\n".join(errors))
    print(
        f"OK: {len(cases)} cases, max prompt {max(len(case['prompt_token_ids']) for case in cases)} "
        f"tokens, sha256={corpus_digest(args.corpus)}"
    )


def probe(args: argparse.Namespace) -> None:
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    architecture = config.get("architectures", ["unknown"])[0]
    quantization = config.get("quantization_config", {}).get("format", "none")
    checks = [
        ("qwc", shutil.which("cargo"), "native safetensors + NVFP4"),
        ("transformers", importlib.util.find_spec("transformers"), "direct checkpoint"),
        ("vllm", importlib.util.find_spec("vllm"), "direct checkpoint"),
        ("sglang", importlib.util.find_spec("sglang"), "direct checkpoint or HTTP"),
        ("TensorRT-LLM", importlib.util.find_spec("tensorrt_llm"), "OpenAI HTTP adapter"),
        ("LMDeploy", importlib.util.find_spec("lmdeploy"), "HTTP adapter"),
        ("llama.cpp", shutil.which("llama-server"), "requires compatible GGUF"),
        ("TGI", shutil.which("text-generation-launcher"), "HTTP adapter, text cases only"),
        ("MLC LLM", shutil.which("mlc_llm"), "OpenAI adapter, converted model"),
        ("Ollama", shutil.which("ollama"), "OpenAI adapter, converted model"),
        ("ExLlamaV2", importlib.util.find_spec("exllamav2"), "no NVFP4 adapter"),
    ]
    print(f"model: {architecture}, quantization: {quantization}")
    print("| Engine | Local | Route |")
    print("|---|---|---|")
    for name, found, route in checks:
        print(f"| {name} | {'yes' if found else 'no'} | {route} |")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    common.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-corpus", parents=[common])
    validate.set_defaults(func=validate_corpus)
    inspect = sub.add_parser("probe", parents=[common])
    inspect.set_defaults(func=probe)

    run = sub.add_parser("run", parents=[common])
    run.add_argument(
        "--engine",
        required=True,
        choices=("qwc", "transformers", "vllm", "sglang", "llamacpp", "tgi", "openai"),
    )
    run.add_argument("--name")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--batch", type=int, default=1, help="qwc execution batch")
    run.add_argument(
        "--embedding",
        choices=("fp8", "bf16"),
        default="fp8",
        help="qwc embedding storage; bf16 is an A/B diagnostic",
    )
    run.add_argument(
        "--lm-head",
        choices=("fp8", "bf16"),
        default="fp8",
        help="qwc lm_head storage; bf16 is an A/B diagnostic",
    )
    run.add_argument(
        "--qwc-kv-cache-dtype",
        choices=("fp8", "bf16"),
        default="fp8",
        help="qwc full-attention KV storage",
    )
    run.add_argument(
        "--qwc-decode-linear",
        choices=("auto", "w4a4"),
        default="auto",
        help="force qwc decode projections to the W4A4 tensor-core path",
    )
    run.add_argument(
        "--qwc-delta-state",
        choices=("wy", "bf16", "fp32"),
        default="wy",
        help="prefill chunk scan: recurrent state precision (bf16/fp32) or the "
        "matrix WY form (wy)",
    )
    run.add_argument("--context", type=int)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    run.add_argument(
        "--vllm-kv-cache-dtype",
        choices=("auto", "fp8"),
        default="auto",
        help="vLLM full-attention KV dtype; mamba state dtype is unchanged",
    )
    run.add_argument("--eager", action="store_true", help="disable vLLM CUDA graphs")
    run.add_argument("--url", help="base URL for an HTTP backend")
    run.add_argument("--served-model-name", help="model field sent to an OpenAI server")
    run.add_argument(
        "--openai-token-ids",
        action="store_true",
        help="send return_tokens_as_token_ids and ignore_eos (vLLM, qwc serve)",
    )
    run.add_argument("--timeout", type=float, default=600.0)
    run.set_defaults(func=dispatch_run)

    comparison = sub.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("candidates", type=Path, nargs="+")
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--json-output", type=Path)
    comparison.add_argument("--min-top1", type=float, default=0.95)
    comparison.add_argument("--min-jaccard", type=float, default=0.70)
    comparison.add_argument("--max-logprob-mae", type=float, default=0.25)
    comparison.add_argument("--fail-on-threshold", action="store_true")
    comparison.set_defaults(func=compare)
    return result


def dispatch_run(args: argparse.Namespace) -> None:
    if not 1 <= args.top_k <= 1000:
        raise EvalError("--top-k must be in 1..=1000")
    runners = {
        "qwc": run_qwc,
        "transformers": run_transformers,
        "vllm": run_vllm,
        "sglang": run_sglang,
        "llamacpp": run_llamacpp,
        "tgi": run_tgi,
        "openai": run_openai,
    }
    if args.engine in {"sglang", "llamacpp", "tgi", "openai"} and not args.url:
        raise EvalError(f"--url is required for {args.engine}")
    runners[args.engine](args)


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except EvalError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
