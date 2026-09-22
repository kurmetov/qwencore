#!/usr/bin/env python3
"""Reproducible QwenCore experiments with raw trials and confidence intervals.

The script deliberately launches a fresh process for every trial.  This is
slower than timing the same loaded engine repeatedly, but makes the interval
include model startup/capture state, allocator state and run-to-run GPU drift.
Variants are rotated between trials so a slow thermal period does not always
belong to the same engine.

Examples:

    # Inspect the exact commands without touching the GPU.
    python3 bench/experiments.py run --suite comparison --dry-run

    # Run seven independent trials of the fair c=1 comparisons.
    ~/.venvs/baseline/bin/python bench/experiments.py run \
      --suite comparison --scenario decode-c1 --scenario decode-c1-mtp \
      --repeats 7 --output bench/results/comparison-trials.jsonl

    # Turn raw trials into a Markdown report with bootstrap 95% CIs.
    python3 bench/experiments.py report \
      --input bench/results/comparison-trials.jsonl \
      --output bench/results/comparison-with-ci.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
SERVEBENCH = ROOT / "bench" / "servebench.py"
DEFAULT_CORPUS = ROOT / "bench" / "corpus" / "serve.jsonl"
LONG_CORPUS = ROOT / "bench" / "corpus" / "serve-4096.jsonl"
DEFAULT_MODEL = Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4"
METRICS = (
    "output_tokens_per_second",
    "ttft_ms_p50",
    "itl_ms_p50",
)


@dataclass(frozen=True)
class Variant:
    name: str
    engine: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    kind: str
    common: tuple[str, ...]
    variants: tuple[Variant, ...]


def _base(
    *, requests: int, concurrency: int, prompt: int, output: int,
    context: int, budget: int, corpus: Path, kv: str = "fp8",
) -> tuple[str, ...]:
    return (
        "--requests", str(requests),
        "--concurrency", str(concurrency),
        "--prompt-tokens", str(prompt),
        "--max-new", str(output),
        "--context", str(context),
        "--engine-budget-gb", str(budget),
        "--kv-cache-dtype", kv,
        "--corpus", str(corpus),
    )


def _without_option(args: tuple[str, ...], option: str) -> tuple[str, ...]:
    """Remove one ``--option value`` pair from a flat argparse sequence."""
    output: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == option:
            index += 2
        else:
            output.append(args[index])
            index += 1
    return tuple(output)


def suites() -> dict[str, tuple[Scenario, ...]]:
    c1 = _base(
        requests=8, concurrency=1, prompt=256, output=128,
        context=2048, budget=26, corpus=DEFAULT_CORPUS,
    )
    c32 = _base(
        requests=160, concurrency=32, prompt=256, output=128,
        context=2048, budget=26, corpus=DEFAULT_CORPUS,
    )
    prefill = _base(
        requests=16, concurrency=16, prompt=4096, output=8,
        context=8192, budget=26, corpus=LONG_CORPUS, kv="bf16",
    )

    comparison = (
        Scenario(
            "decode-c1",
            "Single-stream decode without speculative decoding.",
            "comparison",
            c1,
            (Variant("qwc", "qwc"), Variant("vllm", "vllm")),
        ),
        Scenario(
            "decode-c1-mtp",
            "Single-stream decode with each engine's tuned MTP path.",
            "comparison",
            c1,
            (
                Variant(
                    "qwc-mtp-k3-shortlist32k", "qwc",
                    ("--qwc-speculative", "3", "--qwc-shortlist", "32768"),
                ),
                Variant("vllm-mtp-k3", "vllm", ("--vllm-speculative", "3")),
            ),
        ),
        Scenario(
            "serving-c32",
            "Saturated serving at equal 26 GB engine budget.",
            "comparison",
            c32,
            (
                Variant("qwc-chunk2048", "qwc", ("--prefill-chunk", "2048")),
                Variant("vllm-mbt2048", "vllm", ("--max-batched-tokens", "2048")),
            ),
        ),
        Scenario(
            "prefill-4k",
            "Long-prompt prefill after tuning each engine's token budget.",
            "comparison",
            prefill,
            (
                Variant("qwc-chunk2048", "qwc", ("--prefill-chunk", "2048")),
                Variant("vllm-mbt4096", "vllm", ("--max-batched-tokens", "4096")),
            ),
        ),
    )

    ablations = (
        Scenario(
            "ablate-speculation",
            "Cumulative b=1 ablation; every adjacent row adds one feature.",
            "ablation",
            c1,
            (
                Variant("base", "qwc"),
                Variant("+mtp-k3", "qwc", ("--qwc-speculative", "3")),
                Variant(
                    "+shortlist32k", "qwc",
                    ("--qwc-speculative", "3", "--qwc-shortlist", "32768"),
                ),
            ),
        ),
        Scenario(
            "ablate-prefill-scan",
            "DeltaNet prefill algorithm at fixed chunk and workload.",
            "ablation",
            prefill,
            (
                Variant("recurrent-bf16", "qwc", ("--delta-state", "bf16", "--prefill-chunk", "2048")),
                Variant("recurrent-fp32", "qwc", ("--delta-state", "fp32", "--prefill-chunk", "2048")),
                Variant("wy", "qwc", ("--delta-state", "wy", "--prefill-chunk", "2048")),
            ),
        ),
        Scenario(
            "ablate-prefill-chunk",
            "Prefill token budget with the WY algorithm held fixed.",
            "ablation",
            prefill,
            (
                Variant("chunk512", "qwc", ("--delta-state", "wy", "--prefill-chunk", "512")),
                Variant("chunk2048", "qwc", ("--delta-state", "wy", "--prefill-chunk", "2048")),
            ),
        ),
        Scenario(
            "ablate-kv",
            "KV storage format at fixed saturated workload.",
            "ablation",
            _without_option(c32, "--kv-cache-dtype"),
            (
                Variant("bf16-kv", "qwc", ("--kv-cache-dtype", "bf16")),
                Variant("fp8-kv", "qwc", ("--kv-cache-dtype", "fp8")),
            ),
        ),
    )
    return {"comparison": comparison, "ablations": ablations}


def command_for(
    scenario: Scenario, variant: Variant, model: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SERVEBENCH),
        "--engine", variant.engine,
    ]
    if model is not None:
        command.extend(("--model", str(model)))
    command.extend((*scenario.common, *variant.args))
    return command


def run_text(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return result.stdout.strip()


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


def git_metadata() -> dict[str, object]:
    commit = run_text(("git", "rev-parse", "HEAD")) or "unknown"
    status = run_text(("git", "status", "--porcelain"))
    dirty_lines = status.splitlines() if status else []
    diff = run_text(("git", "diff", "--binary")) or ""
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_files": len(dirty_lines),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "source_sha256": source_fingerprint(),
    }


def source_fingerprint() -> str:
    """Hash sources that determine benchmark behavior, including untracked edits."""
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for pattern in ("crates/**/*.rs", "crates/**/*.cu", "crates/**/*.cuh", "bench/*.py"):
        paths.extend(ROOT.glob(pattern))
    digest = hashlib.sha256()
    for path in sorted(set(path for path in paths if path.is_file())):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def gpu_metadata() -> dict[str, object]:
    query = (
        "name,memory.total,memory.used,utilization.gpu,power.draw,"
        "temperature.gpu,clocks.sm,clocks.mem"
    )
    text = run_text(("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"))
    if not text:
        return {"available": False}
    fields = [part.strip() for part in text.splitlines()[0].split(",")]
    keys = (
        "name", "memory_total_mib", "memory_used_mib", "gpu_util_percent",
        "power_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz",
    )
    record: dict[str, object] = {"available": True}
    for key, value in zip(keys, fields):
        if key == "name":
            record[key] = value
        else:
            try:
                record[key] = float(value)
            except ValueError:
                record[key] = value
    return record


def select_scenarios(suite: str, names: Sequence[str]) -> tuple[Scenario, ...]:
    available = suites()[suite]
    if not names:
        return available
    wanted = set(names)
    selected = tuple(item for item in available if item.name in wanted)
    missing = wanted - {item.name for item in selected}
    if missing:
        raise SystemExit(f"unknown scenarios for {suite}: {', '.join(sorted(missing))}")
    return selected


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("benchmark emitted no JSON object")


def run_suite(args: argparse.Namespace) -> int:
    scenarios = select_scenarios(args.suite, args.scenario)
    for scenario in scenarios:
        for variant in scenario.variants:
            missing = [value for value in scenario.common if value.endswith(".jsonl") and not Path(value).exists()]
            if missing:
                raise SystemExit(f"missing corpus: {missing[0]}")

    if args.dry_run:
        for scenario in scenarios:
            print(f"[{scenario.name}] {scenario.description}")
            for variant in scenario.variants:
                print(" ".join(command_for(scenario, variant, args.model)))
        return 0

    if args.repeats < 3:
        raise SystemExit("at least 3 repeats are required for an interval; use 7 or more")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} exists; pass --resume or choose another output")

    initial_gpu = gpu_metadata()
    used_mib = initial_gpu.get("memory_used_mib")
    limit_mib = args.max_background_memory_gb * 1024
    if (
        not args.allow_busy_gpu
        and isinstance(used_mib, float)
        and used_mib > limit_mib
    ):
        raise SystemExit(
            f"GPU already uses {used_mib / 1024:.1f} GiB; refusing a publishable run "
            f"above {args.max_background_memory_gb:.1f} GiB. Stop the competing workload "
            "or pass --allow-busy-gpu for an explicitly contaminated diagnostic run."
        )

    experiment = {
        "record_type": "experiment",
        "schema_version": 1,
        "suite": args.suite,
        "repeats": args.repeats,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.executable,
        "model": display_path(args.model),
        "git": git_metadata(),
        "gpu": initial_gpu,
        "max_background_memory_gb": args.max_background_memory_gb,
        "allow_busy_gpu": args.allow_busy_gpu,
        "scenarios": [item.name for item in scenarios],
    }
    if not args.resume or not args.output.exists():
        append_jsonl(args.output, experiment)

    existing: set[tuple[str, str, int]] = set()
    if args.resume and args.output.exists():
        for record in read_jsonl(args.output):
            if record.get("record_type") == "measurement" and record.get("status") == "ok":
                existing.add((str(record["scenario"]), str(record["variant"]), int(record["trial"])))

    for scenario in scenarios:
        count = len(scenario.variants)
        for trial in range(args.repeats):
            # Rotate the first engine every trial to counter thermal/time drift.
            order = [scenario.variants[(trial + offset) % count] for offset in range(count)]
            for run_order, variant in enumerate(order):
                key = (scenario.name, variant.name, trial)
                if key in existing:
                    continue
                command = command_for(scenario, variant, args.model)
                before = gpu_metadata()
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                result = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True, check=False,
                )
                record: dict[str, object] = {
                    "record_type": "measurement",
                    "schema_version": 1,
                    "suite": args.suite,
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "kind": scenario.kind,
                    "variant": variant.name,
                    "variant_index": scenario.variants.index(variant),
                    "engine": variant.engine,
                    "trial": trial,
                    "run_order": run_order,
                    "started_at": started,
                    "command": command,
                    "gpu_before": before,
                    "gpu_after": gpu_metadata(),
                }
                if result.returncode == 0:
                    try:
                        record["result"] = parse_last_json(result.stdout)
                        record["status"] = "ok"
                    except ValueError as error:
                        record["status"] = "error"
                        record["error"] = str(error)
                else:
                    record["status"] = "error"
                    record["returncode"] = result.returncode
                    record["stderr_tail"] = result.stderr[-4000:]
                    record["stdout_tail"] = result.stdout[-4000:]
                append_jsonl(args.output, record)
                print(
                    f"{scenario.name} trial={trial + 1}/{args.repeats} "
                    f"variant={variant.name}: {record['status']}",
                    flush=True,
                )
                if record["status"] != "ok" and not args.keep_going:
                    return 1
    return 0


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{number}: expected object")
            records.append(value)
    return records


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def bootstrap_mean_ci(
    values: Sequence[float], *, confidence: float = 0.95,
    samples: int = 20_000, seed: int = 0,
) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    center = statistics.fmean(values)
    if len(values) == 1:
        return center, center, center
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return center, percentile(means, alpha), percentile(means, 1.0 - alpha)


def bootstrap_ratio_ci(
    base: dict[int, float], candidate: dict[int, float], *,
    confidence: float = 0.95, samples: int = 20_000, seed: int = 0,
) -> tuple[float, float, float] | None:
    trials = sorted(base.keys() & candidate.keys())
    if not trials:
        return None
    base_values = [base[index] for index in trials]
    candidate_values = [candidate[index] for index in trials]
    center = statistics.fmean(candidate_values) / statistics.fmean(base_values)
    if len(trials) == 1:
        return center, center, center
    rng = random.Random(seed)
    ratios = []
    n = len(trials)
    for _ in range(samples):
        chosen = [rng.randrange(n) for _ in range(n)]
        denominator = statistics.fmean(base_values[index] for index in chosen)
        numerator = statistics.fmean(candidate_values[index] for index in chosen)
        ratios.append(numerator / denominator)
    alpha = (1.0 - confidence) / 2.0
    return center, percentile(ratios, alpha), percentile(ratios, 1.0 - alpha)


def format_ci(summary: tuple[float, float, float] | None, digits: int = 1) -> str:
    if summary is None or math.isnan(summary[0]):
        return "—"
    center, low, high = summary
    return f"{center:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def report(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    measurements = [
        item for item in records
        if item.get("record_type") == "measurement" and item.get("status") == "ok"
    ]
    if not measurements:
        raise SystemExit("no successful measurements")

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    scenario_meta: dict[str, dict[str, object]] = {}
    for item in measurements:
        scenario = str(item["scenario"])
        variant = str(item["variant"])
        groups.setdefault((scenario, variant), []).append(item)
        scenario_meta.setdefault(scenario, item)

    lines = [
        "# Benchmark report with confidence intervals",
        "",
        f"Source: `{args.input}`.",
        "",
        "Each point is an independent process run. Variants were rotated between trials. "
        "Intervals are deterministic percentile-bootstrap 95% confidence intervals of the mean "
        "(20,000 resamples); they describe run-to-run uncertainty on this machine, not model quality.",
        "",
        "| scenario | variant | n | output tok/s, mean [95% CI] | TTFT p50 ms | ITL p50 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]

    ordered_groups = sorted(
        groups.items(),
        key=lambda pair: (
            pair[0][0],
            min(int(item.get("variant_index", 0)) for item in pair[1]),
        ),
    )
    for (scenario, variant), items in ordered_groups:
        summaries = {}
        for metric in METRICS:
            values = [float(item["result"][metric]) for item in items if metric in item["result"]]
            summaries[metric] = bootstrap_mean_ci(values, seed=args.seed)
        lines.append(
            f"| {scenario} | {variant} | {len(items)} | "
            f"{format_ci(summaries['output_tokens_per_second'])} | "
            f"{format_ci(summaries['ttft_ms_p50'])} | "
            f"{format_ci(summaries['itl_ms_p50'], 2)} |"
        )

    lines.extend(("", "## Paired effects", ""))
    for scenario in sorted(scenario_meta):
        variants = sorted(
            ((variant, items) for (name, variant), items in groups.items() if name == scenario),
            key=lambda pair: min(int(item.get("variant_index", 0)) for item in pair[1]),
        )
        if len(variants) < 2:
            continue
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append(str(scenario_meta[scenario].get("description", "")))
        lines.append("")
        lines.append("| transition | throughput ratio [95% CI] | ITL ratio [95% CI] |")
        lines.append("|---|---:|---:|")
        for (before_name, before_items), (after_name, after_items) in zip(variants, variants[1:]):
            def by_trial(items: Iterable[dict[str, object]], metric: str) -> dict[int, float]:
                return {
                    int(item["trial"]): float(item["result"][metric])
                    for item in items if metric in item["result"]
                }

            throughput = bootstrap_ratio_ci(
                by_trial(before_items, "output_tokens_per_second"),
                by_trial(after_items, "output_tokens_per_second"),
                seed=args.seed,
            )
            itl = bootstrap_ratio_ci(
                by_trial(before_items, "itl_ms_p50"),
                by_trial(after_items, "itl_ms_p50"),
                seed=args.seed,
            )
            lines.append(
                f"| {before_name} → {after_name} | {format_ci(throughput, 3)} | {format_ci(itl, 3)} |"
            )
        lines.append("")

    failures = [item for item in records if item.get("record_type") == "measurement" and item.get("status") != "ok"]
    if failures:
        lines.extend((
            "## Failed trials",
            "",
            f"{len(failures)} failed trial(s) are present in the raw artifact and were excluded from intervals. "
            "A publishable run should explain or rerun them; do not silently drop failures.",
            "",
        ))

    text = "\n".join(lines).rstrip() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a predefined experiment suite")
    run_parser.add_argument("--suite", choices=tuple(suites()), required=True)
    run_parser.add_argument("--scenario", action="append", default=[], help="scenario name; repeat to select several")
    run_parser.add_argument("--repeats", type=int, default=7)
    run_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run_parser.add_argument("--output", type=Path, default=ROOT / "bench/results/experiment-trials.jsonl")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--keep-going", action="store_true")
    run_parser.add_argument("--max-background-memory-gb", type=float, default=8.0)
    run_parser.add_argument("--allow-busy-gpu", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=run_suite)

    report_parser = subparsers.add_parser("report", help="summarize raw JSONL trials")
    report_parser.add_argument("--input", type=Path, required=True)
    report_parser.add_argument("--output", type=Path)
    report_parser.add_argument("--seed", type=int, default=20260922)
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
