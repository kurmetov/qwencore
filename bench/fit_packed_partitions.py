#!/usr/bin/env python3
"""Подгонка выбора партиций упакованного ядра по плотному свипу packedbench.

    cargo run --release -p qwc-cuda --bin packedbench -- --sweep --dense \
        --rows 32,48,64,96,128,192,256,384,512 > sweep.txt
    bench/fit_packed_partitions.py sweep.txt

Сравнивает прежнее правило «3.5-6 волн» и волновую модель
«волны x (отрезок + k0 ключей)» с сеткой k0 и цены редукции. Промах —
насколько время при выбранном P хуже лучшего замеренного. Модель в
`packed_partition_count` — k0 = 64, без цены редукции. Данные 2026-09-24 —
`bench/results/packed-dense-sweep-2026-09-24.txt` (частичные суммы в fp32,
потолок 2048 пар) и `packed-dense-sweep-f16-2026-09-24.txt` (f16, 4096 пар);
потолок задаёт `--max-pairs`.
"""
import math
import re
import sys

SMS, KV, GROUP, PAGE = 170, 4, 6, 64
# Потолок пар (строка, партиция), PACKED_MAX_PARTIAL_ROWS.
MAX_PAIRS = 4096


def parse(path):
    table = {}
    header = None
    for line in open(path):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 5 and cells[0] == "строк":
            header = [int(c[2:]) for c in cells[4:]]
            continue
        if header and len(cells) == len(header) + 4 and cells[0].isdigit():
            rows, start = int(cells[0]), int(cells[1])
            times = {}
            for p, cell in zip(header, cells[4:]):
                if cell not in ("-", ""):
                    times[p] = float(cell)
            table[(rows, start)] = times
    return table


def tiles(rows):
    return math.ceil(rows * GROUP / 64)


def used(context, partitions):
    span = math.ceil(context / (partitions * PAGE)) * PAGE
    return math.ceil(context / span), span


def wave_rule(rows, context, k0, reduce_cost, max_partitions=48):
    ctas = KV * tiles(rows)
    best = None
    for p in range(1, max_partitions + 1):
        if rows * p > MAX_PAIRS and p > 1:
            break
        n, span = used(context, p)
        cost = math.ceil(ctas * n / SMS) * (span + k0) + (reduce_cost * rows * p if p > 1 else 0)
        if best is None or cost < best[0]:
            best = (cost, p)
    return best[1]


def previous_rule(rows, context):
    """Правило до d181413: 3.5-6 волн с самой полной последней."""
    ctas = KV * tiles(rows)
    if SMS // ctas >= 16:
        wanted = SMS // ctas
    else:
        lowest = max(1, math.ceil(7 * SMS / (2 * ctas)))
        highest = max(6 * SMS // ctas, lowest)
        def key(p):
            total = ctas * p
            return (total * 1000 // (math.ceil(total / SMS) * SMS), -p)
        wanted = max(range(lowest, highest + 1), key=key)
    return max(1, min(wanted, math.ceil(context / 256), 2048 // rows))


def regret(table, rule):
    worst, total, count, details = 0.0, 0.0, 0, []
    for (rows, start), times in sorted(table.items()):
        context = start + rows
        p = rule(rows, context)
        best_p = min(times, key=times.get)
        if p not in times:
            p = min(times, key=lambda q: abs(q - p))
        loss = times[p] / times[best_p] - 1
        worst = max(worst, loss)
        total += loss
        count += 1
        details.append((rows, start, p, times[p], best_p, times[best_p], loss))
    return total / count, worst, details


if __name__ == "__main__":
    if "--max-pairs" in sys.argv:
        at = sys.argv.index("--max-pairs")
        MAX_PAIRS = int(sys.argv.pop(at + 1))
        sys.argv.pop(at)
    table = parse(sys.argv[1])
    print(f"форм: {len(table)}")
    mean, worst, details = regret(table, previous_rule)
    print(f"прежнее правило: среднее {mean:.3f}, худшее {worst:.3f}")
    results = []
    for k0 in (0, 32, 64, 128, 192, 256, 384, 512, 768):
        for reduce_cost in (0, 0.5, 1, 2, 4, 8):
            mean, worst, _ = regret(table, lambda r, c: wave_rule(r, c, k0, reduce_cost))
            results.append((mean, worst, k0, reduce_cost))
    results.sort()
    for mean, worst, k0, reduce_cost in results[:8]:
        print(f"волны: k0={k0:4d} reduce={reduce_cost:4.1f}: среднее {mean:.3f}, худшее {worst:.3f}")
    mean, worst, k0, reduce_cost = results[0]
    print(f"\nлучшее: k0={k0}, reduce={reduce_cost}")
    for rows, start, p, t, best_p, best_t, loss in regret(
            table, lambda r, c: wave_rule(r, c, k0, reduce_cost))[2]:
        mark = "  <--" if loss > 0.05 else ""
        cur = previous_rule(rows, start + rows)
        print(f"  {rows:4d} x {start:6d}: P={p:2d} {t:8.1f}  лучшее P={best_p:2d} {best_t:8.1f}"
              f"  +{loss * 100:4.1f}%  (прежнее P={cur}: {table[(rows, start)].get(cur, float('nan')):.1f}){mark}")
