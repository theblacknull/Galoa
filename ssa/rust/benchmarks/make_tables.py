"""Markdown table: Rssa vs ssa-rs timings (rust/benchmarks/*_times*.json)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
r = {(x["case"], x["op"]): x["sec"] for x in json.loads((HERE / f"rssa_times{SUF}.json").read_text())}
s = {(x["case"], x["op"]): x["sec"] for x in json.loads((HERE / f"rust_times{SUF}.json").read_text())}


def ms(v):
    v *= 1000
    return f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:,.0f}")


print("| Bài toán | Thao tác | Rssa (ms) | ssa-rs (ms) | Nhanh hơn |")
print("|---|---|---:|---:|---:|")
last = None
for key in r:
    case, op = key
    if key not in s:
        continue
    print(f"| {case if case != last else ''} | {op} | {ms(r[key])} | {ms(s[key])} | {r[key] / s[key]:.1f}× |")
    last = case

tot = {}
for (case, op), v in r.items():
    if (case, op) in s:
        a = tot.setdefault(case, [0.0, 0.0])
        a[0] += v
        a[1] += s[(case, op)]
print("\n| Bài toán (tổng các thao tác) | Rssa (ms) | ssa-rs (ms) | Nhanh hơn |")
print("|---|---:|---:|---:|")
for case, (a, b) in tot.items():
    print(f"| {case} | {ms(a)} | {ms(b)} | {a / b:.1f}× |")
