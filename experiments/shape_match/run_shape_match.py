#!/usr/bin/env python3
"""Discrete shape matching on GALOAP1 Gold M1 — not SSA, not periodic.

Idea: when the recent price *path* looks like past paths (z-normalized Pearson),
trade the historical forward-outcome majority/mean side. Abstain if similarity < τ.

Metrics (only): dir_acc, pnl_pct, pnl_over_maxdd.

Example:
  python experiments/shape_match/run_shape_match.py \\
    --select data/xauusd_m1_15d.bin \\
    --oos data/xauusd_m1_prior15d.bin
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MAGIC = b"GALOAP1\0"


def load_galoa_closes(path: Path) -> np.ndarray:
    buf = path.read_bytes()
    if len(buf) < 16 or buf[:8] != MAGIC:
        raise ValueError(f"bad GALOAP1 magic: {path}")
    n = struct.unpack_from("<Q", buf, 8)[0]
    off = 16 + n * 8
    need = off + n * 8
    if len(buf) < need:
        raise ValueError(f"truncated GALOAP1: {path}")
    closes = np.frombuffer(buf, dtype="<f8", count=n, offset=off).astype(np.float64)
    return closes


def log_returns(closes: np.ndarray) -> np.ndarray:
    c0 = closes[:-1]
    c1 = closes[1:]
    out = np.full(c0.shape, np.nan, dtype=np.float64)
    ok = (c0 > 0.0) & (c1 > 0.0)
    out[ok] = np.log(c1[ok] / c0[ok])
    return out


def znorm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise z-score. Flat rows → zeros (will not match)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.maximum(sd, eps)
    Z = (X - mu) / sd
    flat = X.std(axis=1) < eps
    Z[flat] = 0.0
    return Z


def build_close_windows(closes: np.ndarray, L: int) -> np.ndarray:
    """Windows of raw closes; shape (n_win, L). Window i ends at close index i+L-1."""
    n = closes.shape[0]
    if n < L:
        return np.empty((0, L), dtype=np.float64)
    # stride tricks
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(closes, L).copy()


@dataclass
class Metrics:
    dir_acc: float
    pnl_pct: float
    max_dd_pct: float
    pnl_over_maxdd: float | None
    n_fire: int
    n_eval: int
    coverage: float


@dataclass
class Cfg:
    L: int
    H: int
    k: int
    tau: float
    stride: int


def eval_shape_match(
    closes: np.ndarray,
    cfg: Cfg,
    *,
    warm_frac: float = 0.2,
) -> Metrics:
    """Causal expanding library on one series. Query every `stride` bars after warm-up."""
    L, H, k, tau, stride = cfg.L, cfg.H, cfg.k, cfg.tau, cfg.stride
    rets = log_returns(closes)  # len n-1; rets[t] = ln(c[t+1]/c[t])
    wins = build_close_windows(closes, L)  # end_idx = i+L-1
    n_win = wins.shape[0]
    if n_win == 0:
        return Metrics(0.0, 0.0, 0.0, None, 0, 0, 0.0)

    Z = znorm_rows(wins)
    # Forward outcome for window ending at close index e = i+L-1:
    # next H log-returns starting at e: sum rets[e : e+H]  (direction of path)
    # For H=1: sign(rets[e]) = next bar after the shape ends.
    # Need e+H < len(closes)-? rets index e exists if e <= n-2; for sum need e+H-1 <= n-2
    # i.e. e <= n-1-H → i+L-1 <= n-1-H → i <= n-L-H
    max_i = n_win - H  # i <= n_win - H - ? window i ends at i+L-1; need i+L-1+H < n
    # closes index after H steps: end + H = i+L-1+H < n → i < n-L-H+1 = n_win-H
    max_i = n_win - H
    if max_i <= 0:
        return Metrics(0.0, 0.0, 0.0, None, 0, 0, 0.0)

    # Precompute forward sum returns and next close for PnL (H-step)
    # outcome_ret[i] = sum of rets[end : end+H] where end = i+L-1
    ends = np.arange(max_i) + (L - 1)
    fwd = np.zeros(max_i, dtype=np.float64)
    for h in range(H):
        fwd += rets[ends + h]
    # PnL uses close path: enter at closes[end], exit at closes[end+H]
    c_enter = closes[ends]
    c_exit = closes[ends + H]

    warm = max(L + H + k + 50, int(warm_frac * max_i))
    warm = min(warm, max_i - 1)
    if warm < k + 1:
        return Metrics(0.0, 0.0, 0.0, None, 0, 0, 0.0)

    scores: list[float] = []
    next_r: list[float] = []
    c0s: list[float] = []
    c1s: list[float] = []

    # Vectorized similarity: for query i, library = [0 .. i-H) so outcomes known before query
    # Gap of H avoids peeking into the forward window of neighbors that overlap query future.
    for i in range(warm, max_i, stride):
        lib_end = i - H
        if lib_end <= k:
            continue
        q = Z[i]
        if not np.any(q):
            continue
        # Pearson of z-rows ≈ (Z · q) / L
        sims = Z[:lib_end] @ q / L
        # top-k
        if lib_end <= k:
            idx = np.argsort(sims)[::-1]
        else:
            # argpartition for speed
            part = np.argpartition(sims, -k)[-k:]
            idx = part[np.argsort(sims[part])[::-1]]
        top_sims = sims[idx]
        mean_sim = float(top_sims.mean())
        if mean_sim < tau:
            continue  # abstain
        top_fwd = fwd[idx]
        mean_fwd = float(top_fwd.mean())
        if mean_fwd == 0.0:
            continue
        side = 1.0 if mean_fwd > 0.0 else -1.0
        scores.append(side)
        next_r.append(float(fwd[i]))
        c0s.append(float(c_enter[i]))
        c1s.append(float(c_exit[i]))

    n_eval = len(range(warm, max_i, stride))
    return _metrics(scores, next_r, c0s, c1s, n_eval)


def _metrics(
    scores: list[float],
    next_r: list[float],
    c0: list[float],
    c1: list[float],
    n_eval: int,
) -> Metrics:
    if not scores:
        return Metrics(0.0, 0.0, 0.0, None, 0, n_eval, 0.0)
    hits = 0
    nz = 0
    for s, r in zip(scores, next_r):
        if s == 0.0 or r == 0.0:
            continue
        nz += 1
        if (s > 0) == (r > 0):
            hits += 1
    dir_acc = hits / nz if nz else 0.0

    eq = 10_000.0
    peak = eq
    max_dd = 0.0
    for s, a, b in zip(scores, c0, c1):
        if a <= 0.0 or b <= 0.0 or s == 0.0:
            continue
        eq *= 1.0 + s * (b / a - 1.0)
        peak = max(peak, eq)
        if peak > 0.0:
            max_dd = max(max_dd, (peak - eq) / peak)
    pnl = (eq / 10_000.0 - 1.0) * 100.0
    mdd = max_dd * 100.0
    ratio = (pnl / mdd) if mdd > 1e-12 else None
    cov = len(scores) / n_eval if n_eval else 0.0
    return Metrics(dir_acc, pnl, mdd, ratio, len(scores), n_eval, cov)


def grid_select(closes: np.ndarray) -> tuple[Cfg, Metrics, list[dict]]:
    """Small grid — runnable tonight on ~15k bars."""
    Ls = [16, 32, 64]
    ks = [5, 15]
    taus = [0.70, 0.80, 0.90]
    H = 1
    stride = 2  # half-density queries for select speed
    rows: list[dict] = []
    best: tuple[Cfg, Metrics] | None = None
    for L in Ls:
        for k in ks:
            for tau in taus:
                cfg = Cfg(L=L, H=H, k=k, tau=tau, stride=stride)
                t0 = time.perf_counter()
                m = eval_shape_match(closes, cfg)
                elapsed = time.perf_counter() - t0
                row = {
                    **asdict(cfg),
                    **asdict(m),
                    "elapsed_sec": round(elapsed, 3),
                }
                rows.append(row)
                print(
                    f"select L={L} k={k} tau={tau:.2f} "
                    f"dir={m.dir_acc:.4f} pnl={m.pnl_pct:.3f}% "
                    f"mdd={m.max_dd_pct:.3f}% ratio={m.pnl_over_maxdd} "
                    f"fire={m.n_fire}/{m.n_eval} ({m.coverage:.2%}) "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                # Rank by pnl_over_maxdd (None → -inf), require some fires
                if m.n_fire < 30:
                    continue
                key = m.pnl_over_maxdd if m.pnl_over_maxdd is not None else float("-inf")
                if best is None or key > (
                    best[1].pnl_over_maxdd
                    if best[1].pnl_over_maxdd is not None
                    else float("-inf")
                ):
                    best = (cfg, m)
    if best is None:
        # fallback: most fires among completed
        rows_ok = [r for r in rows if r["n_fire"] > 0]
        if not rows_ok:
            cfg = Cfg(L=32, H=1, k=5, tau=0.7, stride=2)
            return cfg, eval_shape_match(closes, cfg), rows
        r = max(rows_ok, key=lambda x: (x["pnl_over_maxdd"] or -1e9))
        cfg = Cfg(L=r["L"], H=r["H"], k=r["k"], tau=r["tau"], stride=r["stride"])
        return cfg, eval_shape_match(closes, cfg), rows
    return best[0], best[1], rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--oos", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    select_c = load_galoa_closes(args.select)
    oos_c = load_galoa_closes(args.oos)
    print(f"select bars={len(select_c)} from {args.select}", flush=True)
    print(f"oos    bars={len(oos_c)} from {args.oos}", flush=True)

    t0 = time.perf_counter()
    best_cfg, select_m, grid = grid_select(select_c)
    print(f"\nLOCKED cfg: {best_cfg}", flush=True)
    print(f"select metrics: {select_m}", flush=True)

    # OOS: same L/H/k/τ, stride=1 for denser report
    oos_cfg = Cfg(
        L=best_cfg.L,
        H=best_cfg.H,
        k=best_cfg.k,
        tau=best_cfg.tau,
        stride=1,
    )
    oos_m = eval_shape_match(oos_c, oos_cfg)
    print(f"oos metrics: {oos_m}", flush=True)
    print(f"total elapsed {time.perf_counter()-t0:.1f}s", flush=True)

    payload = {
        "method": "discrete_znorm_pearson_shape_match",
        "note": "Not SSA. Fixed-L z-path Pearson; τ-gate; side=mean historical H-fwd return.",
        "select_data": str(args.select).replace("\\", "/"),
        "oos_data": str(args.oos).replace("\\", "/"),
        "locked_cfg": asdict(best_cfg),
        "oos_cfg": asdict(oos_cfg),
        "select": asdict(select_m),
        "oos": asdict(oos_m),
        "grid": grid,
    }
    out = args.out
    if out is None:
        out = Path(__file__).resolve().parent / "results_15d_prior15d.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
