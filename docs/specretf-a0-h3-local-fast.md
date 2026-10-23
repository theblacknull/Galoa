# SpecReTF Gold M1 — α=0, H=3, local fast path

Canonical copy also in Project store: `docs/specretf-a0-h3-local-fast.md` (agent store).

Date: 2026-10-23 (Skywalker Windows).  
Data: `data/xauusd_m1_5d.bin` (6197 bars).  
Branch: `cursor/galoa-rust-mvp-gold-b290` · PR #1.

## Config

W=64, H=3, M=8, hop=4, **α=0**, K=20, signal=`log_returns`, stride=1, `--release`.

## Metrics vs naive_last

| Method | MSE close | MAE close | dir_H |
|--------|----------:|----------:|------:|
| specretf | 3.135 | 1.255 | 0.502 |
| naive_last | **3.009** | **1.216** | 0.002 |
| td_znorm_l2_knn | 3.176 | 1.265 | 0.521 |

Δ MSE close vs naive: +4.18% (no edge claim).

## Runtime

| | seconds |
|--|--------:|
| Before (serial cached) | ~56.0 |
| After total | ~9.2 |
| feature_build | 0.031 |
| retrieval | 9.11 |
| naive full-STFT estimate | ~111 |

~6× vs prior release; ~12× vs naive full-STFT estimate. Metrics unchanged.

## Speedups

Library STFT cache, thread-local FFT plans, query descriptor reuse, α≤0 no-alloc mean, rayon + top-K heap, timing prints.
