# Galoa

Real-time **spectral similar-segment** price forecasting (Rust). Matching is **spectral-only** on M1 close (bid):

1. **Live / gated path:** log-returns → STFT magnitude → exact kNN → selective gates → **ABSTAIN** (default) or **FIRE**
2. **SpecReTF offline eval** (arXiv:2606.19412 §3.2–3.3): Hanning STFT 50% hop → JSD₂ amp + `cos(mean ΔΦ)` → `s=s_amp+s_phase` → EMA (no renormalize) → top-K → **softmax** futures (**always FIRE**)

There is **no** time-domain product matching path (z-norm L2 appears only as an eval baseline).

**MVP status:** offline **Gold (XAUUSD) M1 bid-close** from IC Markets MT5 demo. Live **cTrader Open API** ingest is stubbed. Synthetic series is **dev-only**.

## Credentials (`.env`)

```bash
cp .env.example .env
# fill MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_SYMBOL
```

- `.env` is **gitignored** — never commit it.
- Demo credentials only; do not put the password in logs, README, PR bodies, or chat.

## Quick start (real Gold M1)

Default series: **`data/xauusd_m1_5d.bin`** (canonical `GALOAP1` binary — MT5 M1 **bar close = bid**).

```bash
# Single query: ABSTAIN or FIRE + bands (gated live-style path)
cargo run -p galoa --release -- forecast

# Walk coverage (FIRE vs ABSTAIN)
cargo run -p galoa --release -- forecast --scan

# SpecReTF offline eval — ALWAYS forecasts; metrics vs naive + time-domain kNN
cargo run -p galoa --release -- eval-specretf --data data/xauusd_m1_5d.bin

# JSON + write metrics file
cargo run -p galoa --release -- eval-specretf --data data/xauusd_m1_5d.bin \
  --metrics-out /tmp/specretf-metrics.json --json

# Paper-protocol retrieval-only on ETT/Exchange (no neural fusion; CSV via download script)
bash scripts/download_paper_datasets.sh
cargo run -p galoa --release -- eval-paper-specretf \
  --data data/ETT/ETTh1.csv --dataset ETTh1 \
  --lookback 720 --horizons 96,192,336,720 \
  --k 10 --n-fft 32 --hop 16 --ema-alpha 0.1 \
  --library-stride 4 --query-stride 24

# Dev-only synthetic / HTTP
cargo run -p galoa -- forecast --synthetic --json
cargo run -p galoa -- serve --port 3000
```

### Defaults

| Param | Gated `forecast` | SpecReTF `eval-specretf` |
|-------|------------------|---------------------------|
| `W` / `H` / `k` | 64 / 16 / 10 | 64 / 16 / 10 |
| STFT | `n_fft=16` `hop=4` (mag → F) | `n_fft=16` `hop=8` (~50% overlap), Hann |
| Score | L2 on L2-normalized mag features | `s_amp=1−JSD₂` + `cos(mean ΔΦ)`; `S=Σα(1−α)^{W−w}s` |
| Aggregate | gated ensemble | **softmax(`S_k`)** over level-aligned futures |
| Gates | `d_max` / `n_min` / `σ_max` | **none** (always FIRE) |

Override: `--window --horizon --k --n-fft --hop` (+ `--ema-alpha --stride` for eval).

## Pipelines

**Gated (live-style):**

```
M1 close bid → log-returns → STFT magnitude → F → kNN → gates → ABSTAIN | FIRE
```

**SpecReTF eval (phase 1):**

```
M1 close bid → log-returns → STFT (Hanning, B=M/2, M bins)
  → p=A/ΣA → JSD₂ → s_amp=1−JSD; s_phase=cos(mean ΔΦ); s=s_amp+s_phase
  → S=Σ α(1−α)^{W−w} s^w (no weight renormalize)
  → top-K causal → level-align Y → softmax(S) aggregate + P10/P90
  → metrics vs actual (MSE/MAE, direction, band)
```

## Refresh Gold from MT5

**Close convention:** MT5 chart / `copy_rates` **close** is **bid-based**.

```bash
python3 scripts/dump_mt5_xauusd_m1.py --days 5          # mt5linux :18812
python scripts/dump_mt5_xauusd_m1.py --days 5 --native  # Windows MT5
```

## Binary format (`GALOAP1`)

Little-endian: magic `GALOAP1\0`, `u64` count, `i64` `ts_ns[]`, `f64` `close[]`.

## Layout

```
crates/
  galoa-core/      # returns, STFT, SpecReTF score, gates, ensemble
  galoa-index/     # gated kNN + SpecReTF index / walk-forward eval
  galoa-ingest/    # GALOAP1 .bin + CSV
  galoa-ctrader/   # stub
  galoa/           # CLI: forecast | eval-specretf | serve
configs/default.toml
data/xauusd_m1_5d.bin
```

## Deferred

- Neural fusion (TimesNet / linear head) — out of scope for MVP retrieval core
- Mel-filterbank; HNSW; live cTrader WS; Parquet; trading/execution
- Phase-2 abstain on SpecReTF path (reuse gated `d_max` / `N_min` / `σ_max`)

## Tests

```bash
cargo test --workspace
```

## License

MIT
