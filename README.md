# Galoa

Real-time **spectral similar-segment** price forecasting (Rust). Matching is **spectral-only** on M1 close (bid):

1. **Live / gated path:** log-returns → STFT magnitude → exact kNN → selective gates → **ABSTAIN** (default) or **FIRE**
2. **SpecReTF offline eval:** STFT frames → Jensen–Shannon(amplitude) + phase cosine + EMA recency → top-K → weighted aggregate futures (**always FIRE** — phase-1 accuracy protocol)

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

# Dev-only synthetic / HTTP
cargo run -p galoa -- forecast --synthetic --json
cargo run -p galoa -- serve --port 3000
```

### Defaults

| Param | Gated `forecast` | SpecReTF `eval-specretf` |
|-------|------------------|---------------------------|
| `W` / `H` / `k` | 64 / 16 / 10 | 64 / 16 / 10 |
| STFT | `n_fft=16` `hop=4` (mag → F) | `n_fft=16` `hop=8` (~50% overlap), Hann |
| Score | L2 on L2-normalized mag features | JS(amp dist) + phase cosine + EMA α=0.1 |
| Gates | `d_max` / `n_min` / `σ_max` | **none** (always FIRE) |

Override: `--window --horizon --k --n-fft --hop` (+ `--ema-alpha --stride` for eval).

## Pipelines

**Gated (live-style):**

```
M1 close bid → log-returns → STFT magnitude → F → kNN → gates → ABSTAIN | FIRE
```

**SpecReTF eval (phase 1):**

```
M1 close bid → log-returns → STFT frames (amp+phase)
  → JS(amp) + cosine(phase) + EMA recency → top-K (exclude overlap, causal)
  → level-align futures → weighted mean / median / P10 / P90
  → metrics vs actual (MSE/MAE, direction, band coverage)
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
