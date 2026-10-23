# Galoa

Real-time **spectral similar-segment** price forecasting (Rust). Matching is **STFT-only** on log-returns of M1 close (bid): historical windows become feature vectors `F`, exact kNN finds analogues, then selective gates decide **ABSTAIN** (default) or **FIRE** with an H-step median + P10/P90 ensemble.

There is **no** time-domain z-norm L2 shape-matching path.

**MVP status:** offline **Gold (XAUUSD) M1 bid-close** replay from an IC Markets MT5 demo. Live **cTrader Open API** ingest is stubbed. Synthetic series is **dev-only** (`--synthetic` / `generate`).

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
# Single query: ABSTAIN or FIRE + bands
cargo run -p galoa --release -- forecast

# JSON
cargo run -p galoa --release -- forecast --data data/xauusd_m1_5d.bin --json

# Walk the 5d series → coverage (FIRE vs ABSTAIN)
cargo run -p galoa --release -- forecast --scan
cargo run -p galoa --release -- forecast --scan --json

# Dev-only synthetic
cargo run -p galoa -- forecast --synthetic --json

# Tiny HTTP API
cargo run -p galoa -- serve --port 3000
```

### Defaults

| Param | Value | Notes |
|-------|-------|--------|
| `W` / `H` / `k` | 64 / 16 / 10 | window length / horizon / neighbours |
| STFT `n_fft` / `hop` | 16 / 4 | Hann on log-returns; onesided mag → `F`, L2-normalized |
| `d_max` | 0.48 | max feature L2 for support (~low-% FIRE on 5d M1) |
| `n_min` | 3 | min support to FIRE |
| `sigma_max` | 0.0025 | max std of H-step log-return across support |

Override via CLI: `--window --horizon --k --n-fft --hop --d-max --n-min --sigma-max`.

## Pipeline (spectral only)

```
M1 close bid
  → log-returns over window W
  → STFT magnitude (rustfft) → feature F
  → exact kNN in F-space
  → gates: d_max / N_min / σ_max
  → ABSTAIN | FIRE (H-step ensemble + bands)
```

**Removed from matching:** z-normalized close windows + L2 distance as the primary (or any) match path.

## Refresh Gold from MT5

**Close convention:** MT5 chart / `copy_rates` **close** is **bid-based**.

```bash
# Linux + mt5linux Docker bridge on :18812
python3 scripts/dump_mt5_xauusd_m1.py --days 5

# Windows native MT5
python scripts/dump_mt5_xauusd_m1.py --days 5 --native
```

## Binary format (`GALOAP1`)

Little-endian: magic `GALOAP1\0`, `u64` count, `i64` `ts_ns[]`, `f64` `close[]`.

## Layout

```
crates/
  galoa-core/      # PriceEvent, log-returns, STFT, gates, ensemble
  galoa-index/     # Exact kNN over STFT features + scan coverage
  galoa-ingest/    # GALOAP1 .bin + CSV load
  galoa-ctrader/   # cTrader client stub
  galoa/           # CLI + axum
configs/default.toml
data/xauusd_m1_5d.bin
```

## Deferred

- Mel-filterbank on STFT; HNSW ANN; live cTrader WS; Parquet; trading/execution

## Tests

```bash
cargo test --workspace
```

## License

MIT
