# Galoa

Real-time **similar-segment** price forecasting (Rust). Scans history for z-normalized windows that look like the latest move, then ensembles the next *H* bars of those analogues into a median + P10/P90 band.

**MVP status:** offline **Gold (XAUUSD) M1 bid-close** replay from an IC Markets MT5 demo. Live **cTrader Open API** ingest is stubbed. Synthetic series is **dev-only** (`--synthetic` / `generate`).

## Credentials (`.env`)

```bash
cp .env.example .env
# fill MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_SYMBOL
```

- `.env` is **gitignored** — never commit it.
- Demo credentials only; do not put the password in logs, README, PR bodies, or chat.
- Source of truth for this project’s demo secrets may also live outside the repo (agent secret store); copy into `.env` locally.

## Quick start (real Gold M1)

Default series: **`data/xauusd_m1_5d.bin`** (canonical `GALOAP1` binary — MT5 M1 **bar close = bid**). Debug twin: `data/xauusd_m1_5d.csv`. A shorter `*_m1_1d.*` dump may also be present.

```bash
# Forecast from the default 5-day M1 bin
cargo run -p galoa -- forecast

# Explicit path / JSON
cargo run -p galoa -- forecast --data data/xauusd_m1_5d.bin --json
cargo run -p galoa -- forecast --data data/xauusd_m1_5d.csv --json

# Dev-only synthetic path
cargo run -p galoa -- forecast --synthetic --json
cargo run -p galoa -- generate --out data/xauusd_synthetic.csv

# Tiny HTTP API (same default data)
cargo run -p galoa -- serve --port 3000
```

Optional flags: `--window 64 --horizon 16 --k 10 --dt-secs 60`.

## Refresh Gold from MT5

**Close convention:** MT5 chart / `copy_rates` **close** is **bid-based**. That is what we store as `close`. (1-second bars are deferred; MVP timeframe = **M1**.)

### Linux + mt5linux Docker bridge

Official `MetaTrader5` Python wheels are Windows-only. On Linux we use the `lprett/mt5linux` image (Wine + terminal + RPyC). With a container already up on port `18812`:

```bash
python3 scripts/dump_mt5_xauusd_m1.py --days 5
# → data/xauusd_m1_5d.bin + data/xauusd_m1_5d.csv
```

Initialize the remote terminal with an explicit portable path (IPC can time out otherwise). First-time server discovery may need the container’s server-search FIFO (`ICMarketsSC-MT5-6`).

### Windows PC (native MT5)

With MetaTrader 5 installed and logged into the demo:

```bash
pip install MetaTrader5 python-dotenv
python scripts/dump_mt5_xauusd_m1.py --days 5 --native
```

Do **not** invent fake “real” prices if MT5 is unavailable — leave the dump for a machine that can log in.

## Binary format (`GALOAP1`)

Little-endian, mmap-friendly:

| Field | Type |
|-------|------|
| magic | 8 bytes `GALOAP1\0` |
| count `n` | `u64` |
| `ts_ns[0..n)` | `i64` each |
| `close[0..n)` | `f64` each |

CSV debug twin uses header `ts_ns,close`.

## Layout

```
crates/
  galoa-core/      # PriceEvent (ts, close), resample, z-norm, ensemble forecast
  galoa-index/     # Exact in-memory kNN over windows (ANN later)
  galoa-ingest/    # GALOAP1 .bin + CSV load; optional synthetic
  galoa-ctrader/   # cTrader client stub
  galoa/           # CLI + axum
scripts/dump_mt5_xauusd_m1.py
data/xauusd_m1_5d.bin   # default forecast input (committed sample)
configs/default.toml
```

## Method (short)

1. Load `(ts_ns, close)` for `XAUUSD` (default: 5d M1 bid closes).
2. Optional last-close resample at Δt.
3. Index every window of length `W` (z-normalized); keep next `H` absolute closes.
4. Query with the latest window → exact top-`k` L2 neighbours in z-space.
5. Level-align analogue futures → median / P10 / P90.

Defaults: `W=64`, `H=16`, `k=10`.

## Deferred

- 1-second time bars; live cTrader WS/Protobuf; HNSW ANN; Parquet; trading/execution

## Tests

```bash
cargo test --workspace
```

## License

MIT
