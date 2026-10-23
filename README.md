# Galoa

Real-time **similar-segment** price forecasting (Rust). Scans history for z-normalized windows that look like the latest move, then ensembles the next *H* bars of those analogues into a median + P10/P90 band.

**MVP status:** offline **Gold (XAUUSD)** test/replay only. Live **cTrader Open API** ingest is stubbed until credentials arrive — no secrets and no fake OpenAPI credentials in this repo.

## Quick start (Gold test data)

```bash
# Generate synthetic XAUUSD closes (random-walk + planted similar segments)
cargo run -p galoa -- generate --out data/xauusd_synthetic.csv

# Forecast from that file (W=64, H=16, k=10)
cargo run -p galoa -- forecast --data data/xauusd_synthetic.csv

# JSON output
cargo run -p galoa -- forecast --data data/xauusd_synthetic.csv --json

# Or skip the file and use in-memory synthetic data
cargo run -p galoa -- forecast --synthetic --json

# Tiny HTTP API
cargo run -p galoa -- serve --data data/xauusd_synthetic.csv --port 3000
# curl -s -X POST http://127.0.0.1:3000/forecast -H 'content-type: application/json' -d '{}'
```

Optional flags: `--window 64 --horizon 16 --k 10 --dt-secs 60` (resample last-close to 60s buckets).

## Layout

```
crates/
  galoa-core/      # PriceEvent (ts, close), resample, z-norm, ensemble forecast
  galoa-index/     # Exact in-memory kNN over windows (ANN later)
  galoa-ingest/    # CSV load + synthetic Gold generator
  galoa-ctrader/   # cTrader client stub (map spot/bar → PriceEvent)
  galoa/           # CLI + axum (`forecast` / `serve` / `generate`)
configs/default.toml
data/              # generated CSV (gitignored content optional; regenerate anytime)
```

## Method (short)

1. Load `(ts_ns, close)` events for `XAUUSD`.
2. Optional last-close resample at Δt.
3. Index every window of length `W` (z-normalized); keep next `H` absolute closes.
4. Query with the latest window → exact top-`k` L2 neighbours in z-space.
5. Level-align analogue futures to the query last close → median / P10 / P90.

Defaults: `W=64`, `H=16`, `k=10`.

## Deferred (not in this PR)

- Live cTrader WebSocket / Protobuf auth, heartbeat, subscribe, reconnect
- HNSW / usearch ANN (exact kNN is fine for MVP series length)
- Parquet history, multi-symbol, walk-forward eval crate
- Trading / order execution (read-only product)

## Tests

```bash
cargo test --workspace
```

## License

MIT
