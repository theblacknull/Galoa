# Galoa — SSA H=1 (Gold M1)

Chỉ còn **SSA** + **data** IC Markets. SpecReTF / RAFT / DNN–HMM đã gỡ.

## Layout

```
data/                  # GALOAP1 XAUUSD M1 bid close
ssa/                   # thư viện SSA Rust (GPL-3.0, vendored)
experiments/ssa_h1/    # runner H=1 (Rust)
scripts/dump_mt5_xauusd_m1.py
```

## Chạy thí nghiệm

```bash
cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
  --data data/xauusd_m1_prior5d_b.bin \
  --out experiments/ssa_h1/ssa_prior5d_b_rust.json
```

Pipeline: fit → nhóm tín hiệu theo cumsum contribution → `L_forecast` / `K_forecast` H=1 trên log-return.  
Metric: dir / PnL% / PnL÷MaxDD.

## Data / secrets

- `.env` (gitignored): `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` / `MT5_SYMBOL`
- Dump MT5: `python scripts/dump_mt5_xauusd_m1.py --native --days 5 --stem …`
- Attach terminal đã login: thêm `--attach` (không cần password)
- CSV debug: thêm `--csv` (mặc định chỉ ghi `.bin`)
