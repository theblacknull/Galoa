# SSA H=1 (Rust only)

Canonical runner: **Rust** + `ssa_rs` (GPL-3.0 at `Desktop/ssa/rust`).  
This crate is **not** a Galoa workspace member (empty `[workspace]` in `rust/Cargo.toml`).  
Legacy `*.py` scripts here are unused.

```bash
# from Galoa root
cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
  --data data/xauusd_m1_prior5d_b.bin \
  --out experiments/ssa_h1/ssa_prior5d_b_rust.json
```

Architecture: fit → signal group by cumsum contribution → L/K forecast H=1 on log-returns.  
Metrics: dir_acc, pnl_pct, pnl_over_maxdd. Parallel: rayon, stride=3.

Data: `data/xauusd_m1_prior5d_b.bin` (IC Markets). Do not commit GPL `ssa` into Galoa members.
