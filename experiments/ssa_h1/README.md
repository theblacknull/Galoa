# SSA H=1 runner (Rust)

Depends on vendored `../../ssa/rust` (`ssa_rs`). Not part of any MIT workspace.

```bash
# from Galoa root
cargo run --release --manifest-path experiments/ssa_h1/rust/Cargo.toml -- \
  --data data/xauusd_m1_prior5d_b.bin \
  --out experiments/ssa_h1/ssa_prior5d_b_rust.json
```
