# Discrete shape matching (Gold M1)

**Not SSA.** Not periodic. Not “window finds the cycle.”

User framing: the problem is **discrete** — retrieve past *shapes* that look like the current path; when similarity is high enough, **select** that trade side from historical forward outcomes.

## Tonight prototype

`run_shape_match.py`

1. Load `GALOAP1` closes → build length-`L` close windows  
2. Z-score each window (level/scale free “shape”)  
3. Similarity = Pearson ≈ `(Z_i · Z_q) / L`  
4. Causal library only (past ends before query − `H`)  
5. If mean top-`k` similarity ≥ `τ` → FIRE; side = sign(mean historical H-step log-return)  
6. Else ABSTAIN  

Metrics: **dir_acc**, **pnl_pct**, **pnl_over_maxdd**.

## Run

```bash
python experiments/shape_match/run_shape_match.py \
  --select data/xauusd_m1_15d.bin \
  --oos data/xauusd_m1_prior15d.bin
```

Select: small grid on 15d (`L∈{16,32,64}`, `k∈{5,15}`, `τ∈{0.7,0.8,0.9}`, `H=1`, stride=2).  
OOS: lock best by `pnl_over_maxdd` (min 30 fires), stride=1 on prior15d.

## Next (not tonight)

- Variable-length motifs + DTW / soft-DTW  
- Multi-scale library (several L at once)  
- Rust port if select cell survives OOS  
- Abandon sibling SSA H>1 work for this product ask  
