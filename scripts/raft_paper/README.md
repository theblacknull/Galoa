# Paper-faithful RAFT (official architecture)

Port of [archon159/RAFT](https://github.com/archon159/RAFT) retrieval + linear `f/g/h` head
(arXiv:2505.04163). Independent of the Rust walk-forward MVP in `crates/galoa-raft/`.

## Dependencies

```bash
pip install 'numpy<2' 'pandas==2.2.2' 'scikit-learn==1.4.2' 'tqdm' \
  --extra-index-url https://download.pytorch.org/whl/cpu
pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
# Paper datasets (gitignored CSVs)
bash scripts/download_paper_datasets.sh

# ETTh1 — Appendix B for F=96 (L=720, m=20, lr=1e-3)
python3 scripts/raft_paper/run_raft_faithful.py --dataset etth1 \
  --data data/ETT/ETTh1.csv --seq-len 720 --pred-len 96 --topm 20 --lr 1e-3 \
  --epochs 10 --metrics-out /tmp/raft-etth1.json

# Gold M1 — same head; L/F adapted to series length
python3 scripts/raft_paper/run_raft_faithful.py --dataset gold \
  --data data/xauusd_m1_5d.bin --seq-len 96 --pred-len 16 --topm 10 --lr 1e-3 \
  --epochs 10 --metrics-out /tmp/raft-gold.json
```

Results write-up (Vietnamese): Project store
`docs/galoa-raft-paper-faithful-results.md`.
