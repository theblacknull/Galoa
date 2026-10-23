#!/usr/bin/env bash
# Gold M1 $10k trading-style backtest (ours — not paper PnL).
# Papers: original series + StandardScaler → MSE/MAE. This script: USD equity.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/xauusd_m1_5d.bin}"
EQUITY="${EQUITY:-10000}"
COST_BPS="${COST_BPS:-2}"
MLP_OUT="${MLP_OUT:-/tmp/galoa-raft-mlp-signals.json}"
METRICS_OUT="${METRICS_OUT:-/tmp/galoa-gold-10k-backtest.json}"
SKIP_MLP="${SKIP_MLP:-0}"

echo "==> optional paper-faithful RAFT MLP signals"
if [[ "$SKIP_MLP" != "1" ]]; then
  python3 scripts/raft_paper/dump_raft_mlp_signals.py \
    --data "$DATA" --seq-len 96 --pred-len 16 --topm 10 --epochs 10 --cpu \
    --out "$MLP_OUT"
  MLP_ARG=(--raft-mlp-signals "$MLP_OUT")
else
  MLP_ARG=()
fi

echo "==> Rust backtest (SpecReTF fusion + RAFT ridge + baselines)"
cargo run -p galoa --release -- backtest \
  --data "$DATA" \
  --equity "$EQUITY" \
  --cost-rt-bps "$COST_BPS" \
  --metrics-out "$METRICS_OUT" \
  "${MLP_ARG[@]}"

echo "metrics → $METRICS_OUT"
