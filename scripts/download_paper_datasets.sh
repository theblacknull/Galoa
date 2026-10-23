#!/usr/bin/env bash
# Download SpecReTF / Autoformer-style benchmark CSVs (not committed).
# Sources: zhouhaoyi/ETDataset (ETT), AutonLab Timeseries-PILE (Exchange).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/ETT"
mkdir -p "${OUT}"

ETT_BASE="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
for f in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv; do
  dest="${OUT}/${f}"
  if [[ -f "${dest}" ]]; then
    echo "exists: ${dest}"
  else
    echo "fetch ${f}..."
    curl -fsSL -o "${dest}" "${ETT_BASE}/${f}"
  fi
done

EX_DEST="${OUT}/exchange_rate.csv"
if [[ -f "${EX_DEST}" ]]; then
  echo "exists: ${EX_DEST}"
else
  echo "fetch exchange_rate.csv..."
  curl -fsSL -o "${EX_DEST}" \
    "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/exchange_rate.csv"
fi

echo "Done. Files in ${OUT}:"
ls -lh "${OUT}"
