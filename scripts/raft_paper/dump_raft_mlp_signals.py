#!/usr/bin/env python3
"""Dump paper-faithful RAFT MLP test-split signals for the Gold $10k backtest.

Outputs JSON: [{end, close, pred_h}, ...] where `end` is the absolute bar index
of the first future bar (same convention as Rust ForecastSignal).

Representation note: RAFT trains/evals on StandardScaler(original closes) —
not log-returns as the primary series. This dump is for our trading wrapper only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_util import load_galoa_bin, load_gold  # noqa: E402
from model import RAFTModel  # noqa: E402


def adjust_learning_rate(optimizer, epoch, lr0, lradj="type1"):
    if lradj == "type1":
        lr = lr0 * (0.5 ** ((epoch - 1) // 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr
    return lr0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/xauusd_m1_5d.bin")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=16)
    p.add_argument("--topm", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--n-period", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--retrieval-batch", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", required=True, help="JSON signals path")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    bin_path = Path(args.data)
    closes = load_galoa_bin(bin_path)
    t = len(closes)
    n_train = int(t * 0.70)
    n_val = int(t * 0.15)
    # test slice border1 (with lookback overlap) matches load_gold
    test_b1 = n_train + n_val - args.seq_len

    splits = load_gold(bin_path, args.seq_len, args.pred_len)
    channels = splits["train"].channels
    model = RAFTModel(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        channels=channels,
        n_period=args.n_period,
        topm=args.topm,
        temperature=args.tau,
    ).to(device)

    t0 = time.time()
    model.prepare_retrieval(
        splits["train"].x,
        splits["train"].y,
        splits["train"].idx,
        splits["valid"].x,
        splits["valid"].idx,
        splits["test"].x,
        splits["test"].idx,
        device=device,
        batch_size=args.retrieval_batch,
    )
    print(f"retrieval {time.time() - t0:.1f}s device={device}", file=sys.stderr)

    def make_loader(name, shuffle):
        sp = splits[name]
        ds = TensorDataset(
            torch.from_numpy(sp.x),
            torch.from_numpy(sp.y),
            torch.from_numpy(sp.idx),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader("train", True)
    valid_loader = make_loader("valid", False)
    test_loader = make_loader("test", False)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, index in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            index = index.to(device)
            optim.zero_grad()
            out = model(batch_x, index, mode="train")
            loss = criterion(out, batch_y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, index in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                index = index.to(device)
                out = model(batch_x, index, mode="valid")
                val_losses.append(float(criterion(out, batch_y).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        adjust_learning_rate(optim, epoch, args.lr)
        print(
            f"epoch {epoch} train={np.mean(losses):.6f} val={val_loss:.6f}",
            file=sys.stderr,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _batch_y, index in test_loader:
            batch_x = batch_x.to(device)
            index = index.to(device)
            out = model(batch_x, index, mode="test")
            preds.append(out.cpu().numpy())
    pred_s = np.concatenate(preds, axis=0)  # [N, F, C]
    scaler = splits["test"].scaler
    flat = pred_s.reshape(-1, pred_s.shape[-1])
    pred_r = scaler.inverse_transform(flat).reshape(pred_s.shape)

    signals = []
    for i in range(len(pred_r)):
        end = test_b1 + i + args.seq_len
        close = float(closes[end - 1])
        pred_h = float(pred_r[i, -1, 0])
        signals.append({"end": end, "close": close, "pred_h": pred_h})

    out = {
        "note": "paper-faithful RAFT MLP on Gold test split only; for our backtest wrapper",
        "params": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "topm": args.topm,
            "tau": args.tau,
            "epochs": args.epochs,
            "test_b1": test_b1,
            "n_signals": len(signals),
        },
        "signals": signals,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(signals)} signals → {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
