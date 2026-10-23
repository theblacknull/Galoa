#!/usr/bin/env python3
"""Paper-faithful RAFT runner (arXiv:2505.04163 / archon159/RAFT).

Examples:
  # ETTh1, paper App. B for F=96: L=720, m=20, lr=1e-3
  python3 scripts/raft_paper/run_raft_faithful.py --dataset etth1 \\
      --seq-len 720 --pred-len 96 --topm 20 --lr 1e-3 --epochs 10

  # Gold M1 5d, paper arch with adapted L/F
  python3 scripts/raft_paper/run_raft_faithful.py --dataset gold \\
      --data data/xauusd_m1_5d.bin --seq-len 96 --pred-len 16 --topm 10 --lr 1e-3
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

# local imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_util import load_etth1, load_gold  # noqa: E402
from model import RAFTModel, retrieval_identity_forecast  # noqa: E402


PAPER_ETTH1_F96 = {
    "mse": 0.367,
    "source": "Table 16 (Appendix G), ETTh1 F=96",
    "L": 720,
    "m": 20,
    "lr": 1e-3,
    "tau": 0.1,
    "periods": [1, 2, 4],
}


def adjust_learning_rate(optimizer, epoch, lr0, lradj="type1"):
    if lradj == "type1":
        lr = lr0 * (0.5 ** ((epoch - 1) // 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr
    return lr0


def mse_mae(pred: np.ndarray, true: np.ndarray):
    return float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))


def dir_hit(pred: np.ndarray, true: np.ndarray, x_last: np.ndarray) -> float:
    """Horizon-end direction vs last lookback (univariate channel 0)."""
    # pred/true: [N, F, C]
    da = true[:, -1, 0] - x_last[:, 0]
    dp = pred[:, -1, 0] - x_last[:, 0]
    ok = np.sign(da) == np.sign(dp)
    # ignore exact zeros
    mask = (np.abs(da) > 1e-12) | (np.abs(dp) > 1e-12)
    if mask.sum() == 0:
        return 0.0
    return float(ok[mask].mean())


def naive_last(x: np.ndarray, pred_len: int) -> np.ndarray:
    last = x[:, -1:, :]
    return np.repeat(last, pred_len, axis=1)


def train_and_eval(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")

    if args.dataset == "etth1":
        csv = Path(args.data) if args.data else Path("data/ETT/ETTh1.csv")
        splits = load_etth1(csv, args.seq_len, args.pred_len, features=args.features)
        dataset_name = "ETTh1"
    elif args.dataset == "gold":
        bin_path = Path(args.data) if args.data else Path("data/xauusd_m1_5d.bin")
        splits = load_gold(bin_path, args.seq_len, args.pred_len)
        dataset_name = "Gold_M1_5d"
    else:
        raise SystemExit(f"unknown dataset {args.dataset}")

    channels = splits["train"].channels
    print(
        f"{dataset_name}: channels={channels} "
        f"n_train={len(splits['train'].x)} n_val={len(splits['valid'].x)} "
        f"n_test={len(splits['test'].x)} L={args.seq_len} F={args.pred_len}"
    )

    # Pred_len must be divisible by all periods (official compresses F/g)
    for g in [4, 2, 1][: args.n_period]:
        if args.pred_len % g != 0:
            raise SystemExit(f"pred_len={args.pred_len} must be divisible by period g={g}")
        if args.seq_len % g != 0:
            raise SystemExit(f"seq_len={args.seq_len} must be divisible by period g={g}")

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
    retrieval_secs = time.time() - t0
    print(f"retrieval precompute: {retrieval_secs:.1f}s")

    def make_loader(split_name: str, shuffle: bool):
        sp = splits[split_name]
        ds = TensorDataset(
            torch.from_numpy(sp.x),
            torch.from_numpy(sp.y),
            torch.from_numpy(sp.idx),
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
        )

    train_loader = make_loader("train", shuffle=True)
    valid_loader = make_loader("valid", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    def eval_loader(loader, mode):
        model.eval()
        preds, trues, xs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, index in loader:
                batch_x_d = batch_x.to(device)
                index_d = index.to(device)
                out = model(batch_x_d, index_d, mode=mode)
                preds.append(out.cpu().numpy())
                trues.append(batch_y.numpy())
                xs.append(batch_x.numpy())
        if not preds:
            return 0.0, None, None, None
        p = np.concatenate(preds, axis=0)
        t = np.concatenate(trues, axis=0)
        x = np.concatenate(xs, axis=0)
        return float(np.mean((p - t) ** 2)), p, t, x

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None
    history = []

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
        train_loss = float(np.mean(losses)) if losses else 0.0

        val_loss, _, _, _ = eval_loader(valid_loader, "valid")
        test_loss_ep, _, _, _ = eval_loader(test_loader, "test")
        lr_now = adjust_learning_rate(optim, epoch, args.lr, args.lradj)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "test_mse": test_loss_ep,
                "lr": lr_now,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs} train={train_loss:.6f} "
            f"val={val_loss:.6f} test={test_loss_ep:.6f} lr={lr_now}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    _, pred_s, true_s, x_s = eval_loader(test_loader, "test")
    assert pred_s is not None and true_s is not None and x_s is not None

    mse_s, mae_s = mse_mae(pred_s, true_s)
    naive_s = naive_last(x_s, args.pred_len)
    naive_mse_s, naive_mae_s = mse_mae(naive_s, true_s)

    # Identity retrieval baseline (no trained head)
    rt = model.retrieval_dict["test"]
    with torch.no_grad():
        id_parts = []
        for start in range(0, len(x_s), args.batch_size):
            end = min(start + args.batch_size, len(x_s))
            xb = torch.from_numpy(x_s[start:end]).float()
            # index into retrieval_dict is 0..N-1 for this split
            idx = torch.arange(start, end)
            # manually slice retrieval for these indices
            # retrieval_identity needs full retrieval tensor for batch — slice G dim
            # Use encoder path without linear: call helper with rt[:, start:end]
            id_pred = retrieval_identity_forecast(
                rt[:, start:end], xb, model.period_num
            )
            id_parts.append(id_pred.numpy())
        id_s = np.concatenate(id_parts, axis=0)
    id_mse_s, id_mae_s = mse_mae(id_s, true_s)

    # Raw-scale metrics (inverse StandardScaler per channel, last-dim)
    scaler = splits["test"].scaler

    def inv(arr):
        # arr [N,F,C]
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        return scaler.inverse_transform(flat).reshape(shape)

    pred_r = inv(pred_s)
    true_r = splits["test"].raw_y
    naive_r = inv(naive_s)
    id_r = inv(id_s)
    x_r = inv(x_s)
    mse_r, mae_r = mse_mae(pred_r, true_r)
    naive_mse_r, naive_mae_r = mse_mae(naive_r, true_r)
    id_mse_r, id_mae_r = mse_mae(id_r, true_r)
    dir_mlp = dir_hit(pred_r, true_r, x_r[:, -1, :])
    dir_id = dir_hit(id_r, true_r, x_r[:, -1, :])
    dir_naive = dir_hit(naive_r, true_r, x_r[:, -1, :])

    paper_ref = None
    fidelity = None
    if (
        args.dataset == "etth1"
        and args.pred_len == 96
        and args.seq_len == 720
        and args.features == "M"
    ):
        paper_ref = PAPER_ETTH1_F96
        # lower MSE is better; fidelity as 1 - |ours-paper|/paper, clipped
        fidelity = {
            "paper_mse": paper_ref["mse"],
            "ours_mse_scaled": mse_s,
            "abs_gap": abs(mse_s - paper_ref["mse"]),
            "rel_gap": abs(mse_s - paper_ref["mse"]) / paper_ref["mse"],
            "fidelity_pct": max(
                0.0, 100.0 * (1.0 - abs(mse_s - paper_ref["mse"]) / paper_ref["mse"])
            ),
        }

    report = {
        "dataset": dataset_name,
        "protocol": "paper_faithful_official_arch",
        "device": str(device),
        "params": {
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "topm": args.topm,
            "tau": args.tau,
            "n_period": args.n_period,
            "periods_official": model.period_num,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "features": args.features if args.dataset == "etth1" else "S",
            "channels": channels,
        },
        "splits": {
            "n_train": len(splits["train"].x),
            "n_valid": len(splits["valid"].x),
            "n_test": len(splits["test"].x),
        },
        "timing": {"retrieval_precompute_sec": retrieval_secs},
        "train_history": history,
        "best_val_mse_scaled": best_val,
        "test_scaled": {
            "raft_mlp_mse": mse_s,
            "raft_mlp_mae": mae_s,
            "raft_identity_mse": id_mse_s,
            "raft_identity_mae": id_mae_s,
            "naive_last_mse": naive_mse_s,
            "naive_last_mae": naive_mae_s,
        },
        "test_raw": {
            "raft_mlp_mse": mse_r,
            "raft_mlp_mae": mae_r,
            "raft_identity_mse": id_mse_r,
            "raft_identity_mae": id_mae_r,
            "naive_last_mse": naive_mse_r,
            "naive_last_mae": naive_mae_r,
            "dir_hit_h_mlp": dir_mlp,
            "dir_hit_h_identity": dir_id,
            "dir_hit_h_naive": dir_naive,
        },
        "paper_reference": paper_ref,
        "fidelity_vs_paper": fidelity,
        "deviations": [],
        "notes": [
            "Head is shallow linear f/g/h as in official models/RAFT.py (paper calls this shallow MLP).",
            "Retrieval library = train split only; val/test retrieve from train (official protocol).",
            "Scaled metrics use StandardScaler fit on train (paper / TSLib).",
        ],
    }

    # Document deviations
    if device.type != "cuda":
        report["deviations"].append("Ran on CPU (paper used NVIDIA A100).")
    if args.epochs != 10:
        report["deviations"].append(f"epochs={args.epochs} (official run.py default 10).")
    if args.dataset == "gold":
        report["deviations"].extend(
            [
                "Gold univariate M1 close (paper multivariate benchmarks).",
                f"Adapted L={args.seq_len} F={args.pred_len} (paper ETTh1 uses L=720 F∈{{96..720}}).",
                "Chronological 70/15/15 split (paper uses ETT fixed calendar borders).",
                "Always-FIRE MSE eval — not a trading backtest.",
            ]
        )
    if args.dataset == "etth1" and args.seq_len != 720:
        report["deviations"].append(
            f"seq_len={args.seq_len} (paper App. B ETTh1 F={args.pred_len} uses L=720)."
        )
    report["deviations"].append("Single random seed (paper averages 3 seeds).")

    return report


def main():
    p = argparse.ArgumentParser(description="Paper-faithful RAFT (official architecture)")
    p.add_argument("--dataset", choices=["etth1", "gold"], required=True)
    p.add_argument("--data", type=str, default="", help="CSV (ETT) or GALOAP1 .bin (Gold)")
    p.add_argument("--seq-len", type=int, default=720)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--topm", type=int, default=20)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--n-period", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--retrieval-batch", type=int, default=128)
    p.add_argument("--lradj", type=str, default="type1")
    p.add_argument("--features", type=str, default="M", choices=["M", "S"])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--metrics-out", type=str, default="")
    args = p.parse_args()

    # Sensible Gold defaults if user left ETTh1-sized L
    if args.dataset == "gold" and args.seq_len == 720 and args.pred_len == 96:
        # keep as requested — may still fit on 5d
        pass

    report = train_and_eval(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
