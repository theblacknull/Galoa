"""Dataset builders for paper-faithful RAFT (ETT splits + Gold GALOAP1)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitArrays:
    x: np.ndarray  # [N, L, C]
    y: np.ndarray  # [N, F, C]
    idx: np.ndarray  # [N] index into this split's window list (0..N-1)
    raw_y: np.ndarray  # unscaled y for inverse metrics (same shape as y)
    scaler: StandardScaler
    n_total_points: int
    channels: int


def _make_windows(
    data: np.ndarray, seq_len: int, pred_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """data: [T, C] → x [N,L,C], y [N,F,C]."""
    t = data.shape[0]
    n = t - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError(f"series too short: T={t}, L={seq_len}, F={pred_len}")
    xs, ys = [], []
    for i in range(n):
        xs.append(data[i : i + seq_len])
        ys.append(data[i + seq_len : i + seq_len + pred_len])
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(
        np.float32
    )


def load_etth1(
    csv_path: Path, seq_len: int, pred_len: int, features: str = "M"
) -> Dict[str, SplitArrays]:
    """Official ETT hourly borders (TSLib / RAFT)."""
    df_raw = pd.read_csv(csv_path)
    if features == "M":
        cols = df_raw.columns[1:]
        df_data = df_raw[cols]
    else:
        df_data = df_raw[["OT"]]

    # borders identical to Dataset_ETT_hour
    border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    scaler = StandardScaler()
    train_raw = df_data.values[border1s[0] : border2s[0]]
    scaler.fit(train_raw)
    data = scaler.transform(df_data.values).astype(np.float64)
    raw = df_data.values.astype(np.float64)

    out: Dict[str, SplitArrays] = {}
    for name, bi in zip(["train", "valid", "test"], range(3)):
        b1, b2 = border1s[bi], border2s[bi]
        scaled = data[b1:b2]
        raw_slice = raw[b1:b2]
        x, y = _make_windows(scaled, seq_len, pred_len)
        _, y_raw = _make_windows(raw_slice, seq_len, pred_len)
        out[name] = SplitArrays(
            x=x,
            y=y,
            idx=np.arange(len(x), dtype=np.int64),
            raw_y=y_raw,
            scaler=scaler,
            n_total_points=b2 - b1,
            channels=x.shape[-1],
        )
    return out


def load_galoa_bin(path: Path) -> np.ndarray:
    """Load GALOAP1 → close array [T]."""
    buf = path.read_bytes()
    if buf[:8] != b"GALOAP1\0":
        raise ValueError(f"bad magic in {path}")
    n = struct.unpack_from("<Q", buf, 8)[0]
    close_off = 16 + n * 8
    closes = np.frombuffer(buf, dtype="<f8", count=n, offset=close_off).astype(
        np.float64
    )
    return closes


def load_gold(
    bin_path: Path,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, SplitArrays]:
    """Chronological split on univariate Gold M1 close; scaler fit on train only.

    Borders mimic ETT spirit: val/test windows start with seq_len overlap into
    previous split so lookback can use prior context, but retrieval library is
    train-only (set in prepare_retrieval).
    """
    closes = load_galoa_bin(bin_path)
    t = len(closes)
    n_train = int(t * train_ratio)
    n_val = int(t * val_ratio)
    n_test = t - n_train - n_val
    if n_train < seq_len + pred_len or n_val < seq_len + pred_len or n_test < seq_len + pred_len:
        raise ValueError(
            f"Gold too short for L={seq_len} F={pred_len}: T={t} "
            f"splits=({n_train},{n_val},{n_test})"
        )

    # Point borders (like ETT): train [0, n_train), val starts at n_train-seq_len
    border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
    border2s = [n_train, n_train + n_val, t]

    raw = closes.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(raw[border1s[0] : border2s[0]])
    data = scaler.transform(raw)

    out: Dict[str, SplitArrays] = {}
    for name, bi in zip(["train", "valid", "test"], range(3)):
        b1, b2 = border1s[bi], border2s[bi]
        scaled = data[b1:b2]
        raw_slice = raw[b1:b2]
        x, y = _make_windows(scaled, seq_len, pred_len)
        _, y_raw = _make_windows(raw_slice, seq_len, pred_len)
        out[name] = SplitArrays(
            x=x,
            y=y,
            idx=np.arange(len(x), dtype=np.int64),
            raw_y=y_raw,
            scaler=scaler,
            n_total_points=b2 - b1,
            channels=1,
        )
    return out
