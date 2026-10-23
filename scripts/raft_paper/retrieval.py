"""RAFT retrieval module — faithful to official layers/Retrieval.py (archon159/RAFT).

Multi-period via unfold/mean/repeat_interleave (periods from [16,8,4,2,1][-n_period:]),
Pearson correlation, top-m softmax with temperature τ, weighted sum of offset values.
Library is built from the training split only (paper / official protocol).
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class RetrievalTool:
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        n_period: int = 3,
        temperature: float = 0.1,
        topm: int = 20,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period :]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)  # e.g. [4, 2, 1]
        self.temperature = temperature
        self.topm = topm

        self.train_data_all: Optional[torch.Tensor] = None
        self.train_data_all_mg: Optional[torch.Tensor] = None
        self.y_data_all: Optional[torch.Tensor] = None
        self.y_data_all_mg: Optional[torch.Tensor] = None
        self.n_train = 0

    def prepare_from_arrays(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """x_train: [T, L, C], y_train: [T, F, C] (already scaled)."""
        self.train_data_all = torch.tensor(x_train).float()
        self.y_data_all = torch.tensor(y_train).float()
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)
        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(
        self, data_all: torch.Tensor, remove_offset: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        data_all = copy.deepcopy(data_all)  # T, S, C
        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            # pad/truncate to original length if needed
            if cur.shape[1] < data_all.shape[1]:
                pad = data_all.shape[1] - cur.shape[1]
                cur = torch.cat([cur, cur[:, -1:, :].repeat(1, pad, 1)], dim=1)
            elif cur.shape[1] > data_all.shape[1]:
                cur = cur[:, : data_all.shape[1], :]
            mg.append(cur)
        mg = torch.stack(mg, dim=0)  # G, T, S, C
        if remove_offset:
            offsets = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:, -1:, :]
                mg[i] = data_p - cur_offset
                offsets.append(cur_offset)
            offset = torch.stack(offsets, dim=0)
        else:
            offset = None
        return mg, offset

    def periodic_batch_corr(
        self, data_all: torch.Tensor, key: torch.Tensor, in_bsz: int = 512
    ) -> torch.Tensor:
        _, bsz, _features = key.shape
        _, train_len, _ = data_all.shape
        bx = key - torch.mean(key, dim=2, keepdim=True)
        iters = math.ceil(train_len / in_bsz)
        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            cur_sim = torch.bmm(
                F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2)
            )
            sim.append(cur_sim)
        return torch.cat(sim, dim=2)

    def retrieve(
        self, x: torch.Tensor, index: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        assert self.train_data_all_mg is not None
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_mg, _ = self.decompose_mg(x)
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),
            x_mg.flatten(start_dim=2),
        )  # G, B, T

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(
                x.device
            )
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (
                index - self.seq_len - self.pred_len + 1
            ).unsqueeze(dim=1)
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(
                sliding_index < self.n_train, sliding_index, self.n_train - 1
            )
            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.0)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)
            sim = sim.masked_fill_(self_mask.bool(), float("-inf"))

        sim = sim.reshape(self.n_period * bsz, self.n_train)
        topm_index = torch.topk(sim, min(self.topm, self.n_train), dim=1).indices
        ranking_sim = torch.ones_like(sim) * float("-inf")
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train)

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2).detach().cpu()
        y_data_all = self.y_data_all_mg.flatten(start_dim=2)  # G, T, P*C
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(
            self.n_period, bsz, -1, channels
        )
        return pred_from_retrieval.to(x.device)

    @torch.no_grad()
    def retrieve_all(
        self,
        xs: np.ndarray,
        indices: np.ndarray,
        train: bool,
        batch_size: int = 256,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """xs: [N, L, C], indices: [N]. Returns [G, N, F, C]."""
        device = device or torch.device("cpu")
        outs = []
        n = xs.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_x = torch.tensor(xs[start:end]).float().to(device)
            idx = torch.tensor(indices[start:end]).long().to(device)
            outs.append(self.retrieve(batch_x, idx, train=train).cpu())
        return torch.cat(outs, dim=1)
