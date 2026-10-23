"""RAFT forecast head — faithful to official models/RAFT.py.

ŷ = h( f(ˆx) ⊕ Σ_p g^(p)(ṽ^(p)) ) + x^L
with linear f: L→F, g^(p): F/p → F (after compressing repeat dims), h: 2F→F.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from retrieval import RetrievalTool


class RAFTModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        n_period: int = 3,
        topm: int = 20,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period
        self.topm = topm

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        self.rt = RetrievalTool(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            n_period=n_period,
            temperature=temperature,
            topm=topm,
        )
        self.period_num = self.rt.period_num[-1 * self.n_period :]
        self.retrieval_pred = nn.ModuleList(
            [nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num]
        )
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.retrieval_dict: Dict[str, torch.Tensor] = {}

    def prepare_retrieval(
        self,
        train_x,
        train_y,
        train_idx,
        valid_x,
        valid_idx,
        test_x,
        test_idx,
        device: torch.device,
        batch_size: int = 256,
    ) -> None:
        self.rt.prepare_from_arrays(train_x, train_y)
        print(f"Retrieval library: n_train={self.rt.n_train} periods={self.period_num}")
        print("Doing Train Retrieval")
        train_rt = self.rt.retrieve_all(
            train_x, train_idx, train=True, batch_size=batch_size, device=device
        )
        print("Doing Valid Retrieval")
        valid_rt = self.rt.retrieve_all(
            valid_x, valid_idx, train=False, batch_size=batch_size, device=device
        )
        print("Doing Test Retrieval")
        test_rt = self.rt.retrieve_all(
            test_x, test_idx, train=False, batch_size=batch_size, device=device
        )
        del self.rt
        self.retrieval_dict["train"] = train_rt.detach()
        self.retrieval_dict["valid"] = valid_rt.detach()
        self.retrieval_dict["test"] = test_rt.detach()

    def encoder(self, x: torch.Tensor, index: torch.Tensor, mode: str) -> torch.Tensor:
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        pred_from_retrieval = self.retrieval_dict[mode][:, index].to(x.device)
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            retrieval_pred_list.append(pr)

        retrieval_sum = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        pred = torch.cat([x_pred_from_x, retrieval_sum], dim=1)
        pred = (
            self.linear_pred(pred.permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(bsz, self.pred_len, self.channels)
        )
        return pred + x_offset

    def forward(self, x: torch.Tensor, index: torch.Tensor, mode: str = "train"):
        return self.encoder(x, index, mode)


@torch.no_grad()
def retrieval_identity_forecast(
    retrieval: torch.Tensor, x: torch.Tensor, period_num
) -> torch.Tensor:
    """No-train baseline: mean over periods of compressed-then-repeated ṽ + offset.

    retrieval: [G, B, F, C], x: [B, L, C]
    """
    bsz = x.shape[0]
    x_offset = x[:, -1:, :]
    parts = []
    for i, g in enumerate(period_num):
        pr = retrieval[i]
        pr = pr.reshape(bsz, pr.shape[1] // g, g, pr.shape[2])
        pr = pr[:, :, 0, :]
        pr = pr.repeat_interleave(g, dim=1)
        if pr.shape[1] < retrieval.shape[2]:
            pad = retrieval.shape[2] - pr.shape[1]
            pr = torch.cat([pr, pr[:, -1:, :].repeat(1, pad, 1)], dim=1)
        parts.append(pr[:, : retrieval.shape[2], :])
    agg = torch.stack(parts, dim=0).mean(dim=0)
    return agg + x_offset
