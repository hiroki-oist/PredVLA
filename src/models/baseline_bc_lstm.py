"""Matched-size BC-LSTM baseline (robomimic-style) on the SAME frozen features.

Single LSTM over concat[v_t, q_t, l] with the same GMM action head, same
losses where applicable (action NLL + next-proprio mse), same noise /
scheduled-sampling recipe. No vision prediction, no PB, no dual timescale —
i.e. the ablation of everything PC-specific. Param-matched via bc_lstm_hidden.

Vision is held constant within a vision_stride window (same information rate
as the PC-RNN's 5 Hz vision loop) in both training and rollout.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pc_cells import mlp, ActionPCRNN


class BCLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pp = cfg["preprocess"]
        m = cfg["model"]
        self.v_dim = pp["agentview_pca_dim"] * (pp["agentview_grid"] ** 2) + pp["eye_pca_dim"]
        self.q_dim = pp["proprio_dim"]
        self.a_dim = pp["action_dim"]
        self.l_dim = pp["language_pca_dim"] or 384
        self.stride = m["vision_stride"]
        self.hidden = int(m.get("bc_lstm_hidden", m.get("bcrnn_hidden", 150)))
        self.gmm = m.get("action_head", "mlp") == "gmm"
        self.action_tanh = m.get("action_tanh", False)

        in_dim = self.v_dim + self.q_dim + self.l_dim
        self.cell = nn.LSTMCell(in_dim, self.hidden)
        # reuse the head implementation from ActionPCRNN via a tiny shim
        self._heads = ActionPCRNN(
            q_dim=1, l_dim=1, pb_dim=1, hidden=self.hidden,
            tau_fast=1.0, tau_slow=1.0, head_hidden=m["head_hidden"],
            proprio_dim=self.q_dim, action_dim=self.a_dim,
            action_tanh=self.action_tanh,
            action_head=m.get("action_head", "mlp"),
            gmm_k=m.get("gmm_k", 5), gmm_min_std=m.get("gmm_min_std", 0.01))
        # drop the recurrent parts of the shim; only its heads are used
        for name in ("lower", "upper", "Wx_low", "Wtd_low", "Wbu_up", "Wp_up",
                     "Wl_low", "Wl_up"):
            if hasattr(self._heads, name):
                setattr(self._heads, name, None)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lang_ctx(self, l: torch.Tensor) -> torch.Tensor:
        return l

    # --- generic rollout interface (used by Rollout.run_episode) ---
    def rollout_reset(self, B: int, device):
        return (torch.zeros(B, self.hidden, device=device),
                torch.zeros(B, self.hidden, device=device))

    def rollout_step(self, v_t, q_t, l, state):
        a, _q, st = self.rollout_step_q(v_t, q_t, l, state)
        return a, st

    def rollout_step_q(self, v_t, q_t, l, state):
        """rollout_step と同じだが 1 step 先の固有感覚予測 q̂ も返す。

        開ループ評価（2026-08-01 追加）のため。観測を与えない条件では
        q を自分の q̂ に置き換えて自走させる必要がある。
        rollout_step の数値は変えていない（こちらを呼ぶだけ）。
        """
        h, c = self.cell(torch.cat([v_t, q_t, l], dim=-1), state)
        a, q_hat = self._heads.heads(h)
        return a, q_hat, (h, c)

    def forward(self, v, q, a, l, burn_in: int,
                lambda_v=1.0, lambda_q=1.0, lambda_a=10.0,
                noise_v=0.0, noise_q=0.0, training=True, ss_prob=0.0,
                use_ss=None, mask=None) -> Dict[str, object]:
        """mask: (B, L) の 0/1。窓より短いデモの末尾パディングを損失から外す
        (詳細は agent.LegacyPCAgent.forward の docstring)。None は全 1 と同じ。"""
        B, L, _ = v.shape
        msk = v.new_ones(B, L) if mask is None else mask.to(v.dtype)
        v_in = v + noise_v * torch.randn_like(v) if (training and noise_v > 0) else v
        q_in = q + noise_q * torch.randn_like(q) if (training and noise_q > 0) else q
        if use_ss is None:
            use_ss = training and float(ss_prob) > 0.0
        else:
            use_ss = bool(use_ss) and training

        h = v.new_zeros(B, self.hidden)
        c = v.new_zeros(B, self.hidden)
        q_hat_prev = None
        loss_q = v.new_zeros(())
        loss_a = v.new_zeros(())
        nq = v.new_zeros(())
        na = v.new_zeros(())
        a_absmean_sum = v.new_zeros(())

        for s in range(L):
            if s == burn_in:
                h = h.detach(); c = c.detach()
                if q_hat_prev is not None:
                    q_hat_prev = q_hat_prev.detach()
            in_loss = s >= burn_in
            # vision held constant within the stride window (5 Hz information)
            v_t = v_in[:, (s // self.stride) * self.stride]
            q_input = q_in[:, s]
            if use_ss and in_loss and q_hat_prev is not None:
                m = (torch.rand(B, 1, device=v.device) < ss_prob).float()
                q_input = m * q_hat_prev.detach() + (1.0 - m) * q_input
            h, c = self.cell(torch.cat([v_t, q_input, l], dim=-1), (h, c))
            a_hat, q_hat = self._heads.heads(h)
            q_hat_prev = q_hat
            a_absmean_sum = a_absmean_sum + a_hat.abs().mean().detach()

            if in_loss:
                if self.gmm:
                    nll = self._heads.action_nll(h, a[:, s], reduce="none")
                else:
                    nll = F.huber_loss(a_hat, a[:, s], reduction="none").mean(dim=-1)
                loss_a = loss_a + (nll * msk[:, s]).sum()
                na = na + msk[:, s].sum()
                if s + 1 < L:
                    e = (q[:, s + 1] - q_hat).pow(2).sum(-1) / self.q_dim
                    loss_q = loss_q + (e * msk[:, s + 1]).sum()
                    nq = nq + msk[:, s + 1].sum()

        loss_q = loss_q / nq.clamp(min=1.0)
        loss_a = loss_a / na.clamp(min=1.0)
        total = lambda_q * loss_q + lambda_a * loss_a
        zero = v.new_zeros(())
        return {
            "total": total, "loss_v": zero, "loss_q": loss_q, "loss_a": loss_a,
            "fe": loss_q.detach(), "pb_norm": zero,
            "a_absmean": a_absmean_sum / L,
            "a_target_absmean": (a[:, burn_in:].abs().mean(-1) * msk[:, burn_in:]).sum()
            / msk[:, burn_in:].sum().clamp(min=1.0),
        }


# 旧名（公開 ckpt の cfg や古いスクリプトのため残す）
BaselineRNN = BCLSTM
