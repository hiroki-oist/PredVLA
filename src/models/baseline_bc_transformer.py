"""Matched-size causal-Transformer baseline on the SAME frozen features.

Per timestep, concat[v_t, q_t, l] is linearly embedded to d_model; a small
causal TransformerEncoder predicts, per position, the same GMM action head and
next-proprio head used by the PC-RNN / BC-LSTM. Same losses, same noise /
scheduled-sampling recipe, vision held per stride window (5 Hz information).

Size is controlled by model.tf_dmodel / tf_layers / tf_heads (size-agnostic:
tune d_model to parameter-match whatever PC-RNN size is being compared).
Rollout keeps a sliding context of the last `chunk_len` embeddings.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pc_cells import ActionPCRNN


class BCTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pp = cfg["preprocess"]
        m = cfg["model"]
        self.v_dim = pp["agentview_pca_dim"] * (pp["agentview_grid"] ** 2) + pp["eye_pca_dim"]
        self.q_dim = pp["proprio_dim"]
        self.a_dim = pp["action_dim"]
        self.l_dim = pp["language_pca_dim"] or 384
        self.stride = m["vision_stride"]
        self.d = m.get("tf_dmodel", 96)
        self.n_layers = m.get("tf_layers", 3)
        self.n_heads = m.get("tf_heads", 4)
        self.ctx_len = cfg["train"]["burn_in"] + cfg["train"]["loss_len"]
        self.gmm = m.get("action_head", "mlp") == "gmm"
        self.action_tanh = m.get("action_tanh", False)
        # --- action chunking（2026-08-01 追加。既定 1 で従来と完全に同一） ---
        # 1 観測から chunk step 分の行動をまとめて出し、rollout ではその chunk を
        # 開ループで実行してから再観測する。ACT / Diffusion Policy / OpenVLA の標準手法。
        # Diffusion Policy の標準値は action horizon 8 / prediction horizon 16。
        # 行動ヘッドは chunk*a_dim 次元の同時 GMM（K 成分）にする。
        self.chunk = int(m.get("action_chunk", 1))
        self.a_out = self.a_dim * self.chunk

        in_dim = self.v_dim + self.q_dim + self.l_dim
        self.embed = nn.Linear(in_dim, self.d)
        self.pos = nn.Parameter(torch.zeros(1, self.ctx_len, self.d))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d, nhead=self.n_heads, dim_feedforward=2 * self.d,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self._heads = ActionPCRNN(
            q_dim=1, l_dim=1, pb_dim=1, hidden=self.d,
            tau_fast=1.0, tau_slow=1.0, head_hidden=m["head_hidden"],
            proprio_dim=self.q_dim, action_dim=self.a_out,
            action_tanh=self.action_tanh,
            action_head=m.get("action_head", "mlp"),
            gmm_k=m.get("gmm_k", 5), gmm_min_std=m.get("gmm_min_std", 0.01))
        for name in ("lower", "upper", "Wx_low", "Wtd_low", "Wbu_up", "Wp_up",
                     "Wl_low", "Wl_up"):
            if hasattr(self._heads, name):
                setattr(self._heads, name, None)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lang_ctx(self, l: torch.Tensor) -> torch.Tensor:
        return l

    def _encode(self, x):
        """x: (B, T, in_dim) -> (B, T, d) with causal mask."""
        T = x.shape[1]
        h = self.embed(x) + self.pos[:, :T]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.encoder(h, mask=mask, is_causal=True)

    # --- rollout interface (sliding context) ---
    def rollout_reset(self, B: int, device):
        return torch.zeros(B, 0, self.v_dim + self.q_dim + self.l_dim, device=device)

    def rollout_step(self, v_t, q_t, l, state):
        a, _q, state = self.rollout_step_q(v_t, q_t, l, state)
        return a, state

    def rollout_step_q(self, v_t, q_t, l, state):
        """rollout_step と同じだが 1 step 先の固有感覚予測 q̂ も返す。

        開ループ評価（2026-08-01 追加）のため。観測を与えない条件では
        q を自分の q̂ に置き換えて自走させる必要がある。
        rollout_step の数値は変えていない（こちらを呼ぶだけ）。

        chunk>1 のときは a を (chunk, a_dim) にほどいて返す。実行側
        （Rollout._run_episode_generic）が chunk step 分をキャッシュして開ループで流す。
        """
        tok = torch.cat([v_t, q_t, l], dim=-1).unsqueeze(1)
        state = torch.cat([state, tok], dim=1)[:, -self.ctx_len:]
        h = self._encode(state)[:, -1]
        a, q_hat = self._heads.heads(h)
        if self.chunk > 1:
            a = a.view(a.shape[0], self.chunk, self.a_dim)
        return a, q_hat, state

    def forward(self, v, q, a, l, burn_in: int,
                lambda_v=1.0, lambda_q=1.0, lambda_a=10.0,
                noise_v=0.0, noise_q=0.0, training=True, ss_prob=0.0,
                use_ss=None, mask=None) -> Dict[str, object]:
        """mask: (B, L) の 0/1。窓より短いデモの末尾パディングを損失から外す
        (詳細は agent.LegacyPCAgent.forward の docstring)。None は全 1 と同じ。"""
        B, L, _ = v.shape
        # SS のベルヌーイ変数と混同しないよう msk。引数 mask は (B, L) の 0/1。
        msk = v.new_ones(B, L) if mask is None else mask.to(v.dtype)
        v_in = v + noise_v * torch.randn_like(v) if (training and noise_v > 0) else v
        q_in = q + noise_q * torch.randn_like(q) if (training and noise_q > 0) else q
        if use_ss is None:
            use_ss = training and float(ss_prob) > 0.0
        else:
            use_ss = bool(use_ss) and training

        # vision held per stride window (same information rate as PC-RNN)
        idx = (torch.arange(L, device=v.device) // self.stride) * self.stride
        v_held = v_in[:, idx]
        l_seq = l.unsqueeze(1).expand(B, L, l.shape[-1])

        # two-pass scheduled sampling: first pass predicts q_hat sequence,
        # second pass feeds a mixture of observed / predicted proprio.
        x = torch.cat([v_held, q_in, l_seq], dim=-1)
        h = self._encode(x)
        if use_ss:
            with torch.no_grad():
                _, q_hat_seq = self._heads.heads(h.reshape(B * L, -1))
                q_hat_seq = q_hat_seq.reshape(B, L, -1)
            # q_hat at position t-1 predicts q_t
            q_mix = q_in.clone()
            ss_m = (torch.rand(B, L - 1, 1, device=v.device) < ss_prob).float()
            q_mix[:, 1:] = ss_m * q_hat_seq[:, :-1] + (1 - ss_m) * q_in[:, 1:]
            x = torch.cat([v_held, q_mix, l_seq], dim=-1)
            h = self._encode(x)

        hf = h.reshape(B * L, -1)
        a_hat, q_hat = self._heads.heads(hf)
        a_hat = a_hat.reshape(B, L, -1)
        q_hat = q_hat.reshape(B, L, -1)

        sl = slice(burn_in, L)
        wa = msk[:, sl].reshape(-1)                       # 行動損失の重み (B*len,)
        # chunk>1 なら位置 t の教師は a[t : t+chunk] を連結したもの。
        # 系列末尾は最後の行動で埋め、その位置は mask で損失から外れる。
        if self.chunk > 1:
            pad = a[:, -1:].expand(B, self.chunk - 1, self.a_dim)
            a_pad = torch.cat([a, pad], dim=1)                       # (B, L+chunk-1, A)
            a_tgt = torch.stack([a_pad[:, k:k + L] for k in range(self.chunk)],
                                dim=2).reshape(B, L, self.a_out)
        else:
            a_tgt = a
        if self.gmm:
            nll = self._heads.action_nll(h[:, sl].reshape(-1, self.d),
                                         a_tgt[:, sl].reshape(-1, self.a_out),
                                         reduce="none")
        else:
            nll = F.huber_loss(a_hat[:, sl], a_tgt[:, sl],
                               reduction="none").mean(-1).reshape(-1)
        loss_a = (nll * wa).sum() / wa.sum().clamp(min=1.0)
        eps_q = q[:, burn_in + 1:] - q_hat[:, burn_in:-1]
        wq = msk[:, burn_in + 1:]                         # 目標は q_{t+1}
        loss_q = ((eps_q.pow(2).sum(-1) / self.q_dim) * wq).sum() / wq.sum().clamp(min=1.0)
        zero = v.new_zeros(())
        total = lambda_q * loss_q + lambda_a * loss_a
        return {
            "total": total, "loss_v": zero, "loss_q": loss_q, "loss_a": loss_a,
            "fe": loss_q.detach(), "pb_norm": zero,
            "a_absmean": a_hat[:, sl].abs().mean().detach(),
            "a_target_absmean": (a[:, sl].abs().mean(-1) * msk[:, sl]).sum()
            / msk[:, sl].sum().clamp(min=1.0),
        }


# 旧名（公開 ckpt の cfg や古いスクリプトのため残す）
BaselineTransformer = BCTransformer
