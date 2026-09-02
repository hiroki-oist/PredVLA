"""MTRNN-BC: BC-LSTM に時定数だけを入れたベースライン（2026-08-12）。

## なぜ要るか

表③ A1（`ab_tau_flat`）は「時定数の階層を消すと spatial −68.02pt」を示している。
査読者が真っ先に聞くのは逆向きの問いで、

    「では普通の RNN に時定数を付けたら 80% 前後まで伸びるのか？」

これは既存の subtractive な ablation では答えられない。本ファイルはその additive 側。

## 何を変えたか（BC-LSTM との差は 1 点だけ）

    BC-LSTM     nn.LSTMCell 1 層              （`src/models/baseline_bc_lstm.py`）
    MTRNN-BC   LeakyRNNCell 2 層（τ 可変）   ← ★ここだけ

入力（concat[v, q, l]）・GMM 行動ヘッド・固有感覚ヘッド・損失・scheduled sampling・
vision_stride の扱いはすべて BaselineRNN と同一にしてある。予測符号化の要素
（視覚予測・PB・A→V efference・自由変数 c・ER）は一切入れていない。

## 交絡を割るための 3 条件

    tau_fast=1,  tau_slow=1    素の 2 層 leaky RNN。★LSTM→leaky RNN のセル変更ぶんの基準
    tau_fast=2,  tau_slow=5    PredVLA の行動枝と同じ。★階層あり
    tau_fast=8,  tau_slow=8    遅いが均一。★「階層」と「遅さ」を分ける

τ=1 の条件が要るのは、セルを LSTM から leaky RNN に替えること自体が性能を動かす
可能性があるため。これが無いと「時定数の効果」とセル変更の効果が混ざる。

params は `mtrnn_hidden` で BC-LSTM(674,276) / PredVLA(675,732) に合わせる。
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pc_cells import ActionPCRNN, LeakyRNNCell


class BaselineMTRNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pp = cfg["preprocess"]
        m = cfg["model"]
        self.v_dim = pp["agentview_pca_dim"] * (pp["agentview_grid"] ** 2) + pp["eye_pca_dim"]
        self.q_dim = pp["proprio_dim"]
        self.a_dim = pp["action_dim"]
        self.l_dim = pp["language_pca_dim"] or 384
        self.stride = m["vision_stride"]
        self.hidden = m.get("mtrnn_hidden", 232)
        self.gmm = m.get("action_head", "mlp") == "gmm"
        self.action_tanh = m.get("action_tanh", False)
        self.tau_fast = float(m.get("mtrnn_tau_fast", 2.0))
        self.tau_slow = float(m.get("mtrnn_tau_slow", 5.0))
        # ★言語を上位（遅い）層にも入れるか（2026-08-12 追加）。
        #   既定 False は「言語は下位層に観測と concat」= 最初の実装。
        #   True にすると PredVLA と同じ「遅い層がタスク文脈を直接受け取る」構成になる。
        #   最初の実装では上位層がボトムアップしか受けず、
        #   「遅い層＝タスク文脈」という PredVLA の設計思想を再現できていなかった。
        self.lang_up = bool(m.get("mtrnn_lang_up", False))
        # ★X2（2026-08-12）: 観測を順方向に入れない条件。
        #   PredVLA は観測が順方向の経路に一切入らず、A→V ブリッジで自分の予測
        #   (â, q̂) を送る「生成モデル」。閉ループ BC（BC-LSTM 15.9 / BC-Transformer 24.9 /
        #   MTRNN-BC 13〜18）と開ループ生成（平均軌道 67 / NI0 72.4 / 表① 82.5）の
        #   差が時定数ではなく「観測を順方向に入れるか」で決まっている疑いがある。
        #   ★PC 構造に観測を順方向に入れた PredVLA_2 は 82.5 → ≤44 に落ちた（逆方向の証拠）。
        #   X2 はその 2×2 の残り 1 マス（階層あり・PC なし・観測なし）を埋める。
        #     use_v=False  視覚を入力から外す
        #     self_q=True  固有感覚は毎 step 自分の予測 q̂ を使う（t=0 は 0）
        self.use_v = bool(m.get("mtrnn_use_v", True))
        self.self_q = bool(m.get("mtrnn_self_q", False))

        in_dim = (self.v_dim if self.use_v else 0) + self.q_dim + self.l_dim
        H = self.hidden
        # 下位（速い）: 観測+言語を受け、上位からトップダウンを受ける
        self.low = LeakyRNNCell(H, self.tau_fast)
        self.Wx_low = nn.Linear(in_dim, H, bias=True)
        self.Wtd_low = nn.Linear(H, H, bias=False)
        # 上位（遅い）: 下位からボトムアップ
        self.up = LeakyRNNCell(H, self.tau_slow)
        self.Wbu_up = nn.Linear(H, H, bias=False)
        self.Wl_up = nn.Linear(self.l_dim, H, bias=False) if self.lang_up else None

        # ヘッドは BaselineRNN と同じ実装を流用する（GMM / 固有感覚）
        self._heads = ActionPCRNN(
            q_dim=1, l_dim=1, pb_dim=1, hidden=H,
            tau_fast=1.0, tau_slow=1.0, head_hidden=m["head_hidden"],
            proprio_dim=self.q_dim, action_dim=self.a_dim,
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

    def _inp(self, v_t, q_t, l):
        """入力を組み立てる。use_v=False なら視覚を入れない。"""
        return torch.cat(([v_t] if self.use_v else []) + [q_t, l], dim=-1)

    # --- 1 step ---
    def _step(self, x, h_low, h_up, lu=None):
        h_low = self.low.step(self.Wx_low(x) + self.Wtd_low(h_up), h_low)
        ext_up = self.Wbu_up(h_low)
        if lu is not None:
            ext_up = ext_up + lu          # ★言語を遅い層にも（PredVLA と同じ）
        h_up = self.up.step(ext_up, h_up)
        return h_low, h_up

    # --- rollout 用（Rollout.run_episode の共通インタフェース）---
    def rollout_reset(self, B: int, device):
        # self_q のときは自分の q̂ を持ち回るので状態に足す
        st = (torch.zeros(B, self.hidden, device=device),
              torch.zeros(B, self.hidden, device=device))
        return st + ((torch.zeros(B, self.q_dim, device=device),) if self.self_q else ())

    def rollout_step(self, v_t, q_t, l, state):
        a, _q, st = self.rollout_step_q(v_t, q_t, l, state)
        return a, st

    def rollout_step_q(self, v_t, q_t, l, state):
        lu = self.Wl_up(l) if self.Wl_up is not None else None
        if self.self_q:
            h_low, h_up, q_prev = state
            q_in = q_prev                      # ★観測の q は使わない
        else:
            h_low, h_up = state
            q_in = q_t
        h_low, h_up = self._step(self._inp(v_t, q_in, l), h_low, h_up, lu=lu)
        a, q_hat = self._heads.heads(h_low)
        st = (h_low, h_up) + ((q_hat,) if self.self_q else ())
        return a, q_hat, st

    # --- 学習 ---
    def forward(self, v, q, a, l, burn_in: int,
                lambda_v=1.0, lambda_q=1.0, lambda_a=10.0,
                noise_v=0.0, noise_q=0.0, training=True, ss_prob=0.0,
                use_ss=None, mask=None) -> Dict[str, object]:
        """BaselineRNN.forward と同一。再帰部分だけ 2 層 leaky RNN に替えてある。"""
        B, L, _ = v.shape
        msk = v.new_ones(B, L) if mask is None else mask.to(v.dtype)
        v_in = v + noise_v * torch.randn_like(v) if (training and noise_v > 0) else v
        q_in = q + noise_q * torch.randn_like(q) if (training and noise_q > 0) else q
        if use_ss is None:
            use_ss = training and float(ss_prob) > 0.0
        else:
            use_ss = bool(use_ss) and training

        h_low = v.new_zeros(B, self.hidden)
        h_up = v.new_zeros(B, self.hidden)
        lu = self.Wl_up(l) if self.Wl_up is not None else None   # 時間に依らないので 1 回
        q_hat_prev = None
        loss_q = v.new_zeros(())
        loss_a = v.new_zeros(())
        nq = v.new_zeros(())
        na = v.new_zeros(())
        a_absmean_sum = v.new_zeros(())

        for s in range(L):
            if s == burn_in:
                h_low = h_low.detach(); h_up = h_up.detach()
                if q_hat_prev is not None:
                    q_hat_prev = q_hat_prev.detach()
            in_loss = s >= burn_in
            v_t = v_in[:, (s // self.stride) * self.stride]
            q_input = q_in[:, s]
            if self.self_q:
                # ★学習時も毎 step 自分の予測を使う（t=0 は 0）。detach しない
                #   ことで開ループ生成の誤差が伝播する（PredVLA の A→V と同じ扱い）
                q_input = (q_hat_prev if q_hat_prev is not None
                           else v.new_zeros(B, self.q_dim))
            elif use_ss and in_loss and q_hat_prev is not None:
                mm = (torch.rand(B, 1, device=v.device) < ss_prob).float()
                q_input = mm * q_hat_prev.detach() + (1.0 - mm) * q_input
            h_low, h_up = self._step(self._inp(v_t, q_input, l),
                                     h_low, h_up, lu=lu)
            a_hat, q_hat = self._heads.heads(h_low)
            q_hat_prev = q_hat
            a_absmean_sum = a_absmean_sum + a_hat.abs().mean().detach()

            if in_loss:
                if self.gmm:
                    nll = self._heads.action_nll(h_low, a[:, s], reduce="none")
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
