"""PC-RNN building blocks (Phase 1, §4).

- LeakyRNNCell: MTRNN-style leaky cell with time constant tau (optional FiLM).
- VisionPCRNN: 1-layer slow RNN that predicts the next-vision-step latent.
- PBGenerator: maps (vision state, vision error) -> Parametric Bias.
- ActionPCRNN: hierarchical 2-layer MTRNN. Upper (slow) receives PB and encodes
  "what am I doing"; lower (fast) receives proprio + top-down and emits action.
- LanguageTop: static CERNet-style top state s = f(l), computed once per
  episode; conditions Vision and Action-upper top-down (lang_inject top_*).

See DECISIONS.md D4-1..3 for the deviations from the doc:
  * PB injected into the UPPER action layer only.
  * action/proprio losses do NOT backprop into the vision RNN (PB fed detached).
  * action RNN input is the proprioceptive observation q_t only.

Language injection modes (model.lang_inject):
  * additive   — l added (via a linear map) to the ext input of Vision and BOTH
                 action layers every step (original Phase-1 wiring).
  * top_static — a static top state s = MLP(l) sits above the hierarchy and is
                 added (linear) to the ext of Vision and Action-UPPER only.
                 The lower action layer receives no direct language.
  * top_film   — same topology as top_static, but s modulates the cell
                 pre-activation of Vision and Action-upper via FiLM
                 (gamma near 1, beta near 0 at init). No language to lower.
"""
import torch
import torch.nn as nn


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out)
    )


class LeakyRNNCell(nn.Module):
    """h_t = (1 - 1/tau) h_{t-1} + (1/tau) tanh(FiLM(ext + W_h h_{t-1} + b))."""

    def __init__(self, hidden: int, tau: float):
        super().__init__()
        assert tau >= 1.0
        self.hidden = hidden
        self.tau = float(tau)
        self.W_h = nn.Linear(hidden, hidden, bias=True)
        nn.init.orthogonal_(self.W_h.weight, gain=0.9)
        nn.init.zeros_(self.W_h.bias)

    def step(self, ext: torch.Tensor, h_prev: torch.Tensor,
             gamma: torch.Tensor = None, beta: torch.Tensor = None) -> torch.Tensor:
        pre = ext + self.W_h(h_prev)
        if gamma is not None:
            pre = gamma * pre + beta
        leak = 1.0 / self.tau
        return (1.0 - leak) * h_prev + leak * torch.tanh(pre)


def film_layer(ctx_dim: int, hidden: int) -> nn.Linear:
    """Linear producing [d_gamma | beta]; zero-init -> identity modulation."""
    f = nn.Linear(ctx_dim, 2 * hidden)
    nn.init.zeros_(f.weight)
    nn.init.zeros_(f.bias)
    return f


def film_params(f: nn.Linear, ctx: torch.Tensor, hidden: int):
    gb = f(ctx)
    return 1.0 + gb[..., :hidden], gb[..., hidden:]


class LanguageTop(nn.Module):
    """Static top state s = MLP(l) (CERNet-style class embedding). Computed
    once per sequence; time-invariant."""

    def __init__(self, l_dim: int, s_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(l_dim, hidden, s_dim)
        self.s_dim = s_dim

    def forward(self, l: torch.Tensor) -> torch.Tensor:
        return self.net(l)


class VisionPCRNN(nn.Module):
    """Slow vision RNN: consumes v_t, predicts the next vision-step latent.

    `ctx` in step() is the language latent l (additive) or the static top
    state s (top_static / top_film)."""

    def __init__(self, v_dim: int, l_dim: int, hidden: int, tau: float,
                 head_hidden: int, lang_inject: str = "additive", ctx_dim: int = None,
                 a_dim: int = 0, q_dim: int = 0):
        """a_dim>0 / q_dim>0 で「自分がどう動いたか」を視覚 RNN に渡す(2026-07-30 追加)。

        動機: 視覚 RNN は「次の視覚を予測する」順モデルなのに、行動も固有感覚も
        受け取っていなかった(入力は v_t と言語だけ、しかも言語はエピソード内で定数)。
        情報は 視覚 -> PB -> 行動 の一方向にしか流れず、行動 -> 視覚 の経路が無い
        (さらに _pb の detach で勾配も戻らない)。
        192 次元のうち 64 次元は手先カメラで、見えているものはほぼ全部自分の動きで
        決まるので、行動を知らずに予測するのは原理的に無理。
        実測との整合: 閉ループ eps_v 0.74-0.89 が下がらず、窓推論による改善は -3.9%。
        自由エネルギーの 77% を占める視覚項が「原理的に予測できないもの」だった。

        a_dim: 直前 stride step の実行行動(平均)を渡す。efference copy。
               「これからこう動く」という因果的に正しい情報。将来 他者モデルの
               予測行動を入れる拡張もこの経路に乗る。
        q_dim: 現在の固有感覚を渡す。既に動いた結果だが、q から手先位置が
               0.42cm 誤差で決まる(実測)ので手先カメラの見え方に直結する。
        """
        super().__init__()
        self.cell = LeakyRNNCell(hidden, tau)
        self.W_x = nn.Linear(v_dim, hidden, bias=False)
        self.W_a = nn.Linear(a_dim, hidden, bias=False) if a_dim > 0 else None
        self.W_q = nn.Linear(q_dim, hidden, bias=False) if q_dim > 0 else None
        self.lang_inject = lang_inject
        if lang_inject == "additive":
            self.W_l = nn.Linear(l_dim, hidden, bias=False)
        elif lang_inject == "top_static":
            self.W_s = nn.Linear(ctx_dim, hidden, bias=False)
        elif lang_inject == "top_film":
            self.film = film_layer(ctx_dim, hidden)
        else:
            raise ValueError(f"unknown lang_inject: {lang_inject}")
        self.head = mlp(hidden, head_hidden, v_dim)
        self.hidden = hidden

    def lang_term(self, ctx: torch.Tensor):
        """時間に依らない言語項を 1 回だけ計算して返す(高速化用。additive のみ)。
        ctx はシーケンス内で一定なので、毎 step 射影し直す必要がない。"""
        return self.W_l(ctx) if self.lang_inject == "additive" else None

    def _motor(self, a_ctx, q_ctx):
        """自分の動きの項。どちらも無効なら 0 を返す(既定 = 従来と同一)。"""
        t = None
        if self.W_a is not None and a_ctx is not None:
            t = self.W_a(a_ctx)
        if self.W_q is not None and q_ctx is not None:
            t = self.W_q(q_ctx) if t is None else t + self.W_q(q_ctx)
        return t

    def step(self, v_t: torch.Tensor, ctx: torch.Tensor, h_prev: torch.Tensor,
             lang: torch.Tensor = None, a_ctx: torch.Tensor = None,
             q_ctx: torch.Tensor = None) -> torch.Tensor:
        mt = self._motor(a_ctx, q_ctx)
        if self.lang_inject == "additive":
            lt = self.W_l(ctx) if lang is None else lang
            ext = self.W_x(v_t) + lt
            return self.cell.step(ext if mt is None else ext + mt, h_prev)
        if self.lang_inject == "top_static":
            ext = self.W_x(v_t) + self.W_s(ctx)
            return self.cell.step(ext if mt is None else ext + mt, h_prev)
        g, b = film_params(self.film, ctx, self.hidden)
        ext = self.W_x(v_t)
        return self.cell.step(ext if mt is None else ext + mt, h_prev, gamma=g, beta=b)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class PBGenerator(nn.Module):
    """PB_t = MLP([h^v, eps_v]).  Inputs are detached by the caller so no
    gradient reaches the vision RNN (D4-2)."""

    def __init__(self, vision_hidden: int, v_dim: int, pb_dim: int, pb_hidden: int):
        super().__init__()
        self.net = mlp(vision_hidden + v_dim, pb_hidden, pb_dim)

    def forward(self, h_v: torch.Tensor, eps_v: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_v, eps_v], dim=-1))


class ForwardDynamics(nn.Module):
    """Δq = f(q_t, a_t) を学習する、OSC + 物理の微分可能な代理モデル。

    なぜ必要か(2026-07-29): 従来の固有感覚予測 q_hat = proprio_head(h_low) は行動 a を
    入力に取らないので、順運動学ではなく「デモで見た関節角パターンの記憶」だった。
    閉ループでは「動かない」と予測するより数十倍悪い(skill -5 〜 -75)。
    行動は 7 次元の OSC 指令(手先ポーズの目標変位)で、そこから関節角への写像は
    逆運動学 + インピーダンス制御 + 物理積分を通る。それを小さな MLP で近似する。

    実測の当てはまり(libero_spatial, 500デモ 61,750ペア, z-score 空間の MSE/dim):
      「動かない」(Δq=0)         0.003249
      リッジ回帰 Δq ~ q_t         0.002589
      リッジ回帰 Δq ~ [q_t, a_t]  0.001068   ← 線形でも現行 proprio ヘッド(0.00761)の 7 倍良い
      リッジ回帰 + 交差項         0.000770
    つまり 1 step 変位は行動からほぼ決まるので、MLP で十分に近似できる。

    退化の防止: proprio 損失と同時に学習すると「a を無視して q_t だけから予測する」解に
    落ちうる(それでは現状に戻る)。デモの (q_t, a_t, q_{t+1}) だけで事前学習して凍結する。
    """

    def __init__(self, q_dim: int, a_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(q_dim + a_dim, hidden, q_dim)
        self.q_dim = q_dim
        self.a_dim = a_dim

    def forward(self, q_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """-> q_{t+1} の予測。残差形（Δq を出して足す）にして恒等写像を初期解にする。"""
        return q_t + self.net(torch.cat([q_t, a_t], dim=-1))


class ActionPCRNN(nn.Module):
    """Hierarchical 2-layer action MTRNN.

    lower (fast, tau_fast): inputs q_t, top-down h_upper (+ l if additive)
                            -> action/proprio heads
    upper (slow, tau_slow): inputs bottom-up h_lower, PB (+ language context)
                            -> subtask context
    """

    def __init__(self, q_dim: int, l_dim: int, pb_dim: int, hidden: int,
                 tau_fast: float, tau_slow: float, head_hidden: int,
                 proprio_dim: int, action_dim: int, action_tanh: bool = False,
                 lang_inject: str = "additive", ctx_dim: int = None,
                 action_head: str = "mlp", gmm_k: int = 5, gmm_min_std: float = 0.01,
                 chunk_len: int = 1, use_proprio_head: bool = True):
        super().__init__()
        self.hidden = hidden
        self.action_tanh = action_tanh
        self.lang_inject = lang_inject
        self.head_type = action_head
        self.chunk = chunk_len            # >1: predict a chunk of future actions
        self.action_dim = action_dim * chunk_len   # head output width
        self.a_dim_single = action_dim
        self.gmm_k = gmm_k
        self.gmm_min_std = gmm_min_std
        self.lower = LeakyRNNCell(hidden, tau_fast)
        self.upper = LeakyRNNCell(hidden, tau_slow)
        # lower inputs
        self.Wx_low = nn.Linear(q_dim, hidden, bias=False)
        self.Wtd_low = nn.Linear(hidden, hidden, bias=False)   # top-down from upper
        # upper inputs
        self.Wbu_up = nn.Linear(hidden, hidden, bias=False)    # bottom-up from lower
        self.Wp_up = nn.Linear(pb_dim, hidden, bias=False)     # PB injection (upper only)
        if lang_inject == "additive":
            self.Wl_low = nn.Linear(l_dim, hidden, bias=False)
            self.Wl_up = nn.Linear(l_dim, hidden, bias=False)
        elif lang_inject == "top_static":
            self.Ws_up = nn.Linear(ctx_dim, hidden, bias=False)
        elif lang_inject == "top_film":
            self.film_up = film_layer(ctx_dim, hidden)
        else:
            raise ValueError(f"unknown lang_inject: {lang_inject}")
        # heads (from lower/fast layer)
        if action_head == "mlp":
            self.action_head = mlp(hidden, head_hidden, self.action_dim)
        elif action_head == "gmm":
            # per component: mixture logit + mean + log-scale (robomimic-style
            # GMM policy head; fixes mode averaging of multimodal demos)
            self.action_head = mlp(hidden, head_hidden, gmm_k * (1 + 2 * self.action_dim))
        else:
            raise ValueError(f"unknown action_head: {action_head}")
        # proprio_from_action=True のときは呼ばれないので作らない(100,873 パラメータの死重
        # を避ける。384->256->9 の MLP)。代わりに ForwardDynamics(3.7k, 凍結)を使う。
        self.proprio_head = mlp(hidden, head_hidden, proprio_dim) if use_proprio_head else None

    def lang_terms(self, ctx):
        """時間に依らない言語項(下位/上位)を 1 回だけ計算して返す(高速化用)。"""
        if self.lang_inject == "additive":
            return self.Wl_low(ctx), self.Wl_up(ctx)
        return None, None

    def proprio(self, h_low):
        """固有感覚予測のみ。scheduled sampling で毎 step 必要だが、行動ヘッド(GMM)は
        損失計算時に一括で回せるので分離する(高速化用)。"""
        return self.proprio_head(h_low)

    def step(self, q_t, ctx, pb_t, h_low_prev, h_up_prev, lang=None, pb_proj=None):
        """lang: (Wl_low(ctx), Wl_up(ctx)) を事前計算したもの。
        pb_proj: Wp_up(pb_t) を事前計算したもの(PB は 4 step の窓内で線形補間なので、
        アンカー 2 本を射影して混ぜれば毎 step 射影する必要がない)。"""
        if self.lang_inject == "additive":
            ll, lu = self.lang_terms(ctx) if lang is None else lang
            pj = self.Wp_up(pb_t) if pb_proj is None else pb_proj
            ext_low = self.Wx_low(q_t) + self.Wtd_low(h_up_prev) + ll
            ext_up = self.Wbu_up(h_low_prev) + pj + lu
            h_low = self.lower.step(ext_low, h_low_prev)
            h_up = self.upper.step(ext_up, h_up_prev)
        else:
            ext_low = self.Wx_low(q_t) + self.Wtd_low(h_up_prev)
            ext_up = self.Wbu_up(h_low_prev) + self.Wp_up(pb_t)
            h_low = self.lower.step(ext_low, h_low_prev)
            if self.lang_inject == "top_static":
                h_up = self.upper.step(ext_up + self.Ws_up(ctx), h_up_prev)
            else:  # top_film
                g, b = film_params(self.film_up, ctx, self.hidden)
                h_up = self.upper.step(ext_up, h_up_prev, gamma=g, beta=b)
        return h_low, h_up

    def _gmm_params(self, h_low):
        """-> logits (B,K), mu (B,K,A), std (B,K,A). tanh bounds mu if action_tanh."""
        B = h_low.shape[0]
        K, A = self.gmm_k, self.action_dim
        out = self.action_head(h_low).view(B, K, 1 + 2 * A)
        logits = out[..., 0]
        mu = out[..., 1:1 + A]
        if self.action_tanh:
            mu = torch.tanh(mu)
        std = nn.functional.softplus(out[..., 1 + A:]) + self.gmm_min_std
        return logits, mu, std

    def heads(self, h_low):
        """-> (point action, q_hat). For gmm the point action is the mean of the
        highest-weight component (used for rollout and logging).
        proprio_head が無い構成(proprio_from_action)では q_hat は None を返す。"""
        if self.head_type == "mlp":
            a = self.action_head(h_low)
            if self.action_tanh:
                a = torch.tanh(a)
        else:
            logits, mu, _ = self._gmm_params(h_low)
            best = logits.argmax(dim=-1)                       # (B,)
            a = mu[torch.arange(mu.shape[0], device=mu.device), best]
        return a, (self.proprio_head(h_low) if self.proprio_head is not None else None)

    def action_nll(self, h_low, target, reduce: str = "mean"):
        """GMM negative log-likelihood of target actions (B,A).
        Normalized per action dim so its scale matches the per-dim mse of
        loss_v / loss_q (keeps the multi-task gradient balance sane).
        reduce="mean" -> scalar (既定, 従来と同一)。
        reduce="none" -> (B,) の未縮約値。パディング位置をマスクするために使う。
          注意: NLL は「誤差」ではないので target に自分の予測を入れても 0 にならず、
          std を縮め混合重みを1成分に集める勾配が残る。ゼロ寄与にしたい位置は
          必ずこの未縮約値に 0/1 マスクを掛けて和から外すこと。"""
        logits, mu, std = self._gmm_params(h_low)
        log_w = torch.log_softmax(logits, dim=-1)              # (B,K)
        z = (target.unsqueeze(1) - mu) / std
        log_comp = (-0.5 * z.pow(2) - std.log()
                    - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))).sum(-1)  # (B,K)
        nll = -(torch.logsumexp(log_w + log_comp, dim=-1)) / self.action_dim  # (B,)
        return nll if reduce == "none" else nll.mean()
