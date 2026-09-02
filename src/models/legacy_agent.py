"""PC-RNN VLA agent (Phase 1, §4): ties Vision + PB + Action together and runs
the asymmetric dual-timescale loop over a chunk.

forward() consumes a chunk (B, L, dim) and returns a dict of scalar losses plus
logging aux. Time is looped in Python (L ~ 70) which is fine for these small
RNNs. Vision updates every `vision_stride` action steps; PB is linearly
interpolated between vision anchors (causal: ramps from the previous anchor to
the newly computed one across the window).
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pc_cells import (VisionPCRNN, PBGenerator, ActionPCRNN, LanguageTop,
                              ForwardDynamics)


class LegacyPCAgent(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pp = cfg["preprocess"]
        m = cfg["model"]
        self.v_dim = pp["agentview_pca_dim"] * (pp["agentview_grid"] ** 2) + pp["eye_pca_dim"]
        self.q_dim = pp["proprio_dim"]
        self.a_dim = pp["action_dim"]
        # language input dim: PCA dim, or 384 for raw MiniLM (language_pca_dim 0)
        raw_l_dim = pp["language_pca_dim"] or 384
        # optional learned projection (Linear + LayerNorm) of the language input
        # before injection — the sane path for raw 384-d MiniLM.
        if m.get("lang_proj") == "learned":
            proj_dim = m.get("lang_proj_dim", 32)
            self.lang_proj = nn.Sequential(nn.Linear(raw_l_dim, proj_dim),
                                           nn.LayerNorm(proj_dim))
            self.l_dim = proj_dim
        else:
            self.lang_proj = None
            self.l_dim = raw_l_dim
        self.stride = m["vision_stride"]
        self.pb_dim = m["pb_dim"]

        # language injection mode (see pcrnn.py docstring): additive (Phase-1
        # wiring) or CERNet-style static top state (top_static / top_film).
        self.lang_inject = m.get("lang_inject", "additive")
        ctx_dim = None
        if self.lang_inject != "additive":
            ctx_dim = m.get("lang_top_dim", 64)
            self.lang_top = LanguageTop(self.l_dim, ctx_dim, m["pb_hidden"])

        # 視覚 RNN に「自分がどう動いたか」を渡す(2026-07-30)。既定 False は従来と同一。
        #   vision_use_action : 直前 stride step の実行行動の平均(efference copy)
        #   vision_use_proprio: 現在の固有感覚 q_t
        # 視覚 RNN は次の視覚を予測する順モデルなのに、これまで行動も固有感覚も
        # 受け取っていなかった(詳細は VisionPCRNN の docstring)。
        self.vis_use_a = bool(m.get("vision_use_action", False))
        self.vis_use_q = bool(m.get("vision_use_proprio", False))
        self.vision = VisionPCRNN(self.v_dim, self.l_dim, m["vision_hidden"],
                                  m["vision_tau"], m["head_hidden"],
                                  lang_inject=self.lang_inject, ctx_dim=ctx_dim,
                                  a_dim=self.a_dim if self.vis_use_a else 0,
                                  q_dim=self.q_dim if self.vis_use_q else 0)
        self.pb_gen = PBGenerator(m["vision_hidden"], self.v_dim, m["pb_dim"], m["pb_hidden"])
        self.chunk = m.get("chunk_len", 1)
        self.action = ActionPCRNN(self.q_dim, self.l_dim, m["pb_dim"], m["action_hidden"],
                                  m["action_tau_fast"], m["action_tau_slow"],
                                  m["head_hidden"], self.q_dim, self.a_dim,
                                  action_tanh=m.get("action_tanh", False),
                                  lang_inject=self.lang_inject, ctx_dim=ctx_dim,
                                  action_head=m.get("action_head", "mlp"),
                                  gmm_k=m.get("gmm_k", 5),
                                  gmm_min_std=m.get("gmm_min_std", 0.01),
                                  chunk_len=self.chunk,
                                  use_proprio_head=not bool(
                                      m.get("proprio_from_action", False)))
        self.gmm = m.get("action_head", "mlp") == "gmm"
        # train-time windowed PC inference (v1): burn-in becomes an inference
        # window whose initial vision state + PB correction are optimized by
        # K gradient steps on the window free energy (unrolled: weights learn
        # through the inference).
        self.pc_k = m.get("pc_infer_k", 0)
        self.pc_eta = m.get("pc_infer_eta", 0.1)
        self.pc_beta = m.get("pc_infer_beta", 0.1)
        # delta 側の beta を分離できるようにする(既定は pc_infer_beta と同値 = 従来動作)。
        # reduce="sum" にすると h_v0(384次元)と delta(32次元)で Complexity の効き方が
        # 12 倍ずれるため、単一の beta では両方を適正域に置けない。
        # 実測(anchor から 0.05 離れた点での Complexity勾配/Accuracy勾配、beta=0.1):
        #   reduce=mean  h_v0 0.017 / delta 0.050   ← 実質無効
        #   reduce=sum   h_v0 6.709 / delta 1.596   ← 強すぎる
        # 比 0.25 を狙うなら sum で beta_h ≈ 0.004、beta_delta ≈ 0.015。
        self.pc_beta_delta = m.get("pc_infer_beta_delta", None)
        self.pc_vars = m.get("pc_infer_vars", "both")   # both | delta | hv
        # Complexity の縮約。mean が従来動作(次元数で割るので実質無効)、sum が本来の形。
        self.pc_complexity_reduce = m.get("pc_complexity_reduce", "mean")   # mean | sum
        # 内側推論の勾配の取り方。autograd(既定・従来と同一) | func(torch.func.grad)。
        # func は requires_grad_() の葉テンソルを作らないので torch.compile が
        # graph break しない。数値は厳密に一致することを scripts/check_pc_changes.py で確認。
        self.pc_backend = m.get("pc_infer_backend", "autograd")   # autograd | func
        # 対照: 内側ループで潜在変数は推論するが、重みはその推論を貫通して学習しない
        # (create_graph=False)。計算量・推論手続き・データを同一に保ったまま
        # unrolled 勾配の項だけを消すので、「効いているのは unroll か、それとも
        # 単に状態がよく初期化されることか」を切り分けられる。
        self.pc_detach = m.get("pc_infer_detach", False)
        # --- PC 構造の ablation フラグ(2026-07-28 追加。既定はすべて従来動作) ---
        # pb_use_eps=False : PB 生成器に予測誤差 eps_v を渡さない(ゼロで埋める)。
        #   PB = MLP([h_v, eps_v]) の誤差信号経路を切る = 予測符号化の核を外す対照。
        # pb_bypass=True   : PB ボトルネックを使わず h_v を線形射影して行動 RNN に渡す。
        #   32 次元の橋が必要かを見る。
        # pb_interp=step   : アンカー間の線形ランプをやめて段階関数にする。
        self.pb_use_eps = bool(m.get("pb_use_eps", True))
        # pb_detach(2026-07-29 追加。既定 True = 従来動作 = D4-2):
        #   True  : PB 生成器に h_v.detach() を渡す。行動/固有感覚の損失は視覚 RNN に
        #           届かないので、Vision PC-RNN は視覚予測損失だけで学習される。
        #   False : h_v をそのまま渡す。行動損失(GMM NLL)が PB 経由で視覚 RNN に逆伝播する。
        # なぜ試すか: object の 10 タスクはすべて「X を掴んでバスケットに入れる」で運動が
        # ほぼ同一。視覚予測損失は「次フレームの見た目」を当てる課題なので、物体の
        # identity(どれが tomato sauce か)を符号化する圧力が原理的に無い。PB は
        # MLP([h_v.detach(), eps_v.detach()]) なので、行動側から「identity を表現せよ」
        # という勾配も戻れない。detach を外すと行動損失が直接その圧力をかける。
        # eps_v は detach したままにする(変更を 1 機構に限定するため)。
        self.pb_detach = bool(m.get("pb_detach", True))
        # learn_init_state(2026-07-30 追加。既定 False = 従来のゼロ初期化):
        # h_v / h_low / h_up の初期値を学習可能なパラメータにする。
        # 動機: h_v は leaky セル τ=8・5Hz なのでゼロ初期値の残存率が
        #   step 20(5tick) 0.51 / step 40 0.26 / step 80 0.069 と遅く消える。
        # handoff 実験(2026-07-30)で「デモに最初の 20 step をやらせると成功率が
        # 1/8 → 8/8 に跳ぶ」ことが分かり、破綻が冒頭の立ち上がりに限定されると判明した。
        # ゼロは「正しい初期状態」ではないので、学習させる。
        # train.fixed_start=True（全デモを先頭から切り出す）と組み合わせて使う想定。
        # 追加パラメータは vision_hidden + 2*action_hidden = 1,152 個のみ。
        self.learn_init = bool(m.get("learn_init_state", False))
        self.pb_bypass = bool(m.get("pb_bypass", False))
        self.pb_interp = m.get("pb_interp", "ramp")      # ramp | step
        # bptt_through_burnin: pc_k=0 のまま burn-in 境界の detach をしない。
        # H2(2026-07-28)で pcv1/detach の +7pt の実体は「推論」ではなく
        # 「burn-in 窓を貫通して BPTT が伸びること」だと勾配比較で示された
        # (pc_k=0 vs pc_k=5(eta=0) の勾配相対差 0.164 に対し、推論由来は 0.025)。
        # このフラグは pc_k=5 + eta=0 と厳密に同じ勾配を与える(相対差 0.0000、
        # cos 1.000000 を実測)。ただし現行レシピの eta=0.1 とは別条件で、
        # eta=0.1 との勾配相対差は 0.025 ある(成功率への影響は未検証)。
        # したがって「eta=0.1 の高速化」ではなく「わずかな推論に意味があるか」を
        # 問う ablation 用(configs/abl_eta0.yaml)。副作用として内側 autograd ループが
        # 消えるので torch.compile が使えるようになる。
        self.bptt_burnin = bool(m.get("bptt_through_burnin", False))
        # fast_loop: 数値を変えずにループ内のカーネル数を減らす(2026-07-28)。
        #   ・言語項 Wl(ctx) は時間に依らないので 1 回だけ計算して使い回す(3箇所)
        #   ・PB は窓内で線形補間なので、アンカー 2 本を射影して混ぜる
        #     (Wp_up は線形なので Wp_up((1-f)pb_prev + f pb_curr)
        #      = (1-f)Wp_up(pb_prev) + f Wp_up(pb_curr))
        #   ・行動ヘッド(GMM)は毎 step 不要。h_low を貯めて損失計算時に一括で回す
        #     (固有感覚ヘッドは scheduled sampling の入力に使うので毎 step 必要)
        self.fast_loop = bool(m.get("fast_loop", False))
        # --- 固有感覚予測を行動経由にする(2026-07-29 追加。既定 False は従来動作) ---
        # proprio_from_action=True: q_hat = dyn(q_t, a_hat) にする。
        #   従来 q_hat = proprio_head(h_low) は行動を入力に取らないため、q 予測は
        #   順運動学ではなくデモの関節角パターンの記憶で、閉ループでは「動かない」と
        #   予測するより数十倍悪い(skill -5 〜 -75)。行動経由にすると
        #     ・eps_q が本物の 1 step 予測誤差になる
        #     ・eps_q の勾配が a_hat(argmax 成分の mu)に流れ、行動が「結果」で監督される
        #     ・内部推論の delta が意味のある信号で駆動される
        #   dyn は デモの (q_t, a_t, q_{t+1}) で事前学習して凍結する(dyn_freeze)。
        #   同時学習すると a を無視する退化解に落ちうるため。
        self.proprio_from_action = bool(m.get("proprio_from_action", False))
        self.dyn = None
        if self.proprio_from_action:
            self.dyn = ForwardDynamics(self.q_dim, self.a_dim, m.get("dyn_hidden", 128))
            ck = m.get("dyn_ckpt", "")
            if ck:
                import os as _os
                p = ck if _os.path.isabs(ck) else _os.path.join(
                    _os.path.dirname(_os.path.dirname(_os.path.dirname(
                        _os.path.abspath(__file__)))), ck)
                sd = torch.load(p, weights_only=False, map_location="cpu")
                self.dyn.load_state_dict(sd["model"] if "model" in sd else sd)
            if bool(m.get("dyn_freeze", True)):
                for prm in self.dyn.parameters():
                    prm.requires_grad_(False)
                self.dyn.eval()
        if self.pb_bypass:
            self.pb_proj = nn.Linear(m["vision_hidden"], m["pb_dim"])
        self.vision_hidden = m["vision_hidden"]
        self.action_hidden = m["action_hidden"]
        if self.learn_init:
            self.init_hv = nn.Parameter(torch.zeros(m["vision_hidden"]))
            self.init_hl = nn.Parameter(torch.zeros(m["action_hidden"]))
            self.init_hu = nn.Parameter(torch.zeros(m["action_hidden"]))
        else:
            self.init_hv = self.init_hl = self.init_hu = None

        # 初期内部状態エンコーダ(2026-07-30、ユーザー提案)。
        #
        # 動機: 窓の切り出し位置ごとに「その時点の正しい内部状態」は違うのに、
        # これまでの選択肢は (a) ゼロ初期化 (b) 学習した 1 個の定数 しかなかった。
        # (b) は先頭固定(fixed_start)と組み合わせると object t5 で d_min を
        # 32.5 → 2.3cm に縮めたが、開始位置の多様性が 25,250 → 500 に落ちるため
        # goal 10 タスクでは 3.21% (従来レシピ 11.57%) に悪化した。
        # 混合(fixed_start_prob=0.5)は init_hv 1 個に「t=0 用」と「途中用」の
        # 矛盾する 2 役を負わせるので d_min が 31cm に戻った。
        #
        # ここでは h_0 = act(W_v v_0 + W_q q_0 + W_l ctx) を作り、そこから独立な
        # 重みで (h_v, h_low, h_up) の初期値へ写す。入力依存なので窓の位置ごとに
        # 違う初期状態を出せる = 多様性を保ったまま初期状態を与えられる。
        # PC の観点では anchor(Complexity の参照点)が定数から f(obs_0) になる。
        # PV-RNN が A_0 を系列ごとの最適化で求めて新規系列に汎化しないのに対し、
        # これはその amortize 版に当たる。
        #
        # 出力層はゼロ初期化するので、学習開始時点の挙動はゼロ初期化と厳密に一致する
        # (tanh(0)=0)。切り分けのため他の部分は一切変えない。
        self.init_enc = bool(m.get("init_encoder", False))
        if self.init_enc:
            d = int(m.get("init_enc_dim", 64))
            self.init_enc_dim = d
            enc_l_dim = ctx_dim if self.lang_inject != "additive" else self.l_dim
            # 入力ごとに独立な重み。"v"/"q"/"l" の部分集合を指定すると
            # その入力だけを使う(ablation 用。既定 "vql" は 3 つ全部)。
            use = str(m.get("init_enc_inputs", "vql"))
            self.init_enc_use = use
            self.init_enc_v = nn.Linear(self.v_dim, d) if "v" in use else None
            self.init_enc_q = nn.Linear(self.q_dim, d) if "q" in use else None
            self.init_enc_l = nn.Linear(enc_l_dim, d) if "l" in use else None
            # 出力も 3 つ独立な重み
            self.init_out_hv = nn.Linear(d, m["vision_hidden"])
            self.init_out_hl = nn.Linear(d, m["action_hidden"])
            self.init_out_hu = nn.Linear(d, m["action_hidden"])
            for _lin in (self.init_out_hv, self.init_out_hl, self.init_out_hu):
                nn.init.zeros_(_lin.weight)
                nn.init.zeros_(_lin.bias)
            # 各入力を足す前に tanh を通すか。False(既定)だと
            #   W_v v + W_q q + W_l l == [W_v W_q W_l][v;q;l]
            # で連結 1 本の線形と数学的に同一(パラメータ数も同じ)。True にすると
            # 経路ごとに非線形が入り非等価になる。
            self.init_enc_stream_act = bool(m.get("init_enc_stream_act", False))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lang_ctx(self, l: torch.Tensor) -> torch.Tensor:
        """Language context passed to vision/action steps: (projected) l
        (additive) or the static top state s = f(l); constant per sequence."""
        if self.lang_proj is not None:
            l = self.lang_proj(l)
        if self.lang_inject == "additive":
            return l
        return self.lang_top(l)

    def _pb(self, h_v, eps_v):
        """PB アンカーを作る。ablation フラグを一箇所に集約する。
        pb_use_eps=False なら予測誤差の経路をゼロで塞ぐ(重み形状は変えないので
        既存 ckpt と互換)。pb_bypass=True なら PB 生成器を通さず h_v を線形射影する。

        重要(2026-07-29 測定): ここで h_v を detach しているため、PB は h_v から勾配を
        受けない。これが「PredVLA = detach」の detach 本体で、内部推論に次の帰結を持つ。
          視覚予測誤差    -> h_v0 のみに勾配（PB より先に進めない）
          固有感覚予測誤差 -> delta のみに勾配（h_v0 への経路が PB で切れている）
        つまり 2 つの誤差項と 2 つの自由変数は完全に分離している。実測(w200mv100_s0
        @20k, 窓20): Accuracy 6.3915 に対し |dA/dh_v0| 1.07e-2、|dA/ddelta| 1.94e-2。
        推論の主成分は delta 側(h_v0 の 1.8 倍)。
        したがって pc_vars="hv"(内部状態のみ推論)と fe_terms="v"(視覚誤差のみ)は
        同一の条件になる(どちらも delta=0、h_v も一致することを確認済み)。"""
        hv = h_v if not self.pb_detach else h_v.detach()
        if self.pb_bypass:
            return self.pb_proj(hv)
        e = eps_v if self.pb_use_eps else torch.zeros_like(eps_v)
        return self.pb_gen(hv, e.detach())

    def _vis_motor(self, a_seq, q_seq, s: int):
        """視覚 tick 時点 s に渡す (a_ctx, q_ctx)。
        a_ctx: 直前 stride step の実行行動の平均(s=0 ではゼロ)。因果的に
               「これから動く指令」ではなく「直前に動いた指令」になる点は
               efference copy と同じ扱い。
        q_ctx: 現在の固有感覚。"""
        a_ctx = q_ctx = None
        if self.vis_use_a:
            if a_seq is None or s == 0:
                a_ctx = None if a_seq is None else a_seq[:, 0] * 0.0
            else:
                lo = max(0, s - self.stride)
                a_ctx = a_seq[:, lo:s].mean(dim=1)
        if self.vis_use_q and q_seq is not None:
            q_ctx = q_seq[:, s]
        return a_ctx, q_ctx

    def initial_states(self, B: int, ref: torch.Tensor,
                       v0=None, q0=None, ctx=None):
        """(h_v, h_low, h_up) の初期値。ref は device/dtype 合わせ用の参照テンソル。

        優先順位:
          init_encoder=True かつ v0 が渡された  -> h_0 = f(v_0, q_0, ctx) から写す
          learn_init_state=True                 -> 学習した定数
          どちらでもない                        -> 従来どおりゼロ

        v0/q0/ctx は「窓の開始時点」の入力(v_in[:, 0], q_in[:, 0], lang_ctx(l))。
        rollout でもエピソード先頭で同じものが手に入るので学習と評価でズレない。
        """
        if self.init_enc and v0 is not None:
            t = None
            for lin, x in ((self.init_enc_v, v0), (self.init_enc_q, q0),
                           (self.init_enc_l, ctx)):
                if lin is None or x is None:
                    continue
                y = lin(x.to(ref.dtype))
                if self.init_enc_stream_act:
                    y = torch.tanh(y)
                t = y if t is None else t + y
            if t is not None:
                h0 = torch.tanh(t)
                # h は leaky-tanh セルの状態で ±1 の範囲なので出力も tanh で押さえる。
                # 出力層はゼロ初期化なので学習開始時は厳密にゼロ = 従来と一致。
                return (torch.tanh(self.init_out_hv(h0)),
                        torch.tanh(self.init_out_hl(h0)),
                        torch.tanh(self.init_out_hu(h0)))
        if not self.learn_init:
            z = ref.new_zeros
            return (z(B, self.vision_hidden), z(B, self.action_hidden),
                    z(B, self.action_hidden))
        return (self.init_hv.unsqueeze(0).expand(B, -1).to(ref.dtype),
                self.init_hl.unsqueeze(0).expand(B, -1).to(ref.dtype),
                self.init_hu.unsqueeze(0).expand(B, -1).to(ref.dtype))

    def _complexity(self, h_v0, anchor, delta, anchor_d):
        """Complexity 項。pc_complexity_reduce で縮約の仕方を切り替える。

        mean(既定・従来と同一): 次元数(384 / 32)で割られるため、名目 beta=0.1 でも
          勾配は Accuracy 項の 2% しかなく信頼領域として機能しない。
        sum: 次元方向は和、バッチ方向は平均。PV-RNN の KL は次元和なのでこちらが本来。
          ただし h_v0 側は 384 倍・delta 側は 32 倍に効くので beta の再調整が必要
          (実測に基づく推奨値は configs/ 側に記載)。
        """
        dh = (h_v0 - anchor).pow(2)
        dd = (delta - anchor_d).pow(2)
        # beta は pc_infer 側で全体に掛かるので、ここでは delta 側の相対倍率だけ入れる
        r = 1.0 if self.pc_beta_delta is None \
            else float(self.pc_beta_delta) / max(float(self.pc_beta), 1e-12)
        if self.pc_complexity_reduce == "sum":
            return dh.sum(-1).mean() + r * dd.sum(-1).mean()
        return dh.mean() + r * dd.mean()

    def q_pred(self, h_low, q_t, a_hat=None):
        """次の固有感覚の予測を返す。-> (q_hat, a_hat)

        proprio_from_action=False(既定): 従来どおり proprio_head(h_low)。a_hat は
          呼び出し側が渡したものをそのまま返す(None のまま)。
        True: q_hat = dyn(q_t, a_hat)。a_hat が未計算なら GMM の代表点(最大重み成分の
          mu)をここで作る。chunk>1 のときは先頭 1 行動だけを動力学に渡す
          (dyn は 1 step の写像なので)。
        """
        if not self.proprio_from_action:
            return self.action.proprio(h_low), a_hat
        if a_hat is None:
            a_hat, _ = self.action.heads(h_low)
        a1 = a_hat[..., :self.a_dim] if self.chunk > 1 else a_hat
        return self.dyn(q_t, a1), a_hat

    def _pb_mix(self, pb_prev, pb_curr, frac):
        """アンカー間の補間。ramp = 線形(既定)、step = 段階関数。"""
        if self.pb_interp == "step":
            return pb_curr
        return (1.0 - frac) * pb_prev + frac * pb_curr

    def _pc_window(self, v_in, q_in, q, v, ctx, burn_in, h_v0, delta, fe_terms="both",
                   rec=None, a_in=None):
        """Run the burn-in window from free variables (initial vision state
        h_v0, PB correction delta) and return the accumulated **accuracy term**
        plus end-of-window states. Mirrors the main loop's dynamics.

        注意(2026-07-28 訂正): 返り値は自由エネルギーではなく Accuracy 項だけ
        (視覚予測誤差 + 固有感覚予測誤差の窓内総和)。自由エネルギーは
        Accuracy + Complexity で、Complexity は pc_infer 側で beta を掛けて
        足している。ログや過去のメモで "FE" と書いていた数値はすべて
        Accuracy 項のみを指す。

        実測の内訳(pcv1_detach_s0, 窓20, 初期状態):
          視覚予測誤差 2.669 (4 項, 77.3%) / 固有感覚予測誤差 0.784 (19 項, 22.7%)
        自由変数への勾配は |dAcc/dh_v0| 0.0186 に対し |dAcc/ddelta| 0.0040 で、
        PB 補正はほとんど信号を受けていない(FE の 77% が視覚項で、delta は
        PB->行動->固有感覚経路だけを通るため)。"""
        B = v_in.shape[0]
        h_v = h_v0
        _, h_low, h_up = self.initial_states(B, v_in, v_in[:, 0], q_in[:, 0], ctx)
        pb_prev = v_in.new_zeros(B, self.pb_dim)
        pb_curr = pb_prev
        v_pred_prev = None
        q_hat_prev = None
        win_start = 0
        FE = v_in.new_zeros(())
        for s in range(burn_in):
            if s % self.stride == 0:
                if v_pred_prev is None:
                    eps_v = torch.zeros_like(v[:, s])
                else:
                    eps_v = v[:, s] - v_pred_prev
                    if fe_terms in ("both", "v"):
                        FE = FE + (eps_v.pow(2).sum(-1) / self.v_dim).mean()
                if rec is not None and v_pred_prev is not None:
                    # 初回 tick は v_pred_prev が無く誤差が定義できないので記録しない
                    # (誤差 0 として入れると平均を不当に下げる)
                    rec["eps_v"].append((s, float((eps_v.pow(2).sum(-1) / self.v_dim).mean())))
                _ac, _qc = self._vis_motor(a_in, q_in, s)
                h_v = self.vision.step(v_in[:, s], ctx, h_v, a_ctx=_ac, q_ctx=_qc)
                v_pred_prev = self.vision.predict(h_v)
                pb_prev = pb_curr
                pb_curr = self._pb(h_v, eps_v)
                win_start = s
            frac = ((s - win_start) + 1) / self.stride
            pb_t = self._pb_mix(pb_prev, pb_curr, frac) + delta
            h_low, h_up = self.action.step(q_in[:, s], ctx, pb_t, h_low, h_up)
            q_hat, _ = self.q_pred(h_low, q_in[:, s])
            if q_hat_prev is not None and fe_terms in ("both", "q"):
                FE = FE + ((q[:, s] - q_hat_prev).pow(2).sum(-1) / self.q_dim).mean()
            if rec is not None:
                # 窓推論後の「窓内の各 step の値」を記録する(可視化用。方策に影響しない)。
                # eps_q は q_hat_prev（1つ前の予測）と観測 q の差なので s>=1 のみ。
                if q_hat_prev is not None:
                    rec["eps_q"].append(
                        (s, float(((q[:, s] - q_hat_prev).pow(2).sum(-1) / self.q_dim).mean())))
                rec["q_hat"].append((s, q_hat[0].detach().cpu().numpy().copy()))
            q_hat_prev = q_hat
        return FE, (h_v, h_low, h_up, pb_prev, pb_curr, v_pred_prev, q_hat_prev, win_start)

    def pc_infer(self, v_in, q_in, q, v, ctx, burn_in, k=None, create_graph=None,
                 h0=None, fe_terms="both", delta0=None, rec=None, a_in=None):
        """K iterations of window free-energy minimization over (h_v0, delta).
        自由エネルギー E = Accuracy + Complexity:
          Accuracy   = _pc_window() の返り値(窓内の予測誤差の総和)
          Complexity = beta * (||h_v0 - anchor||^2 / D_h + ||delta||^2 / D_pb)
        h0: anchor/init for the window-initial vision state (zeros at episode
        start; the stored window-start state when sliding at rollout).
        Returns optimized delta and the final window end states.

        注意(2026-07-28 測定): Complexity は .mean() で次元数(384 / 32)で割られる
        ため、名目 beta=0.1 でも勾配は Accuracy 項の 2% 程度しかなく、信頼領域として
        機能していない。初期状態(h_v0 = anchor)では厳密にゼロ。つまり実装上は
        「ほぼ制約なしの素の勾配降下」であり、変分自由エネルギー最小化になっていない。
        PV-RNN の Complexity は KL(q(z)||p(z)) だが、ここは潜在が点推定なので KL が
        定義できず anchor への L2 で代用している。beta を実効化するには .sum() 化か
        beta を 100-400 倍する必要がある。

        fe_terms (2026-07-29 追加): Accuracy にどの項を入れるか。
          both = 視覚 + 固有感覚(既定、従来と完全に同一)
          v    = 視覚のみ / q = 固有感覚のみ
        なぜ必要か: 閉ループ rollout での固有感覚予測は「動かない」と予測するより
        数十倍悪い(spatial t0 成功例で skill -5.3、t7 -75.1、object t5 -43.9)。
        proprio ヘッドは proprio_head(h_low) で行動を入力に取らないため、q 予測は
        順運動学ではなくデモで見た関節角パターンの記憶であり、未知の状態列では外れる。
        その誤差が窓20 の Accuracy の 22.7% を占めて潜在状態を更新しているので、
        テスト時に v のみへ切り替えて成功率が上がるかを見る(再学習不要の ablation)。"""
        B = v_in.shape[0]
        k = self.pc_k if k is None else k
        cg = (self.training and not self.pc_detach) if create_graph is None else create_graph
        if h0 is None:
            anchor = self.initial_states(B, v_in, v_in[:, 0], q_in[:, 0], ctx)[0]
            if self.learn_init or self.init_enc:
                # 学習可能な初期値そのものは anchor として使うが、勾配は h_v0 経路から
                # 受けるので detach しない（初期値が「良い anchor」になるよう学習される）
                pass
            else:
                anchor = anchor.detach()
        else:
            anchor = h0.detach()
        h_v0 = anchor.clone()
        # delta の事前値。従来は常に 0 だったが、rollout では tick ごとに推論を回すので
        # 「前回の推論結果」を事前値にしないと毎回 0 からやり直しになり、k=5・eta=0.1 では
        # |delta| が 1e-2 程度しか動けなかった(実測 9.68e-3 = 5*0.1*|dA/ddelta| そのもの)。
        # Complexity は「推論の前後で大きく変わらないこと」を要求すべきもので、
        # 「常に 0 の近くにいること」ではない。
        anchor_d = v_in.new_zeros(B, self.pb_dim) if delta0 is None else delta0.detach()
        delta = anchor_d.clone()
        # pc_infer_vars: どの自由変数を推論するか (ablation)。
        #   both  = h_v0 と delta の両方(既定)
        #   delta = PB 補正のみ(初期視覚状態は anchor 固定)
        #   hv    = 初期視覚状態のみ(PB 補正は 0 固定)
        opt_hv = self.pc_vars in ("both", "hv")
        opt_delta = self.pc_vars in ("both", "delta")
        if self.pc_backend == "func":
            # torch.func.grad 版。requires_grad_() の葉テンソルを作らないので
            # torch.compile が graph break しない("requires_grad_() intermediate
            # leaked as output" が消える)。数値は autograd 版と厳密に一致する
            # (同じ逆伝播を関数変換で呼ぶだけ)。
            # cg=False のときは grad を detach する。autograd.grad(create_graph=False)
            # が返すテンソルはグラフを持たないので、それと揃えるため
            # (torch.func.grad の返り値は既定で重みについて微分可能=cg=True 相当)。
            h_v0 = anchor.clone()
            delta = anchor_d.clone()
            argnums = tuple(i for i, on in ((0, opt_hv), (1, opt_delta)) if on)

            def _energy(h, d):
                FE, _ = self._pc_window(v_in, q_in, q, v, ctx, burn_in, h, d, fe_terms,
                                        a_in=a_in)
                return FE + self.pc_beta * self._complexity(h, anchor, d, anchor_d)

            gfn = torch.func.grad(_energy, argnums=argnums)
            for _ in range(k):
                gs = gfn(h_v0, delta)
                if not isinstance(gs, tuple):
                    gs = (gs,)
                if not cg:
                    gs = tuple(g.detach() for g in gs)
                gi = 0
                if opt_hv:
                    h_v0 = h_v0 - self.pc_eta * gs[gi]; gi += 1
                if opt_delta:
                    delta = delta - self.pc_eta * gs[gi]
        else:
            h_v0 = h_v0.requires_grad_(True)
            delta = delta.requires_grad_(True)
            free = tuple(x for x, on in ((h_v0, opt_hv), (delta, opt_delta)) if on)
            for _ in range(k):
                FE, _ = self._pc_window(v_in, q_in, q, v, ctx, burn_in, h_v0, delta, fe_terms,
                                        a_in=a_in)
                E = FE + self.pc_beta * self._complexity(h_v0, anchor, delta, anchor_d)
                grads = torch.autograd.grad(E, free, create_graph=cg)
                gi = 0
                if opt_hv:
                    h_v0 = h_v0 - self.pc_eta * grads[gi]; gi += 1
                if opt_delta:
                    delta = delta - self.pc_eta * grads[gi]
                free = tuple(x for x, on in ((h_v0, opt_hv), (delta, opt_delta)) if on)
        if not cg:
            # create_graph=False のとき h_v0 / delta は重みへの経路を持たない
            # (anchor は h0.detach() かゼロで、grads も detach 済み)。したがって
            # ここで detach しても重みの勾配は厳密に変わらない。
            # torch.compile 対策: requires_grad_() を持つ中間テンソルが出力に漏れると
            # dynamo が graph break する("requires_grad_() intermediate leaked as
            # output")。detach で 3-4 個の break が消える。
            #
            # 例外(2026-07-30 に発覚): learn_init_state=True では anchor が学習パラメータ
            # init_hv になるので「重みへの経路を持たない」前提が崩れる。ここで detach すると
            # init_hv に勾配が一切来ず、厳密にゼロのまま学習されない(実測で確認)。
            # その場合は h_v0 の detach を外し、推論による移動分だけを切り離す
            # (h_v0 = anchor + (h_v0 - anchor).detach()) ことで
            #   ・init_hv には勾配が流れる
            #   ・推論の 5 反復を貫通する二階項は入らない(pc_detach の意図を保つ)
            # の両方を満たす。
            # init_encoder も anchor が重みへの経路を持つので同じ扱いにする。
            if (self.learn_init or self.init_enc) and opt_hv:
                h_v0 = anchor + (h_v0 - anchor).detach()
            else:
                h_v0 = h_v0.detach()
            delta = delta.detach()
        FE, states = self._pc_window(v_in, q_in, q, v, ctx, burn_in, h_v0, delta, fe_terms,
                                     rec=rec, a_in=a_in)
        return delta, states

    def forward(self, v, q, a, l, burn_in: int,
                lambda_v=1.0, lambda_q=1.0, lambda_a=10.0,
                noise_v=0.0, noise_q=0.0, training=True, ss_prob=0.0,
                use_ss=None, mask=None) -> Dict[str, object]:
        """ss_prob: scheduled-sampling probability (§9-3). During the loss region,
        with this per-element probability the model's OWN prediction (q_hat / v_hat)
        is fed as input instead of the dataset value, exposing it to its own error
        accumulation to fight closed-loop covariate shift. Loss targets stay clean.

        mask: (B, L) の 0/1。窓より短いデモを末尾ゼロ埋めで入れたときのパディング位置を
        損失から外す。None は全 1 と同じ(従来と数値一致)。
        マスクは「その位置の項を損失の和に入れない」ことで実現する。0 を掛けてから
        和を取るので ∂(0*l)/∂θ = 0 となり、重み更新への寄与が厳密にゼロになる。
        (target に自分の予測を入れる方法では GMM NLL がゼロにならず、std を縮めて
         混合を1成分に潰す勾配が残るため使えない。)
        パディングは窓の末尾にのみ現れるので、再帰状態が埋め草区間で壊れても
        以降は全てマスクされており影響しない。デモ長 < burn_in は非対応
        (全スイートで最短 75 frame > burn_in 20 なので実際には起こらない)。"""
        B, L, _ = v.shape
        # 変数名は msk。この関数内の m は scheduled sampling のベルヌーイマスクで別物。
        msk = v.new_ones(B, L) if mask is None else mask.to(v.dtype)
        ctx = self.lang_ctx(l)                     # static over the chunk
        # noised inputs (train only); targets stay clean (§5.2)
        v_in = v + noise_v * torch.randn_like(v) if (training and noise_v > 0) else v
        q_in = q + noise_q * torch.randn_like(q) if (training and noise_q > 0) else q
        # ss_prob may be a python float (eager) or a 0-dim tensor (compiled:
        # keeps the value out of the traced graph so it doesn't recompile as
        # the schedule ramps). use_ss must then be passed explicitly (static).
        if use_ss is None:
            use_ss = training and float(ss_prob) > 0.0
        else:
            use_ss = bool(use_ss) and training

        pc_delta = None
        start = 0
        if self.pc_k > 0 and burn_in > 0:
            with torch.enable_grad():
                pc_delta, (h_v, h_low, h_up, pb_prev, pb_curr,
                           v_pred_prev, q_hat_prev, win_start) = \
                    self.pc_infer(v_in, q_in, q, v, ctx, burn_in, a_in=a)
            start = burn_in                        # window consumed the burn-in
        else:
            h_v, h_low, h_up = self.initial_states(B, v, v_in[:, 0], q_in[:, 0], ctx)
            v_pred_prev = None                     # last vision prediction
            q_hat_prev = None                      # last proprio prediction
            pb_prev = v.new_zeros(B, self.pb_dim)
            pb_curr = v.new_zeros(B, self.pb_dim)
            win_start = 0

        loss_v = v.new_zeros(())
        loss_q = v.new_zeros(())
        loss_a = v.new_zeros(())
        # 有効フレーム数(マスクの和)。従来の「損失を足した step 数」の一般化で、
        # mask が全 1 なら nv = step 数 * B となり平均値は従来と一致する。
        nv = v.new_zeros(())
        nq = v.new_zeros(())
        na = v.new_zeros(())
        # aux accumulators stay on-device; converting to float here would force
        # a GPU sync every step (launch-bound model -> huge slowdown)
        fe_sum = v.new_zeros(())
        fe_n = v.new_zeros(())
        pb_norm_sum = v.new_zeros(())
        a_absmean_sum = v.new_zeros(())

        lang_v = self.vision.lang_term(ctx) if self.fast_loop else None
        lang_a = self.action.lang_terms(ctx) if self.fast_loop else None
        # Wp_up(pb_prev), Wp_up(pb_curr) の事前射影。
        # pc_k>0 のときループは s=burn_in から始まるので、burn_in が vision_stride の
        # 倍数でないと初回が視覚 tick にならず、None のまま _pb_mix に渡って落ちていた
        # (2026-07-30 に burn_in=10 で発覚)。窓推論が返した pb_prev/pb_curr から
        # 射影を作っておけば burn_in の値に依らず正しく初期化される（数値も整合する）。
        if self.fast_loop:
            pj_prev = self.action.Wp_up(pb_prev)
            pj_curr = self.action.Wp_up(pb_curr)
        else:
            pj_prev = pj_curr = None
        hl_seq, tgt_seq, w_seq = [], [], []   # 行動損失を一括計算するための保存
        for s in range(start, L):
            if s == burn_in and self.pc_k == 0 and not self.bptt_burnin:
                # truncate graph: burn-in only initializes state (non-PC mode;
                # with pc inference the gradient must flow through the window)
                h_v = h_v.detach()
                h_low = h_low.detach()
                h_up = h_up.detach()
                pb_prev = pb_prev.detach()
                pb_curr = pb_curr.detach()
                if v_pred_prev is not None:
                    v_pred_prev = v_pred_prev.detach()
                if q_hat_prev is not None:
                    q_hat_prev = q_hat_prev.detach()
            in_loss = s >= burn_in
            ss_here = use_ss and in_loss

            # --- vision update every `stride` steps ---
            if s % self.stride == 0:
                if v_pred_prev is None:
                    eps_v = torch.zeros_like(v[:, s])
                else:
                    eps_v = v[:, s] - v_pred_prev          # clean obs - prediction (target stays clean)
                    if in_loss:
                        e = eps_v.pow(2).sum(-1) / self.v_dim
                        loss_v = loss_v + (e * msk[:, s]).sum()
                        nv = nv + msk[:, s].sum()
                        fe_sum = fe_sum + (e * msk[:, s]).sum().detach()
                        fe_n = fe_n + msk[:, s].sum().detach()
                # scheduled sampling: feed own prediction as input with prob ss_prob
                v_input = v_in[:, s]
                if ss_here and v_pred_prev is not None:
                    m = (torch.rand(B, 1, device=v.device) < ss_prob).float()
                    v_input = m * v_pred_prev.detach() + (1.0 - m) * v_input
                _ac, _qc = self._vis_motor(a, q_in, s)
                h_v = self.vision.step(v_input, ctx, h_v, lang=lang_v,
                                       a_ctx=_ac, q_ctx=_qc)
                v_pred_prev = self.vision.predict(h_v)
                # PB anchor (detached vision state -> no grad to vision, D4-2)
                pb_prev = pb_curr
                pb_curr = self._pb(h_v, eps_v)
                if self.fast_loop:
                    pj_prev = pj_curr if pj_curr is not None else self.action.Wp_up(pb_prev)
                    pj_curr = self.action.Wp_up(pb_curr)
                win_start = s

            # --- PB interpolation within the window (causal ramp prev->curr) ---
            i = s - win_start
            frac = (i + 1) / self.stride
            pb_t = self._pb_mix(pb_prev, pb_curr, frac)
            if pc_delta is not None:
                pb_t = pb_t + pc_delta
            pb_norm_sum = pb_norm_sum + pb_t.norm(dim=-1).mean().detach()

            # --- action step (scheduled sampling on proprio input) ---
            q_input = q_in[:, s]
            if ss_here and q_hat_prev is not None:
                m = (torch.rand(B, 1, device=v.device) < ss_prob).float()
                q_input = m * q_hat_prev.detach() + (1.0 - m) * q_input
            if self.fast_loop:
                pj = self._pb_mix(pj_prev, pj_curr, frac)
                if pc_delta is not None:
                    pj = pj + self.action.Wp_up(pc_delta)
                h_low, h_up = self.action.step(q_input, ctx, pb_t, h_low, h_up,
                                               lang=lang_a, pb_proj=pj)
                # proprio_from_action=True では行動の代表点が毎 step 必要になるので
                # 行動ヘッドの一括化(fast_loop の 3 要素のうち 1 つ)は効かなくなる。
                # 言語項と PB 射影の hoist は引き続き効く。
                q_hat, a_hat = self.q_pred(h_low, q_input)
                if a_hat is not None:
                    a_absmean_sum = a_absmean_sum + a_hat.abs().mean().detach()
            else:
                h_low, h_up = self.action.step(q_input, ctx, pb_t, h_low, h_up)
                a_hat, q_head = self.action.heads(h_low)
                q_hat = self.q_pred(h_low, q_input, a_hat)[0] \
                    if self.proprio_from_action else q_head
                a_absmean_sum = a_absmean_sum + a_hat.abs().mean().detach()
            q_hat_prev = q_hat                     # model's estimate of q_{s+1}

            if in_loss and s + self.chunk <= L:
                # chunk_len=1 -> target is a[:, s]; >1 -> flattened future chunk
                tgt = a[:, s:s + self.chunk].reshape(B, -1)
                # チャンク目標は s..s+chunk-1 の全フレームが有効なときだけ損失に入れる
                w = msk[:, s] if self.chunk == 1 else msk[:, s:s + self.chunk].prod(dim=1)
                if self.fast_loop:
                    hl_seq.append(h_low); tgt_seq.append(tgt); w_seq.append(w)
                else:
                    if self.gmm:
                        nll = self.action.action_nll(h_low, tgt, reduce="none")
                    else:
                        nll = F.huber_loss(a_hat, tgt, reduction="none").mean(dim=-1)
                    loss_a = loss_a + (nll * w).sum()
                    na = na + w.sum()
            if in_loss:
                if s + 1 < L:
                    eps_q = q[:, s + 1] - q_hat            # predict next proprio
                    e = eps_q.pow(2).sum(-1) / self.q_dim
                    w = msk[:, s + 1]                        # 目標は q_{s+1} なので s+1 の有効性
                    loss_q = loss_q + (e * w).sum()
                    nq = nq + w.sum()
                    fe_sum = fe_sum + (e * w).sum().detach()
                    fe_n = fe_n + w.sum().detach()

        if self.fast_loop and hl_seq:
            # 行動ヘッドを 1 回の大きな行列積で回す(step ごとの小さな呼び出しを廃止)
            H = torch.cat(hl_seq, dim=0)
            T = torch.cat(tgt_seq, dim=0)
            Wt = torch.cat(w_seq, dim=0)
            if self.gmm:
                nll = self.action.action_nll(H, T, reduce="none")
            else:
                a_all, _ = self.action.heads(H)
                nll = F.huber_loss(a_all, T, reduction="none").mean(dim=-1)
            loss_a = (nll * Wt).sum()
            na = Wt.sum()
            if not self.proprio_from_action:
                # 一括化のため毎 step の a_hat が無いので、目標側の振幅を参考値として使う
                a_absmean_sum = T.abs().mean().detach() * L
        # 有効フレーム数で正規化する。mask を含めた全体で割ると、有効数の少ない
        # バッチだけ勾配が小さくなり実効学習率がゆらぐので必ず nv/nq/na で割る。
        loss_v = loss_v / nv.clamp(min=1.0)
        loss_q = loss_q / nq.clamp(min=1.0)
        loss_a = loss_a / na.clamp(min=1.0)
        total = lambda_v * loss_v + lambda_q * loss_q + lambda_a * loss_a

        # aux values are returned as detached tensors; the caller converts to
        # float only when logging (one sync per log interval, not per step)
        return {
            "total": total,
            "loss_v": loss_v,
            "loss_q": loss_q,
            "loss_a": loss_a,
            "fe": fe_sum / fe_n.clamp(min=1.0),
            "pb_norm": pb_norm_sum / L,
            "a_absmean": a_absmean_sum / L,
            "a_target_absmean": (a[:, burn_in:].abs().mean(-1) * msk[:, burn_in:]).sum()
            / msk[:, burn_in:].sum().clamp(min=1.0),
        }
