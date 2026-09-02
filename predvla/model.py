"""PV-RNN に忠実な PC-RNN。観測は順方向に入らず、予測誤差としてのみ効く。

構造（2026-07-31、ユーザー指示で並列 + ブリッジに変更）

        T（共有の最上位、d_t、τ_t）  ← 言語
              ├────────────────┐
              ↓                ↓
        V（d, τ_v）         A_up（d, τ_up）
          出力 v̂(192)             ↓
              ↑                A_low（d, τ_low）
              │                  出力 q̂(9), â(7, GMM)
              │                    │
              └── ブリッジ ────────┘
                  V → A : PB = W_pb·d^V_t      （同 step。V を先に進める）
                  A → V : W_ba·â_{t−1} + W_bq·q̂_{t−1}
                          ★ 観測ではなくモデルの予測を渡す。
                            観測を渡すと順方向に観測が入り生成モデルでなくなる。
                            pcv6aq で「視覚 RNN は自分の動きを知らないと予測できない」
                            ことを実測（学習時 v 0.2595 → 0.2453）したので、その経路を
                            予測値で再現する。efference copy に相当。

T を低次元（d_t=64）にすることで、T 自体が視覚と行動の結合点（旧 PB の役割）になる。

各層の更新（LibPvrnn の rnn_layer.cpp: _computeRecurrence と同形）
  ĉ^l_t = W^l_pri · (親の d) + b            事前値（トップダウン生成。根は自分の d_{t−1}）
  h^l_t = (1−1/τ)·h^l_{t−1}
        + (1/τ)·tanh( W_dh·d^l_{t−1} + W_td·(親の d) + ブリッジ + B + U^l·c^l_t )
  d^l_t = tanh(h^l_t)

自由エネルギー
  E = Σ_t[ m^v·‖v−v̂‖²/192 + ‖q−q̂‖²/9 + λ_a·NLL(a) ] + Σ_l w_l·Σ_t ½‖c^l−ĉ^l‖²
  c が PV-RNN の A に対応する自由変数（分散なしの決定論版、低ランク U で h を直接ずらす）
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class Layer(nn.Module):
    """1 層。自由変数 c（低ランク U で h に加算）と、その事前値を作る prior を持つ。

    in_dim=0 なら根（親なし）。prior は根では自分の d_{t−1} から作る。
    bridge_dims は横方向の入力（生成モデルの階層外の結合）。prior には入れない。
    """

    def __init__(self, d: int, tau: float, r: int, in_dim: int,
                 lang_dim: int = 0, bridge_dims=(), use_c: bool = True,
                 cell: str = "rnn"):
        """use_c=False（2026-08-27、はしごの L3 以降）で自由変数 c の経路を持たない層になる。

        ★なぜ消してよいか: c を推論しない条件では c = ĉ = W_pri(parent_d) なので、
          h に足される項は U(W_pri(parent_d)) という **低ランクな線形写像**になる。
          同じ parent_d からの全ランク写像 W_td が既に足されているので、両者の和は
          W_td 単独で厳密に表現できる。つまり U と W_pri は**冗長**で、
          残すとパラメータを無駄に食うだけ。消したぶんは層幅 d に回す。
        """
        super().__init__()
        assert tau >= 1.0
        assert cell in ("rnn", "lstm")
        # ★cell="lstm"（2026-08-27、はしごの L5）。
        #   既存の線形写像の出力を d -> 4d にして、その pre-activation を
        #   i/f/g/o の 4 ゲートに割る。つまり「入力の作り方」は leaky RNN と完全に同じで、
        #   **非線形の中身だけが LSTM に変わる**。はしごの 1 段が「セルの違い」だけになる。
        #   状態は [h; c] を連結した 2d のテンソル 1 本で持ち回す（既存の配管を変えない）。
        #   ★τ は使わない（LSTM は自前のゲートで時間積分するため）。L5 は tau=1 前提。
        self.cell = cell
        self.is_lstm = (cell == "lstm")
        g = 4 if self.is_lstm else 1
        self.gates = g
        self.state_dim = 2 * d if self.is_lstm else d
        self.d, self.tau, self.r, self.in_dim = d, float(tau), r, in_dim
        self.W_dh = nn.Linear(d, g * d, bias=True)
        if self.is_lstm:
            for k in range(g):
                nn.init.orthogonal_(self.W_dh.weight[k * d:(k + 1) * d], gain=0.9)
            nn.init.zeros_(self.W_dh.bias)
            # 忘却ゲートのバイアスを 1 に（LSTM の慣行。初期に記憶を保つ）
            with torch.no_grad():
                self.W_dh.bias[d:2 * d].fill_(1.0)
        else:
            nn.init.orthogonal_(self.W_dh.weight, gain=0.9)
            nn.init.zeros_(self.W_dh.bias)
        self.W_td = nn.Linear(in_dim, g * d, bias=False) if in_dim > 0 else None
        self.W_l = nn.Linear(lang_dim, g * d, bias=False) if lang_dim > 0 else None
        self.bridges = nn.ModuleList([nn.Linear(bd, g * d, bias=False)
                                      for bd in bridge_dims])
        self.use_c = bool(use_c)
        if self.use_c:
            self.U = nn.Linear(r, g * d, bias=False)
            nn.init.normal_(self.U.weight, std=1.0 / math.sqrt(r))
            self.W_pri = nn.Linear(in_dim if in_dim > 0 else d, r, bias=True)
            nn.init.zeros_(self.W_pri.weight)  # 初期は ĉ=0（ダイナミクスを信じる）
            nn.init.zeros_(self.W_pri.bias)
        else:
            self.U = None
            self.W_pri = None

    def prior(self, parent_d, d_prev):
        if self.W_pri is None:
            return None
        return self.W_pri(d_prev if self.W_td is None else parent_d)

    def step(self, h_prev, d_prev, c_t, parent_d=None, lang_term=None,
             bridge_in=()):
        pre = self.W_dh(d_prev)
        if self.U is not None and c_t is not None:
            pre = pre + self.U(c_t)
        if self.W_td is not None:
            pre = pre + self.W_td(parent_d)
        if lang_term is not None:
            pre = pre + lang_term
        for lin, x in zip(self.bridges, bridge_in):
            pre = pre + lin(x)
        if self.is_lstm:
            # h_prev は [h; c] を連結した 2d。pre は 4d（i, f, g, o の順）。
            d = self.d
            h_p, c_p = h_prev[..., :d], h_prev[..., d:]
            i, f, gg, o = pre.split(d, dim=-1)
            c_t = torch.sigmoid(f) * c_p + torch.sigmoid(i) * torch.tanh(gg)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
            return torch.cat([h_t, c_t], dim=-1), h_t
        leak = 1.0 / self.tau
        h = (1.0 - leak) * h_prev + leak * torch.tanh(pre)
        return h, torch.tanh(h)


class PredVLA(nn.Module):
    """T（共有上位）+ V（視覚枝）+ A_up/A_low（行動枝）、V↔A のブリッジ付き。"""

    LAYERS = ("t", "v", "up", "low")

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.v_dim, self.q_dim, self.a_dim, self.l_dim = \
            m["v_dim"], m["q_dim"], m["a_dim"], m["l_dim"]
        self.r = m["r"]
        # ★層別 r（2026-08-18 追加、ユーザー指示。既定は全層 m["r"] = 従来と完全に同一）。
        #   動機: d:c（層幅 : 自由変数の次元）が PV-RNN の慣行 10:1 から外れている。
        #     現行  T 64:64 = 1:1 / V・A_up・A_low 256:64 = 4:1
        #   c が広いと posterior が c で何でも説明でき、prior（ダイナミクス）を鍛える
        #   圧力が減る。実測でも posterior は単調改善する一方 prior は 2k step で停滞し
        #   prior-q は悪化する。complexity の重み w の掃引では効果が出なかったので、
        #   効くつまみは w ではなく r の側かもしれない、という仮説の検証用。
        #   r_top / r_v / r_up / r_low で層ごとに指定する。
        self.r_of = {"t":   int(m.get("r_top", self.r)),
                     "v":   int(m.get("r_v",   self.r)),
                     "up":  int(m.get("r_up",  self.r)),
                     "low": int(m.get("r_low", self.r))}
        d, dt = m["d"], m["d_top"]
        self.d, self.d_top = d, dt
        # ★層別 d（2026-08-20 追加、ユーザー指示）。
        #   既定は d_v = d_up = d_low = m["d"] なので**従来と完全に同一**（h0 の
        #   パラメータ名も形も変わらないので既存 ckpt がそのまま読める）。
        #   動機: 行動枝の上層は PB(32) + T(64) を受けるだけなので 256 は過剰かも
        #   しれない。T（言語）は元から d_top=64。指定するのは V / A_up / A_low。
        d_v = int(m.get("d_v", d))
        d_up = int(m.get("d_up", d))
        d_low = int(m.get("d_low", d))
        self.d_v, self.d_up, self.d_low = d_v, d_up, d_low
        self._d_uniform = (d_v == d_up == d_low == d)
        self.pb_dim = m["pb_dim"]
        self.gmm_k, self.gmm_min_std = m["gmm_k"], m["gmm_min_std"]
        self.w = {"t": m["w_top"], "v": m["w_v"], "up": m["w_up"], "low": m["w_low"]}

        # --- ablation フラグ（2026-08-01 追加）。既定値は現行と完全に同一。 ---
        # 目的: ①階層と時定数 ②視覚予測 ③PB ④A→V の efference ⑤自由変数 c の
        # それぞれの寄与を分離する。⑤は n_itr=0 で既に測れている。
        #   ab_no_av_bridge : A→V（前 step の â, q̂ を V に渡す経路）を切る
        #   ab_no_pb        : V→A（PB = W_pb·d^V）を切る
        #   ab_tau_flat     : 全層の τ を 1 にする（多時定数階層を殺す。層数は保つ）
        # λ_v は train 側（free_energy の lambda_v）で扱う。
        # ★feedforward（2026-08-27、はしごの L3）。
        #   「予測していたものを、入力として受け取る」に一括で切り替える:
        #     head_v（視覚の予測）を削除し、v_t を V に前向きに入れる
        #     q_t を V と A_low の両方に前向きに入れる（head_q は次固有感覚の補助損失として残す。
        #       既存 BC-LSTM が同じ補助損失を持つので、はしごの最終段と 1 点差にするため）
        #     自由変数 c の経路（U, W_pri）を全層で削除（use_c=False。上の Layer の説明参照）
        #   ★これで「観測は誤差経由でしか入らない」という本モデルの定義が外れ、
        #     普通の多時定数 RNN（MT-RNN）になる。評価は n_itr=0 必須。
        self.feedforward = bool(m.get("feedforward", False))
        # ★ff_keep_pred（2026-08-30、ユーザー指摘。はしごの L3′）。
        #   feedforward は本来 2 つを同時に変えている:
        #     (a) 生の観測を前向きに入れる   (b) 視覚の予測 head と予測損失を消す
        #   これを立てると (a) だけを残し、**視覚の予測 head と loss_v を戻す**。
        #   → L2(予測 ON/直入力 OFF) L3′(ON/ON) L3(OFF/ON) の 2x2 が閉じる。
        #   下流は self.head_v is None で分岐しているので、ここだけで足りる。
        self.ff_keep_pred = bool(m.get("ff_keep_pred", False))
        # ★ff_delay（2026-08-30、ユーザー指摘で追加。L3′ を正しく作るのに要る）。
        #   前向きに入れる観測を **1 step 遅らせる**（step t には v_{t-1}, q_{t-1} を入れる）。
        #   これが無いと、step t で v_t を V に入れてから head_v に v_t を出させることになり、
        #   「予測」ではなく「いま見たものの復元」を解く別課題になる。
        #   実測（同 step 版、spatial s1 @30k）: loss_v 0.1641 / loss_q 0.0025 に対し、
        #   L2（本物の予測）は 0.5127 / 0.1739。課題がすり替わっていた。
        #   遅らせると head は「見る前に当てる」ままなので L2 と損失の定義が揃う。
        self.ff_delay = bool(m.get("ff_delay", False))
        # ★ff_gate_v（2026-08-31、ユーザー指摘）。順方向に入れる**視覚だけ** mask_v で間引く。
        #   これが無いと feedforward 経路は mv_t を見ないので、vision_stride を 4 にしても
        #   生の視覚は毎 step V に入り、変わるのは loss_v の密度だけになる
        #   （obs_to_low は e = e * mv_t で間引かれているので、両者で意味が違っていた）。
        #   固有感覚は 20Hz で常に取れるので間引かない。
        self.ff_gate_v = bool(m.get("ff_gate_v", False))
        # ★cell（2026-08-27、はしごの L5）。"rnn"（既定・従来と完全に同一）か "lstm"。
        #   lstm のとき状態は [h; c] の 2d になるので、h0 は層別に 2d で持つ。
        self.cell = str(m.get("cell", "rnn"))
        assert self.cell in ("rnn", "lstm")
        self.ab_no_av_bridge = bool(m.get("ab_no_av_bridge", False))
        self.ab_no_pb = bool(m.get("ab_no_pb", False))
        # PB を A_low にも渡す(2026-08-08 追加。既定 False = 従来と完全に同一)。
        #   現行は V→A の PB が A_up にしか入らず、視覚の情報が行動出力層(A_low)まで
        #   2 段の伝播を要する。しかも W_pb で 256→32 に絞られている。
        #   「視覚の誤差が行動に届かない」ことの候補原因なので、直結を試せるようにする。
        self.pb_to_low = bool(m.get("pb_to_low", False)) and not self.ab_no_pb
        tau_flat = bool(m.get("ab_tau_flat", False))
        # ★ab_tau_uniform（2026-08-18 追加。表③ A1 用）。
        #   全層の τ を指定値に統一する。既定 None なら従来と完全に同一。
        #   ab_tau_flat（τ=1 固定）との違い: τ=1 は「階層はあるが全層が最速」なので、
        #   多時定数の必要性と「遅い層があること」を分離できない。平均値（16/8/5/2 の
        #   平均 7.75 → 8）に統一すれば「遅いが均一」との対照になり、多時定数そのものを問える。
        tau_uni = m.get("ab_tau_uniform", None)
        assert not (tau_flat and tau_uni is not None), \
            "ab_tau_flat と ab_tau_uniform は同時に立てられない"
        tau_t, tau_v, tau_u, tau_l = (m["tau_top"], m["tau_v"],
                                      m["tau_up"], m["tau_low"])
        if tau_flat:
            tau_t = tau_v = tau_u = tau_l = 1.0
        elif tau_uni is not None:
            tau_t = tau_v = tau_u = tau_l = float(tau_uni)

        # ★ff_keep_c（2026-08-30）。feedforward は c（PV-RNN の A）の経路も
        #   同時に消していた（U と W_pri で 82,176 params = モデルの 12%）。
        #   これを立てると **観測の直入力はそのままに、c の経路だけ残す**。
        #   L2 は train.no_c でも経路は残っており毎 step 事前値 ĉ を使うので、
        #   L2 と L3′ の差から「c の経路の有無」を外すのに要る。
        self.ff_keep_c = bool(m.get("ff_keep_c", False))
        uc = (not self.feedforward) or self.ff_keep_c
        self.T = Layer(dt, tau_t, self.r_of["t"], in_dim=0, lang_dim=self.l_dim,
                       use_c=uc, cell=self.cell)
        # V は T の子。ブリッジで行動枝の予測 â, q̂ を受ける
        v_bridges = () if self.ab_no_av_bridge else (self.a_dim, self.q_dim)
        if self.feedforward:
            # ★観測を前向きに: 生の視覚 v_t と生の固有感覚 q_t を V にも入れる
            v_bridges = v_bridges + (self.v_dim, self.q_dim)
        self.V = Layer(d_v, tau_v, self.r_of["v"], in_dim=dt,
                       bridge_dims=v_bridges, use_c=uc, cell=self.cell)
        # V → A のブリッジ。切るときは重みを作らない（params が減るので必ず併記する）
        self.W_pb = None if self.ab_no_pb else nn.Linear(d_v, self.pb_dim, bias=False)
        up_bridges = () if self.ab_no_pb else (self.pb_dim,)
        self.A_up = Layer(d_up, tau_u, self.r_of["up"], in_dim=dt,
                          bridge_dims=up_bridges, use_c=uc, cell=self.cell)
        low_bridges = (self.pb_dim,) if self.pb_to_low else ()
        if bool(m.get("err_to_low", False)) or bool(m.get("obs_to_low", False)):
            low_bridges = low_bridges + (
                (np.load(m["v_proj"] if os.path.isabs(m["v_proj"])
                         else os.path.join(_HERE, "..", m["v_proj"]))["P"].shape[1])
                if m.get("v_proj") else self.v_dim,)
        if self.feedforward:
            low_bridges = low_bridges + (self.q_dim,)   # ★生の q_t を A_low へ
        self.A_low = Layer(d_low, tau_l, self.r_of["low"], in_dim=d_up,
                           bridge_dims=low_bridges, use_c=uc, cell=self.cell)

        # ★feedforward では視覚を予測しない（head_v ごと削除して params を層幅に回す）
        self.head_v = (None if (self.feedforward and not self.ff_keep_pred)
                       else mlp(d_v, m["head_hidden"], self.v_dim))
        self.head_q = mlp(d_low, m["head_hidden"], self.q_dim)
        self.head_a = mlp(d_low, m["head_hidden"],
                          self.gmm_k * (1 + 2 * self.a_dim))
        # 視覚誤差を測る部分空間(2026-08-08 追加。既定 None = 従来と完全に同一)。
        #   192 次元の視覚特徴のうち、行動の残差を説明するのは実質 7 次元だけと分かった
        #   (交差検証 R² が 192 次元と 7 次元で同じ 0.215。PCA 32 次元では 0.183)。
        #   現行の loss_v は残り 185 次元、つまり容量の 96% を行動と無関係な予測に
        #   使っている。P で射影してから誤差を測れば、その無駄を外せる。
        #   **学習パラメータではない**(固定の定数行列)ので params は増えない。
        # ★C: 視覚の予測誤差を A_low に前向き投射する(2026-08-08 追加。既定 False)。
        #   predvla は観測を順方向に通さないので、誤差は c の勾配経由でしか行動に触れない。
        #   Jacobian 解析ではその経路が行動に対してほぼ効かない(行動に効く方向の
        #   83〜93% が観測不変)。Rao&Ballard 型の予測符号化では誤差ユニットが
        #   前向きに投射するのが本体なので、その経路を足せるようにする。
        #   射影 P を通してから入れるので +1,792 params(+0.27%)で済む。
        self.err_to_low = bool(m.get("err_to_low", False))
        # ★obs_to_low（2026-08-17 追加。論文のアブレーション「真の feedforward VLA」）。
        #   err_to_low と同じ bridge を使うが、流すものが違う:
        #     err_to_low : e = v_obs − v̂        （予測誤差を前向きに投射。Rao&Ballard 型）
        #     obs_to_low : e = v_obs            （★観測そのものを前向きに入れる）
        #   後者にすると「観測は順方向に入らない」という本モデルの定義そのものが崩れ、
        #   通常の feedforward な視覚条件付け方策に近づく。これが対照の狙い。
        #   ★n_itr=0 が厳密な開ループにならなくなる点に注意（観測が前向きに入るので）。
        #     この変種では ER の有無に関わらず観測が行動に効く。
        self.obs_to_low = bool(m.get("obs_to_low", False))
        assert not (self.err_to_low and self.obs_to_low), \
            "err_to_low と obs_to_low は同時に立てられない（同じ bridge を使う）"
        # ★視覚サーボ(2026-08-09 追加。既定 None = 従来と完全に同一)。
        #   Δ = D·v_obs − D·v̂ (観測から復号した相対位置 − 予測から復号した相対位置)
        #   a = a_gen + σ·tanh(K·Δ/scale)
        #   C(誤差の前向き投射)が壊れた 3 つの理由をすべて避ける設計:
        #     ・**出力への残差**なので内部の力学(再帰状態)には触れない
        #     ・信号は **3 次元の物理量**(m 単位)で、抽象的な特徴ではない
        #     ・tanh と σ で **有界**(ER の er_w に相当する trust region)
        #   実測: object の失敗は d_min 中央値 28.6mm(四分位 17.5〜49.0)、
        #         成功時 22.5mm。飽和点 scale はこの範囲に合わせる。
        self.servo_scale = float(m.get("servo_scale", 0.04))   # 40mm で飽和
        vd = m.get("v_decoder", None)
        vp = m.get("v_proj", None)
        if vp:
            P = np.load(vp if os.path.isabs(vp)
                        else os.path.join(_HERE, "..", vp))["P"]
            assert P.shape[0] == self.v_dim, f"射影の形が合わない {P.shape}"
            self.register_buffer("v_proj", torch.from_numpy(P.astype("float32")))
        else:
            self.v_proj = None
        if vd:
            z = np.load(vd if os.path.isabs(vd) else os.path.join(_HERE, "..", vd))
            self.register_buffer("dec_W", torch.from_numpy(z["W"].astype("float32")))
            self.register_buffer("dec_mu", torch.from_numpy(z["mu"].astype("float32")))
            self.servo_K = nn.Linear(z["W"].shape[1], self.a_dim, bias=False)
            # ★σ と K を両方ゼロにすると勾配が恒等的にゼロになり、鞍点から出られない
            #   (2026-08-10 に実測。∂a/∂σ = tanh(K·Δ/s) = 0、∂a/∂K = σ·sech²·(…) = 0)。
            #   K だけ小さな乱数にする。σ = 0 なので初期の出力は素のモデルと完全に同一。
            nn.init.normal_(self.servo_K.weight, std=0.1)
            self.servo_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.dec_W = None
            self.servo_K = None
            self.servo_sigma = None
        # 誤差の投射は A_low の bridge(最後の 1 本)が担う。重複した重みは作らない。
        self.err_dim = ((self.v_proj.shape[1] if self.v_proj is not None
                         else self.v_dim)
                        if (self.err_to_low or self.obs_to_low) else 0)
        if self.err_to_low or self.obs_to_low:
            # 初期は前向き経路を使わない（素のモデルと出力が完全に一致する）
            nn.init.zeros_(self.A_low.bridges[-1].weight)
        sd = 2 if self.cell == "lstm" else 1     # 状態の本数（lstm は [h; c]）
        self.h0_t = nn.Parameter(torch.zeros(sd * dt))
        if self.cell == "lstm":
            self._d_uniform = False              # 形が変わるので層別に持つ
        # 層別 d のときだけ h0 を 3 本に分ける（一様なら従来の (3,d) を維持して
        # 既存 ckpt との互換を保つ）。
        if self._d_uniform:
            self.h0 = nn.Parameter(torch.zeros(3, d))         # V, A_up, A_low
        else:
            self.h0_v = nn.Parameter(torch.zeros(sd * d_v))
            self.h0_up = nn.Parameter(torch.zeros(sd * d_up))
            self.h0_low = nn.Parameter(torch.zeros(sd * d_low))

    # ---- GMM ----
    def gmm_params(self, d_low):
        B = d_low.shape[0]
        out = self.head_a(d_low).view(B, self.gmm_k, 1 + 2 * self.a_dim)
        mu = torch.tanh(out[..., 1:1 + self.a_dim])
        std = nn.functional.softplus(out[..., 1 + self.a_dim:]) + self.gmm_min_std
        return out[..., 0], mu, std

    def action_nll(self, d_low, target):
        logits, mu, std = self.gmm_params(d_low)
        log_w = torch.log_softmax(logits, dim=-1)
        z = (target.unsqueeze(1) - mu) / std
        log_comp = (-0.5 * z.pow(2) - std.log()
                    - 0.5 * math.log(2.0 * math.pi)).sum(-1)
        return -(torch.logsumexp(log_w + log_comp, dim=-1)) / self.a_dim

    def action_point(self, d_low):
        logits, mu, _ = self.gmm_params(d_low)
        k = logits.argmax(dim=-1)
        return mu[torch.arange(mu.shape[0], device=mu.device), k]

    def init_state(self, B):
        sd = 2 if self.cell == "lstm" else 1
        ht = self.h0_t.unsqueeze(0).expand(B, sd * self.d_top)
        if self._d_uniform:
            h = self.h0.unsqueeze(1).expand(3, B, self.d)
            hv, hu, hl = h[0], h[1], h[2]
        else:
            hv = self.h0_v.unsqueeze(0).expand(B, sd * self.d_v)
            hu = self.h0_up.unsqueeze(0).expand(B, sd * self.d_up)
            hl = self.h0_low.unsqueeze(0).expand(B, sd * self.d_low)
        if self.cell == "lstm":
            # d は h 部分（LSTM の出力はすでに o*tanh(c) で有界）
            return [ht, hv, hu, hl], [ht[:, :self.d_top], hv[:, :self.d_v],
                                      hu[:, :self.d_up], hl[:, :self.d_low]]
        return [ht, hv, hu, hl], [torch.tanh(ht), torch.tanh(hv),
                                  torch.tanh(hu), torch.tanh(hl)]

    def one_step(self, hs, ds, cs, lang_t, prev_a, prev_q, comp_reduce="mean",
                 v_obs=None, mv_t=None, cs_is_delta=False, q_obs=None):
        """1 step 進める。cs は dict（"t","v","up","low" -> (B,r)）。
        cs の値が None の層は事前値 ĉ をそのまま使う（prior 生成）。
        戻り値: hs, ds, v_hat, q_hat, a_hat(代表点), comp（層ごとの ½‖c−ĉ‖² の重み付き和）

        cs_is_delta: True なら cs の値を「事前値からのずれ δ」として扱う（c = ĉ + δ、
          comp = w·½‖δ‖²）。★ER ループを通した学習（2026-08-12）で使う。δ=0 が
          そのまま prior 生成になるので、初期化が prior と一致して扱いやすい。
          既定 False で従来と完全に同一。

        comp_reduce: "mean" は従来どおりバッチ平均のスカラ。"none" は (B,) を返す。
          ★"none" が要る理由（2026-08-03、バッチ ER のため）
            バッチ ER では 1 プロセスで B 本のロールアウトを同時に回す。目的関数を
            バッチ平均にすると 1 サンプルあたりの勾配が 1/B になり、**Adam の
            eps=1e-4（既定の 1 万倍）が相対的に B 倍効いて更新量が変わる**。
            サンプルごとに出して和を取れば、単体で走らせたときと勾配が厳密に一致する。
            既定は "mean" なので学習側の挙動は一切変わらない。
        """
        ht, hv, hu, hl = hs
        dt, dv, du, dl = ds
        comp = None
        parts = {}          # 層別の Complexity（どの層の c が働いているかを見る）

        def pick(name, chat):
            nonlocal comp
            c = cs.get(name)
            if chat is None:
                # ★use_c=False の層（feedforward）。c の経路そのものが無いので
                #   与えられた c は無視する。Complexity も足さない。
                return None
            if c is None:
                return chat
            if cs_is_delta:
                sq = c.pow(2).sum(-1)               # (B,)  δ そのものが「ずれ」
                term = self.w[name] * 0.5 * (sq if comp_reduce == "none" else sq.mean())
                parts[name] = term
                comp = term if comp is None else comp + term
                return chat + c
            sq = (c - chat).pow(2).sum(-1)          # (B,)
            term = self.w[name] * 0.5 * (sq if comp_reduce == "none" else sq.mean())
            parts[name] = term
            comp = term if comp is None else comp + term
            return c

        # T（根）
        c = pick("t", self.T.prior(None, dt))
        ht, dt = self.T.step(ht, dt, c, None, lang_t)
        # V（T の子。ブリッジで前 step の â, q̂ を受ける）
        c = pick("v", self.V.prior(dt, dv))
        v_bridge_in = () if self.ab_no_av_bridge else (prev_a, prev_q)
        if self.feedforward:
            # ★生の観測を前向きに。v_obs / q_obs が無い step は 0 で埋める
            B_ = dt.shape[0]
            vo = (v_obs if v_obs is not None
                  else dt.new_zeros(B_, self.v_dim))
            if self.ff_gate_v and mv_t is not None:
                vo = vo * mv_t.unsqueeze(-1)   # ★視覚が更新された step だけ通す
            qo = (q_obs if q_obs is not None
                  else dt.new_zeros(B_, self.q_dim))
            v_bridge_in = v_bridge_in + (vo, qo)
        hv, dv = self.V.step(hv, dv, c, dt, None, v_bridge_in)
        # A_up（T の子。ブリッジで PB を受ける）
        up_bridge_in = () if self.ab_no_pb else (self.W_pb(dv),)
        c = pick("up", self.A_up.prior(dt, du))
        hu, du = self.A_up.step(hu, du, c, dt, None, up_bridge_in)
        # A_low(pb_to_low なら PB を直結、err_to_low なら視覚の予測誤差も受ける)
        low_in = list(up_bridge_in) if self.pb_to_low else []
        if self.err_to_low or self.obs_to_low:
            if v_obs is None:
                e = torch.zeros(dv.shape[0], self.err_dim,
                                device=dv.device, dtype=dv.dtype)
            else:
                if self.obs_to_low:
                    e = v_obs                      # ★観測そのものを前向きに
                else:
                    e = v_obs - self.head_v(dv)    # 予測誤差を前向きに
                if self.v_proj is not None:
                    e = e @ self.v_proj
                if mv_t is not None:
                    e = e * mv_t.unsqueeze(-1)     # 視覚が更新された step だけ使う
            low_in.append(e)
        if self.feedforward:
            low_in.append(q_obs if q_obs is not None
                          else dl.new_zeros(dl.shape[0], self.q_dim))
        c = pick("low", self.A_low.prior(du, dl))
        hl, dl = self.A_low.step(hl, dl, c, du, bridge_in=tuple(low_in))

        # ★feedforward では視覚を予測しない。下流（ER の診断など）が形を期待するので
        #   ゼロを返す。loss_v は free_energy 側で 0 に落とす。
        v_hat = (self.head_v(dv) if self.head_v is not None
                 else dv.new_zeros(dv.shape[0], self.v_dim))
        q_hat = self.head_q(dl)
        # 視覚サーボ(出力への有界な残差)。v_obs が無い step は 0。
        if self.servo_K is not None and v_obs is not None:
            g_obs = (v_obs - self.dec_mu) @ self.dec_W
            g_hat = (v_hat - self.dec_mu) @ self.dec_W
            d = g_obs - g_hat
            if mv_t is not None:
                d = d * mv_t.unsqueeze(-1)
            servo = self.servo_sigma * torch.tanh(
                self.servo_K(d) / self.servo_scale)
        else:
            servo = None
        if comp is None:
            comp = (v_hat.new_zeros(v_hat.shape[0]) if comp_reduce == "none"
                    else v_hat.new_zeros(()))
        self._last_parts = parts
        self._last_servo = servo
        return [ht, hv, hu, hl], [dt, dv, du, dl], v_hat, q_hat, dl, comp

    def generate(self, C, l, T_len, use_prior=False, mask=None,
                 v_obs=None, mask_v=None, q_obs=None):
        """自由変数 C（dict of (B,T,r) または None）から系列を生成する。
        use_prior=True なら C を無視して毎 step 事前値 ĉ を使う（prior 生成）。

        mask: (B,T) の有効長マスク。与えると Complexity をパディング区間で落とす
        （2026-08-04 修正。従来はパディング区間の comp も足していた。予測損失は
        free_energy 側で常にマスク済みだったので、この修正は comp のみに効く。
        マスク無しの呼び出しは従来と完全に同一の値を返す）。"""
        B = l.shape[0]
        hs, ds = self.init_state(B)
        lang_t = self.T.W_l(l)
        prev_a = l.new_zeros(B, self.a_dim)
        prev_q = l.new_zeros(B, self.q_dim)
        V, Q, D, SV = [], [], [], []
        comp = l.new_zeros(())
        cparts = {k: l.new_zeros(()) for k in self.LAYERS}
        for t in range(T_len):
            cs = {} if use_prior else {k: C[k][:, t] for k in self.LAYERS}
            hs, ds, v_hat, q_hat, dl, cp = self.one_step(
                hs, ds, cs, lang_t, prev_a, prev_q,
                comp_reduce=("mean" if mask is None else "none"),
                # ★ff_delay: 前向き経路だけ 1 step 遅らせる（head の狙いは t のまま）
                v_obs=(None if v_obs is None else
                       (v_obs[:, t] if not self.ff_delay else
                        (torch.zeros_like(v_obs[:, 0]) if t == 0
                         else v_obs[:, t - 1]))),
                mv_t=(None if mask_v is None else
                      (mask_v[:, t] if not self.ff_delay else
                       (torch.zeros_like(mask_v[:, 0]) if t == 0
                        else mask_v[:, t - 1]))),
                q_obs=(None if q_obs is None else
                       (q_obs[:, t] if not self.ff_delay else
                        (torch.zeros_like(q_obs[:, 0]) if t == 0
                         else q_obs[:, t - 1]))))
            if mask is None:
                comp = comp + cp
                for k, v_ in self._last_parts.items():
                    cparts[k] = cparts[k] + v_
            else:
                m_t = mask[:, t]
                comp = comp + (cp * m_t).mean()
                for k, v_ in self._last_parts.items():
                    cparts[k] = cparts[k] + (v_ * m_t).mean()
            V.append(v_hat); Q.append(q_hat); D.append(dl)
            sv = self._last_servo
            SV.append(torch.zeros_like(v_hat[:, :self.a_dim]) if sv is None else sv)
            prev_a = self.action_point(dl) + SV[-1]  # 次 step の V へのブリッジ
            prev_q = q_hat
        self.last_cparts = cparts
        self.last_servo_seq = torch.stack(SV, 1)
        return torch.stack(V, 1), torch.stack(Q, 1), torch.stack(D, 1), comp

    def free_energy(self, C, l, v, q, a, mask, mask_v, lambda_a=1.0,
                    use_prior=False, lambda_v=1.0, lambda_q=1.0,
                    mask_comp=True):
        """mask_comp=True(2026-08-04 以降の既定)で Complexity もパディングを除外する。
        False にすると従来(v1)の挙動と厳密に一致する。"""
        T_len = v.shape[1]
        v_hat, q_hat, d_low, comp = self.generate(
            C, l, T_len, use_prior, mask=(mask if mask_comp else None),
            q_obs=(q if self.feedforward else None),
            # ★視覚サーボも観測を要る(2026-08-10 修正。err_to_low だけを見ていたので
            #   サーボ有効時に v_obs=None となり、学習中はサーボが一度も発火しなかった)。
            v_obs=(v if (self.err_to_low or self.obs_to_low or self.feedforward
                         or self.servo_K is not None) else None),
            mask_v=(mask_v if (self.err_to_low or self.obs_to_low
                               or self.ff_gate_v
                               or self.servo_K is not None) else None))
        mv = mask * mask_v
        nv = mv.sum().clamp(min=1.0)
        nq = mask.sum().clamp(min=1.0)
        if self.head_v is None:
            # ★feedforward: 視覚は予測対象ではない。指標としても 0 を出す。
            loss_v = v.new_zeros(())
        elif self.v_proj is None:
            dv2 = (v - v_hat).pow(2).sum(-1) / self.v_dim
            loss_v = (dv2 * mv).sum() / nv
        else:
            dv2 = ((v - v_hat) @ self.v_proj).pow(2).sum(-1) / self.v_proj.shape[1]
            loss_v = (dv2 * mv).sum() / nv
        loss_q = (((q - q_hat).pow(2).sum(-1) / self.q_dim) * mask).sum() / nq
        a_t = a - self.last_servo_seq if self.servo_K is not None else a
        nll = self.action_nll(d_low.reshape(-1, self.d_low),
                              a_t.reshape(-1, self.a_dim))
        loss_a = (nll.view(v.shape[0], T_len) * mask).sum() / nq
        # 参考指標: 行動の代表点と教師の 2 乗誤差（NLL は下限が無いので目で見る用）
        with torch.no_grad():
            ap = self.action_point(d_low.reshape(-1, self.d_low)).view_as(a)
            if self.servo_K is not None:
                ap = ap + self.last_servo_seq
            mse_a = (((a - ap).pow(2).sum(-1) / self.a_dim) * mask).sum() / nq
        # lambda_v=0 で「視覚予測を学習しない」ablation になる（loss_v は指標として残す）
        total = lambda_v * loss_v + lambda_q * loss_q + lambda_a * loss_a + comp
        out = {"total": total, "loss_v": loss_v, "loss_q": loss_q,
               "loss_a": loss_a, "comp": comp, "mse_a": mse_a}
        for k, v_ in self.last_cparts.items():
            out[f"comp_{k}"] = v_.detach()
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
