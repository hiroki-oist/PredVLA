"""バッチ ER。B 本のロールアウトを 1 プロセスで同時に回す。2026-08-03、ユーザー指示。

## なぜ要るか

compile 後の 1 制御 step は 169.7 ms（実測）で、これは **B=1 の帯域下限**そのもの。

    重み 675,732 × 4B = 2.70MB を fwd+bwd で約 3 回読む = 8.1MB
    1 コアの L3 帯域 40〜55GB/s → 150〜200 µs / タイムステップ
    実測 153〜206 µs                                     ← 一致

行列ベクトル積（B=1）では重み 1 個につき積和 1 回しかせず、演算器は待っている。
B 本まとめれば重みを 1 回読んで B 回使えるので、1 サンプルあたりの時間が下がる。
実測（bench_er_batch.py、reps 9、threads 1）:

     B   1 反復     1 サンプル   1 タイムステップ   B=1 比
     1    9.8ms      9.82ms          245µs        1.00x
     4   17.5ms      4.36ms          109µs        2.25x
     8   26.2ms      3.27ms           82µs        3.00x
    16   38.0ms      2.38ms           59µs        4.13x
    32   37.9ms      1.18ms           30µs        8.29x
    64  119.9ms      1.87ms           47µs        5.24x   ← 作業セットが溢れる

さらに RSS が下がる（実測 rss_split.py）。評価 1 本の RSS 内訳は

    プロセス固定ぶん（torch + libero + model + frontend）  1,148 MB
    env 1 個目（EGL/レンダラの初期化を含む一度きり）        +1,093 MB
    env 2 個目以降                                          +300 MB / 個
    → RSS(B) ≈ 1,941 + 300·B   MB

    B= 1   2,241 MB（1 エピソードあたり 2,241 MB）
    B=10   4,941 MB（1 エピソードあたり   494 MB）

**両機とも CPU が半分遊んでいて RAM で頭打ち**（Fujiwara load 11/24、taketomi
load 32/64、どちらも swap を使用中）なので、ここが実際の律速。

## 設計

**1 枠 = 1 タスク。** B=10 で 10 タスクに 1 枠ずつ割り当て、各枠がその タスク の
ロールアウトを順に消化する。こうすると

  ・枠ごとに env は作りっぱなし（bddl も言語ベクトルも変わらない）
  ・終わった枠は次の初期状態に reset するだけ。詰め直しが env 再生成を伴わない
  ・枠ごとに有効長 n・マスク・窓の先頭確定のタイミングが独立なので、
    エピソード長のばらつきで無駄計算が出ない（横並びで待たせない）

## ERRollout（固定窓）との数値的な関係

**サンプルごとに厳密に一致するように書いてある。** 要点は 2 つ:

  ① 目的関数はバッチ方向を **和**にし、正規化は各サンプル内で閉じる
     平均にすると 1 サンプルの勾配が 1/B になり、Adam の eps=1e-4
     （既定の 1 万倍）が相対的に B 倍効いて更新量が変わる。
     model.one_step(..., comp_reduce="none") で complexity も (B,) で受ける。
  ② 枠ごとの添字（有効長 n、事前値に使う状態、行動の取り出し位置）を
     すべて gather / where で枠別に扱う

検証は B=1 と B=4 が fp64 でビット一致するかで行った。
"""
import torch

from .model import PredVLA


class ERBatch:
    """B 枠の ER。枠ごとに独立したエピソードを走らせる。

        er = ERBatch(model, l_vecs, window=40, n_itr=20, lr=0.1, device="cpu")
        for each control step:
            a = er.step(v, q, mv)      # v (B,v_dim) / q (B,q_dim) / mv (B,)
            # env を枠ごとに進める。終わった枠は er.reset_slot(i) して次へ
    """

    def __init__(self, model, l_vecs, window=40, n_itr=20, lr=0.1,
                 device="cpu", a_init="prior", vision_stride=4,
                 er_w=None, er_lambda_v=1.0, er_lambda_q=1.0,
                 fresh_c=False, er_freeze=(), er_opt="adam", er_eps=1e-4,
                 er_v_proj=None):
        self.m = model
        self.w_train = dict(model.w)
        if er_w is None:
            self.w_er = dict(model.w)
        elif isinstance(er_w, dict):
            self.w_er = {k: float(er_w.get(k, model.w[k])) for k in model.w}
        else:
            self.w_er = {k: float(er_w) for k in model.w}
        self.lam_v = float(er_lambda_v)
        self.lam_q = float(er_lambda_q)
        # ★P0a（2026-08-12）: 視覚誤差から「固有感覚 q から線形に決まる部分空間」を落とす。
        #   C5 で ‖Δv‖ の約半分が腕の姿勢（= q と冗長）だと分かったため。
        #   er_v_proj は (v_dim, k) の正規直交基底 U。誤差 e に対し e − U(Uᵀe) を使う。
        #   ★大きさは変えず向きだけを変える。射影は線形なので再学習は要らない。
        self.v_U = None
        if er_v_proj is not None:
            U = (er_v_proj if torch.is_tensor(er_v_proj)
                 else torch.as_tensor(er_v_proj))
            self.v_U = U.to(device=device, dtype=torch.float32)
        # 記憶なし ER（2026-08-04、ユーザー指示）: 毎制御 step、ER の反復を始める
        # 前に窓内**全** slot の c を prior 軌道の ĉ に貼り直す。comp が毎 step
        # 厳密に 0 から始まり、反復は純粋にデータ方向へ動く。前 step までの
        # 適応（comp の在庫）は持ち越されない。
        self.fresh_c = bool(fresh_c)
        # ER で更新しない層（2026-08-04、ユーザー仮説: 視覚の内部状態を観測に
        # 引っ張らせない）。凍結層の c は prior 初期値のまま。lv の勾配は
        # A→V ブリッジ経由で行動枝の c には引き続き流れる。
        self.freeze = frozenset(er_freeze)
        assert self.freeze <= set(PredVLA.LAYERS), f"不明な層: {self.freeze}"
        # 最適化器の消去実験(2026-08-06、ユーザー指摘)。
        #   adam: 従来(eps は er_eps。既定 1e-4 = LibPvrnn 由来の大きい値)
        #   sgd : 生勾配 SGD。勾配は J_o の行空間に閉じる(Jacobian 解析で実証)ので、
        #         「観測不変方向への漏れ」を構造的に断つ対照になる
        self.er_opt = str(er_opt)
        self.er_eps = float(er_eps)
        self.dev = device
        self.W, self.n_itr, self.lr = int(window), int(n_itr), float(lr)
        self.a_init = a_init
        self.vision_stride = int(vision_stride)

        L = l_vecs.to(device)
        self.B = int(L.shape[0])
        B, W = self.B, self.W
        self.l = L
        self.lang_t = model.T.W_l(L)                 # (B, d_top)
        self.ar = torch.arange(B, device=device)

        hs, ds = model.init_state(B)
        self.head_h = [x.detach().clone() for x in hs]
        self.head_d = [x.detach().clone() for x in ds]
        self.head_a = torch.zeros(B, model.a_dim, device=device)
        self.head_q = torch.zeros(B, model.q_dim, device=device)
        self.c = {k: torch.zeros(B, W, model.r_of[k], device=device)
                  for k in PredVLA.LAYERS}
        self.V = torch.zeros(B, W, model.v_dim, device=device)
        self.Q = torch.zeros(B, W, model.q_dim, device=device)
        self.MV = torch.zeros(B, W, device=device)
        self.M = torch.zeros(B, W, device=device)
        self.n = torch.zeros(B, dtype=torch.long, device=device)
        self._cache = None
        self.last_eps_v = torch.zeros(B, device=device)
        self.last_eps_q = torch.zeros(B, device=device)
        # ★ER の「効き」の記録（2026-08-11、診断用）。record_sens=True のときだけ埋まる。
        #   last_dc … ER で c(最終 slot、全層を連結)がどれだけ動いたか ‖Δc‖
        #   last_da … その結果、出力される行動がどれだけ変わったか ‖a_post − a_pre‖
        #   ER 前の c で余分に 1 回前向きするので step あたり約 1.3 倍のコスト。
        self.record_sens = False
        # ★ER の内側反復ごとの自由エネルギーを記録する（2026-08-18 追加、診断用）。
        #   record_E=True のときだけ E_trace に (制御 step, 反復 i, E, ‖∂E/∂c‖, ‖Δc‖) が積まれる。
#   反復 i は 0..n_itr（n_itr は最後の更新後の値なので n_itr+1 点入る）。
        #   既定 False なので学習・評価の挙動は従来と完全に同一。
        self.record_E = False
        self.E_trace = []
        self._t_ctrl = 0
        self.last_dc = torch.zeros(B, device=device)
        self.last_da = torch.zeros(B, device=device)
        # ★誤差ベクトル(192 次元)。物体位置の復号方向に射影して「希釈」を検証するため
        #   （2026-08-11。スカラーの eps_v では取り落としに反応しなかったが、
        #     v から物体位置は 7〜14mm で線形復元できるので、成分で見る必要がある）
        self.last_ev_vec = torch.zeros(B, model.v_dim, device=device)
        # 枠ごとの初期状態（reset_slot で書き戻す用）
        h1, d1 = model.init_state(1)
        self._init_h = [x.detach().clone() for x in h1]
        self._init_d = [x.detach().clone() for x in d1]

    # ---------------- 枠の入れ替え ----------------
    @torch.no_grad()
    def reset_slot(self, i):
        """枠 i のエピソードを終了して初期化する。次の初期状態から走り直せる。

        リセット後は n[i]=0 なので、次の step で事前値に使う添字が -1 になり
        **キャッシュを一切参照しない**（古い枠の状態が漏れない）。
        """
        for l in range(4):
            self.head_h[l][i] = self._init_h[l][0]
            self.head_d[l][i] = self._init_d[l][0]
        self.head_a[i] = 0.0
        self.head_q[i] = 0.0
        for k in PredVLA.LAYERS:
            self.c[k][i] = 0.0
        self.V[i] = 0.0
        self.Q[i] = 0.0
        self.MV[i] = 0.0
        self.M[i] = 0.0
        self.n[i] = 0

    # ---------------- 窓のロール ----------------
    def _roll(self, cs, want_states=False):
        """必ず窓 W step ぶん前進する。有効長の外はマスクで落とす。
        comp は枠ごとに (B,) で受ける（バッチ平均にすると Adam の eps が効く）。"""
        hs = [x for x in self.head_h]
        ds = [x for x in self.head_d]
        pa, pq = self.head_a, self.head_q
        V, Q, D = [], [], []
        HS, DS, PA, PQ = [], [], [], []
        comp = self.M.new_zeros(self.B)
        for t in range(self.W):
            step_c = {k: cs[k][:, t] for k in PredVLA.LAYERS}
            # ★err_to_low / obs_to_low のとき、観測（誤差 or 生の視覚特徴）を
            #   順方向に入れる。学習時と同じ経路にしないと train/test 不一致になる。
            _need_obs = (getattr(self.m, "err_to_low", False)
                         or getattr(self.m, "obs_to_low", False)
                         or getattr(self.m, "servo_K", None) is not None)
            # ★feedforward（はしごの L3 以降）: 生の v と q を前向きに入れる。
            #   ff_delay なら前向き経路だけ 1 step 遅らせる（窓頭 t=0 は 0 埋め）。
            #   ff_gate_v なら前向きの視覚だけを mask_v で間引く。
            _ff = getattr(self.m, "feedforward", False)
            _fd = getattr(self.m, "ff_delay", False)
            _fg = getattr(self.m, "ff_gate_v", False)
            hs, ds, v_hat, q_hat, dl, cp = self.m.one_step(
                hs, ds, step_c, self.lang_t, pa, pq, comp_reduce="none",
                v_obs=((self.V[:, t] if not _fd else
                        (torch.zeros_like(self.V[:, 0]) if t == 0
                         else self.V[:, t - 1]))
                       if (_need_obs or _ff) else None),
                mv_t=((self.MV[:, t] if not _fd else
                       (torch.zeros_like(self.MV[:, 0]) if t == 0
                        else self.MV[:, t - 1]))
                      if (_need_obs or _fg) else None),
                q_obs=((self.Q[:, t] if not _fd else
                        (torch.zeros_like(self.Q[:, 0]) if t == 0
                         else self.Q[:, t - 1]))
                       if _ff else None))
            sv = getattr(self.m, "_last_servo", None)
            comp = comp + cp * self.M[:, t]
            V.append(v_hat); Q.append(q_hat); D.append(dl)
            pa = self.m.action_point(dl) + (0.0 if sv is None else sv)
            pq = q_hat
            if want_states:
                HS.append(hs); DS.append(ds); PA.append(pa); PQ.append(pq)
        out = (torch.stack(V, 1), torch.stack(Q, 1), torch.stack(D, 1), comp)
        if not want_states:
            return out
        # (W,B,d) に積み直す。枠ごとに違う添字で gather するため
        HSs = [torch.stack([HS[t][l] for t in range(self.W)], 0) for l in range(4)]
        DSs = [torch.stack([DS[t][l] for t in range(self.W)], 0) for l in range(4)]
        return out + (HSs, DSs, torch.stack(PA, 0), torch.stack(PQ, 0))

    def _energy(self, cs):
        v_hat, q_hat, _, comp = self._roll(cs)
        mv = self.MV * self.M
        nv = mv.sum(1).clamp(min=1.0)
        nq = self.M.sum(1).clamp(min=1.0)
        ev = self.V - v_hat
        if self.v_U is not None:
            ev = ev - (ev @ self.v_U) @ self.v_U.T      # ★腕部分空間を落とす
        lv = ((ev.pow(2).sum(-1) / self.m.v_dim) * mv).sum(1) / nv
        lq = (((self.Q - q_hat).pow(2).sum(-1) / self.m.q_dim) * self.M).sum(1) / nq
        # ★バッチ方向は和。こうするとサンプル i の勾配が単体走行と厳密に一致する
        return (self.lam_v * lv + self.lam_q * lq + comp).sum()

    # ---------------- 状態の取り出し ----------------
    @torch.no_grad()
    def _state_at(self, j):
        """枠ごとに「窓の先頭から j[i] step 進めた後」の状態。j[i]<0 は先頭そのもの。"""
        if self._cache is None:
            return (list(self.head_h), list(self.head_d), self.head_a, self.head_q)
        HSs, DSs, PAs, PQs = self._cache
        sel = j.clamp(min=0)
        head = (j < 0)[:, None]
        hs = [torch.where(head, self.head_h[l], HSs[l][sel, self.ar])
              for l in range(4)]
        ds = [torch.where(head, self.head_d[l], DSs[l][sel, self.ar])
              for l in range(4)]
        pa = torch.where(head, self.head_a, PAs[sel, self.ar])
        pq = torch.where(head, self.head_q, PQs[sel, self.ar])
        return (hs, ds, pa, pq)

    @torch.no_grad()
    def _set_prior(self, idx, st):
        """枠ごとに位置 idx[i] の自由変数へ事前値 ĉ を入れる。"""
        if self.a_init == "zero":
            for k in PredVLA.LAYERS:
                self.c[k][self.ar, idx] = 0.0
            return
        m = self.m
        # ★feedforward（はしごの L3 以降）は自由変数 c の経路を持たない。
        #   prior は None を返すので事前値を書く対象が無い
        #   （L2 以降は n_itr=0 なので c は読まれない）。
        if getattr(m, "feedforward", False):
            return
        hs, ds, pa, pq = st
        ht, hv, hu, hl = hs
        dt, dv, du, dl = ds
        ct = m.T.prior(None, dt)
        _, dt2 = m.T.step(ht, dt, ct, None, self.lang_t)
        cv = m.V.prior(dt2, dv)
        v_bridge_in = () if getattr(m, "ab_no_av_bridge", False) else (pa, pq)
        _, dv2 = m.V.step(hv, dv, cv, dt2, None, v_bridge_in)
        up_bridge_in = () if getattr(m, "ab_no_pb", False) else (m.W_pb(dv2),)
        cu = m.A_up.prior(dt2, du)
        _, du2 = m.A_up.step(hu, du, cu, dt2, None, up_bridge_in)
        cl = m.A_low.prior(du2, dl)
        for k, val in (("t", ct), ("v", cv), ("up", cu), ("low", cl)):
            self.c[k][self.ar, idx] = val

    @torch.no_grad()
    def _reanchor(self):
        """窓内全 slot の c を prior 軌道の ĉ に貼り直す（fresh_c=True 用）。

        先頭状態から W step、各 step で ①その時点の prior ĉ を c に書き、
        ②その c で 1 step 前進、を繰り返す。書き終わると c ≡ prior 軌道の ĉ に
        なるので、直後のロールで comp は厳密に 0。有効長 n を超える padding 区間
        にも書くが、損失・comp ともマスクで落ちるので結果に影響しない。
        _set_prior と one_step で階層の再帰は二重計算になるが、コストは
        ロール約 2 回ぶん（n_itr=20 の 1 割）なので許容する。"""
        if getattr(self.m, "feedforward", False):
            return           # ★c を持たないので貼り直す対象が無い
        hs = [x for x in self.head_h]
        ds = [x for x in self.head_d]
        pa, pq = self.head_a, self.head_q
        full = torch.full_like(self.n, 0)
        for t in range(self.W):
            self._set_prior(full + t, (hs, ds, pa, pq))
            step_c = {k: self.c[k][:, t] for k in PredVLA.LAYERS}
            hs, ds, v_hat, q_hat, dl, _ = self.m.one_step(
                hs, ds, step_c, self.lang_t, pa, pq, comp_reduce="none")
            pa = self.m.action_point(dl)
            pq = q_hat

    @torch.no_grad()
    def _commit(self, need):
        """満杯の枠だけ先頭 1 step を確定し、バッファを左に 1 ずらす。
        先頭を 1 step 進めた後の状態は直前のロールのキャッシュ [0] にある。"""
        if not bool(need.any()):
            return
        HSs, DSs, PAs, PQs = self._cache
        nb = need[:, None]
        for l in range(4):
            self.head_h[l] = torch.where(nb, HSs[l][0], self.head_h[l]).clone()
            self.head_d[l] = torch.where(nb, DSs[l][0], self.head_d[l]).clone()
        self.head_a = torch.where(nb, PAs[0], self.head_a).clone()
        self.head_q = torch.where(nb, PQs[0], self.head_q).clone()
        n3 = need[:, None, None]
        for k in PredVLA.LAYERS:
            sh = torch.cat([self.c[k][:, 1:], self.c[k][:, :1] * 0.0], 1)
            self.c[k] = torch.where(n3, sh, self.c[k])
        self.V = torch.where(n3, torch.cat([self.V[:, 1:],
                                            self.V[:, :1] * 0.0], 1), self.V)
        self.Q = torch.where(n3, torch.cat([self.Q[:, 1:],
                                            self.Q[:, :1] * 0.0], 1), self.Q)
        self.MV = torch.where(nb, torch.cat([self.MV[:, 1:],
                                             self.MV[:, :1] * 0.0], 1), self.MV)

    # ---------------- 1 制御 step ----------------
    def step(self, v, q, mv):
        """v (B,v_dim) は最新の視覚特徴、mv (B,) は今 step に視覚が更新された枠。
        mv=0 の枠は 1 つ前の値をそのまま使う（ERRollout と同じ扱い）。"""
        B, W, ar = self.B, self.W, self.ar
        n_prev = self.n.clone()

        # ① 事前値に使う状態は「n_prev step 進めた後」= キャッシュ添字 n_prev-1
        #    満杯の枠でも同じ式でよい（ずらした後の窓から見ると W-1 step 進めた後）
        st = self._state_at(n_prev - 1)
        # ② 満杯の枠だけ先頭を確定して左シフト
        self._commit(n_prev >= W)
        # ③ 観測を書く
        idx = n_prev.clamp(max=W - 1)
        with torch.no_grad():
            self.Q[ar, idx] = q.to(self.dev)
            prev = self.V[ar, (idx - 1).clamp(min=0)]
            keep = ((mv > 0) | (idx > 0))[:, None]
            self.V[ar, idx] = torch.where(
                (mv > 0)[:, None], v.to(self.dev),
                torch.where(keep, prev, torch.zeros_like(prev)))
            self.MV[ar, idx] = mv.to(self.dev)
            self.M[ar, idx] = 1.0
        self.n = (n_prev + 1).clamp(max=W)
        # ④ 事前値
        self._set_prior(idx, st)

        # ⑤ ER（重みは凍結、c だけ更新）
        c_pre = ({k: self.c[k].clone() for k in PredVLA.LAYERS}
                 if (self.record_sens and self.n_itr > 0) else None)
        if self.n_itr > 0:
            if self.fresh_c:
                self._reanchor()      # 窓内全 slot を prior に貼り直してから推論
            self.m.w = self.w_er
            free = {k: self.c[k].clone().requires_grad_(True)
                    for k in PredVLA.LAYERS if k not in self.freeze}
            fixed = {k: self.c[k] for k in self.freeze}
            opt = (torch.optim.SGD(list(free.values()), lr=self.lr)
                   if self.er_opt == "sgd" else
                   torch.optim.Adam(list(free.values()), lr=self.lr,
                                    eps=self.er_eps))
            for _it in range(self.n_itr):
                E = self._energy({**free, **fixed})
                opt.zero_grad(set_to_none=True)
                E.backward()
                if self.record_E:
                    # ★勾配ノルムと実際の歩幅を測る（2026-08-18 追加）。
                    #   「向きが違う」のか「歩幅が大きすぎる」のかを分けるため。
                    gn = float(torch.sqrt(sum(
                        (t.grad.detach() ** 2).sum() for t in free.values()
                        if t.grad is not None)))
                    _before = [t.detach().clone() for t in free.values()]
                opt.step()
                if self.record_E:
                    dn = float(torch.sqrt(sum(
                        ((t.detach() - b) ** 2).sum()
                        for t, b in zip(free.values(), _before))))
                    self.E_trace.append((self._t_ctrl, _it,
                                         float(E.detach()), gn, dn))
            if self.record_E:
                with torch.no_grad():          # 最後の更新後の値も 1 点入れる
                    self.E_trace.append((self._t_ctrl, self.n_itr,
                                         float(self._energy({**free, **fixed})),
                                         float("nan"), float("nan")))
            self._t_ctrl += 1
            for k in free:
                self.c[k] = free[k].detach()
            self.m.w = self.w_train

        # ⑥ 最終ロール → 行動と状態キャッシュ
        with torch.no_grad():
            v_hat, q_hat, D, _, HSs, DSs, PAs, PQs = self._roll(
                self.c, want_states=True)
            self._cache = (HSs, DSs, PAs, PQs)
            last = (self.n - 1).clamp(min=0)
            # PAs は _roll 内でサーボを足した後の行動(サーボ無しなら従来と同一)
            a = PAs[last, ar]
            self.last_eps_v = ((self.V[ar, last] - v_hat[ar, last]).pow(2)
                               .sum(-1) / self.m.v_dim)
            self.last_eps_q = ((self.Q[ar, last] - q_hat[ar, last]).pow(2)
                               .sum(-1) / self.m.q_dim)
            self.last_ev_vec = self.V[ar, last] - v_hat[ar, last]
            # ★ER の効きを測る: ER 前の c で同じ前向きを回し、行動の差を取る。
            #   「修正が起きていない」のか「起きているが小さい」のかを分けるため。
            if c_pre is not None:
                *_, PAs_pre, _ = self._roll(c_pre, want_states=True)
                self.last_da = (a - PAs_pre[last, ar]).norm(dim=-1)
                self.last_dc = torch.sqrt(sum(
                    (self.c[k][ar, last] - c_pre[k][ar, last]).pow(2).sum(-1)
                    for k in PredVLA.LAYERS))
        return a.cpu().numpy()
