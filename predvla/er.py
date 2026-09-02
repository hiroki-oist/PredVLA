"""誤差回帰（ER）によるロールアウト。LibPvrnn の online_error_regression に対応。

LibPvrnn の手順（model.cpp: erIteration / layer.cpp: erSlideWindow）
  窓サイズ W。毎時刻:
    1. 窓をスライドし、末尾の自由変数を a_init で初期化（"prior" 推奨）
    2. n_itr 回:  窓の先頭の保存状態から前進 → 観測との誤差 → 窓内を BPTT
                  → 窓内の自由変数のみ Adam 更新（重みは凍結）
    3. 窓が溢れたら先頭を 1 step 確定させ、その状態を保存して進む

自由変数は c（PV-RNN の A に対応、分散なしの決定論版、低ランク U で h を直接ずらす）。
構造は T（共有上位、言語入力）+ V（視覚枝）+ A_up/A_low（行動枝）、V↔A ブリッジ付き。
観測は順方向に入らないので、世界からの情報はこの推論だけを通って入る。
"""
import torch

from .model import PredVLA


class ERRollout:
    """1 エピソードぶんの ER。

        er = ERRollout(model, l_vec, window=20, n_itr=5, lr=0.1, device=...)
        for each control step:
            a = er.step(v_t, q_t)   # v_t は視覚が更新された step のみ、他は None
            env.step(a)
    """

    def __init__(self, model, l_vec, window=20, n_itr=5, lr=0.1,
                 device="cpu", a_init="prior", vision_stride=4,
                 er_w=None, er_beta=None, er_lambda_v=1.0, er_lambda_q=1.0,
                 er_opt="adam", er_eps=1e-4,
                 fixed_window=False):
        """er_w / er_beta: ER 専用のメタプライア（LibPvrnn の [er] の w / beta）。

        学習時の w とは独立に設定できる。意味も違う:
          学習時 w  自由変数が系列ごとの情報をどれだけ吸収するか（prior の質を決める）
          ER 時 w   推論の信頼領域。観測をどれだけ信じ、prior をどれだけ信じるか
            小 → 変化に速く適応するが、壊れた観測にも引きずられる
            大 → 汚染された観測を無視できるが、適応が遅い
        er_beta は窓の先頭 step 用（LibPvrnn は初回窓の間だけ erW[0]=erBeta）。
        None なら学習時の値をそのまま使う。
        er_w は数値（全層に一律の倍率ではなく絶対値）か dict。
        """
        self.m = model
        self.w_train = dict(model.w)
        if er_w is None:
            self.w_er = dict(model.w)
        elif isinstance(er_w, dict):
            self.w_er = {k: float(er_w.get(k, model.w[k])) for k in model.w}
        else:
            self.w_er = {k: float(er_w) for k in model.w}
        self.beta_er = None if er_beta is None else float(er_beta)
        # ER の目的関数の項ごとの重み（2026-08-01 追加、評価時のみ。既定 1.0 で従来と一致）
        #   E = er_lambda_v · lv + er_lambda_q · lq + comp
        # 動機: n_itr 5 の崩壊は「lq（毎 step、密）を下げるために lv（4 step に 1 回、疎）を
        # 犠牲にし、5 反復で打ち切るとその途中で止まる」ためという仮説の検証。
        # sp_s1 実測: lv 0.786(ER なし) → 1.269(n_itr 5) → 0.788(10) → 0.805(20)
        self.lam_v = float(er_lambda_v)
        self.lam_q = float(er_lambda_q)
        self.dev = device
        self.l = l_vec.to(device)[None]
        self.W, self.n_itr, self.lr = int(window), int(n_itr), float(lr)
        # ★ER の最適化器（2026-08-13）。er_batch と同じ選択肢にそろえる。
        #   既定 adam は従来と完全に同一。sgd は生勾配。
        self.er_opt, self.er_eps = str(er_opt), float(er_eps)
        self.a_init = a_init
        self.vision_stride = int(vision_stride)
        self.lang_t = model.T.W_l(self.l)          # 言語は共有上位 T に入る
        hs, ds = model.init_state(1)
        self.head_h = [x.detach().clone() for x in hs]
        self.head_d = [x.detach().clone() for x in ds]
        # ブリッジ（A → V）に渡す前 step の予測。窓の先頭時点の値
        self.head_a = torch.zeros(1, model.a_dim, device=device)
        self.head_q = torch.zeros(1, model.q_dim, device=device)
        self.buf_v, self.buf_q, self.buf_mv = [], [], []
        self.c = {k: torch.zeros(1, 0, model.r_of[k], device=device)
                  for k in PredVLA.LAYERS}
        self.last_eps_v = None
        self.last_eps_q = None
        self.t = 0

        # ---- 固定窓モード（2026-08-03、ユーザー提案。LibPvrnn と同じ持ち方）----
        # 動機: 既定モードは c が (1,n,r) で n=1→W と伸びるため、torch.compile が
        #   **W 通りの形を特殊化しようとして Dynamo のキャッシュ上限（既定 8）を超え、
        #   超えた分は黙って eager に落ちる**（実測で 8 回再コンパイルの警告）。
        #   最初から (1,W,r) を確保し、有効長 n をマスクで表せば形は 1 つになる。
        #
        # 併せてロール回数も減る。既定モードは 1 制御 step あたり
        #   事前値 1 + ER n_itr + 行動 1 + 先頭確定 1
        # だが、固定窓では **行動を出す最後のロールで各 step の状態を積んで返す**ので、
        # 事前値の生成と先頭確定はそのキャッシュから読める（ロール 2 回ぶん節約）。
        self.fixed = bool(fixed_window)
        if self.fixed:
            W = self.W
            self.c = {k: torch.zeros(1, W, model.r_of[k], device=device)
                      for k in PredVLA.LAYERS}
            self.V = torch.zeros(1, W, model.v_dim, device=device)
            self.Q = torch.zeros(1, W, model.q_dim, device=device)
            self.MV = torch.zeros(1, W, device=device)   # 視覚が更新された step
            self.M = torch.zeros(1, W, device=device)    # 有効長のマスク
            self.n = 0                                   # 有効長
            self._cache = None    # 直前のロールの (HS, DS, PA, PQ)

    def _roll(self, cs, n_steps):
        """窓の先頭の保存状態から n_steps 前進する（微分可能）。
        cs は dict（各 (1,n,r)）。値が None の層は事前値をそのまま使う。"""
        hs = [x for x in self.head_h]
        ds = [x for x in self.head_d]
        pa, pq = self.head_a, self.head_q
        V, Q, D = [], [], []
        comp = self.l.new_zeros(())
        for t in range(n_steps):
            step_c = {k: (None if cs[k] is None else cs[k][:, t])
                      for k in PredVLA.LAYERS}
            hs, ds, v_hat, q_hat, dl, cp = self.m.one_step(
                hs, ds, step_c, self.lang_t, pa, pq)
            comp = comp + cp
            V.append(v_hat); Q.append(q_hat); D.append(dl)
            pa = self.m.action_point(dl)
            pq = q_hat
        return (torch.stack(V, 1), torch.stack(Q, 1), torch.stack(D, 1),
                comp, hs, ds, pa, pq)

    # ================= 固定窓モード =================
    def _roll_fixed(self, cs, want_states=False):
        """必ず窓 W step ぶん前進する。有効長の外はマスクで落とす。

        既定モードの _roll との違いは 2 つだけ:
          ① n_steps が常に W（形が 1 つ = torch.compile が 1 グラフで済む）
          ② want_states=True なら各 step の状態も返す
             （事前値の生成と先頭確定に使い回す）
        パディング区間も計算はするが、comp はマスクで 0 にし、v̂/q̂ も損失側で
        落とすので結果には効かない。

        ★want_states を分けている理由（2026-08-03 08:20）
          状態は 40 step × (hs 4 + ds 4 + pa + pq) = 400 テンソルになる。
          これを使うのは **行動を出す最後の 1 回だけ**で、ER ループの n_itr 回は
          捨てている。ところがコンパイル済み関数の「返り値」は Inductor が
          削除できないので、20 回すべてで 400 個を実体化していた
          （1 制御 step あたり 8,000 個）。フラグで分ければ ER ループ側の
          グラフから消える。Python の bool なので Dynamo は 2 通りに特殊化する
          （どちらも形は 1 通りなのでキャッシュ上限 8 には当たらない）。
        """
        hs = [x for x in self.head_h]
        ds = [x for x in self.head_d]
        pa, pq = self.head_a, self.head_q
        V, Q, D = [], [], []
        HS, DS, PA, PQ = [], [], [], []
        comp = self.l.new_zeros(())
        for t in range(self.W):
            step_c = {k: cs[k][:, t] for k in PredVLA.LAYERS}
            hs, ds, v_hat, q_hat, dl, cp = self.m.one_step(
                hs, ds, step_c, self.lang_t, pa, pq)
            comp = comp + cp * self.M[0, t]      # ★パディング区間を落とす
            V.append(v_hat); Q.append(q_hat); D.append(dl)
            pa = self.m.action_point(dl)
            pq = q_hat
            if want_states:
                HS.append(list(hs)); DS.append(list(ds))
                PA.append(pa); PQ.append(pq)
        out = (torch.stack(V, 1), torch.stack(Q, 1), torch.stack(D, 1), comp)
        return out + ((HS, DS, PA, PQ) if want_states else ())

    def _state_at(self, j):
        """窓の先頭から j step 進めた後の状態 (hs, ds, pa, pq)。
        j=0 は先頭そのもの。j>=1 は直前のロールのキャッシュから読む。"""
        if j == 0:
            return (list(self.head_h), list(self.head_d), self.head_a, self.head_q)
        HS, DS, PA, PQ = self._cache
        return (HS[j - 1], DS[j - 1], PA[j - 1], PQ[j - 1])

    @torch.no_grad()
    def _set_prior_c(self, idx, st):
        """位置 idx の自由変数に事前値 ĉ を入れる。st は idx step 進めた後の状態。
        階層順に評価するのは _append_prior_c と同じ。"""
        if self.a_init == "zero":
            for k in PredVLA.LAYERS:
                self.c[k][:, idx] = 0.0
            return
        m = self.m
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
            self.c[k][:, idx] = val

    @torch.no_grad()
    def _commit_fixed(self):
        """窓が満杯なので先頭 1 step を確定し、バッファを左に 1 ずらす。
        先頭を 1 step 進めた後の状態は直前のロールのキャッシュ [0] にある
        （＝既定モードの _commit_head が毎回やっていた 1 step ロールが不要）。"""
        HS, DS, PA, PQ = self._cache
        self.head_h = [x.detach().clone() for x in HS[0]]
        self.head_d = [x.detach().clone() for x in DS[0]]
        self.head_a = PA[0].detach().clone()
        self.head_q = PQ[0].detach().clone()
        z1 = {k: torch.zeros(1, 1, self.m.r_of[k], device=self.dev)
              for k in PredVLA.LAYERS}
        for k in PredVLA.LAYERS:
            self.c[k] = torch.cat([self.c[k][:, 1:], z1[k]], 1)
        self.V = torch.cat([self.V[:, 1:],
                            torch.zeros(1, 1, self.m.v_dim, device=self.dev)], 1)
        self.Q = torch.cat([self.Q[:, 1:],
                            torch.zeros(1, 1, self.m.q_dim, device=self.dev)], 1)
        self.MV = torch.cat([self.MV[:, 1:],
                             torch.zeros(1, 1, device=self.dev)], 1)

    def _step_fixed(self, v_t, q_t):
        """固定窓での 1 制御 step。既定の step() と数学的に同じものを返す。"""
        W = self.W
        if self.n < W:
            idx = self.n
            st = self._state_at(idx)          # idx step 進めた後
            self.n += 1
            self.M[0, idx] = 1.0
        else:
            idx = W - 1
            # 先頭をずらす前に「W step 進めた後」を取る。ずらした後の窓から見れば
            # これは W-1 step 進めた後にあたる（＝位置 W-1 の事前値に使う状態）
            st = self._state_at(W)
            self._commit_fixed()

        self.Q[0, idx] = q_t.to(self.dev)
        if v_t is None:
            self.V[0, idx] = self.V[0, idx - 1] if idx > 0 else 0.0
            self.MV[0, idx] = 0.0
        else:
            self.V[0, idx] = v_t.to(self.dev)
            self.MV[0, idx] = 1.0
        self._set_prior_c(idx, st)

        if self.n_itr > 0:
            self.m.w = self.w_er
            free = {k: self.c[k].clone().requires_grad_(True)
                    for k in PredVLA.LAYERS}
            opt = (torch.optim.SGD(list(free.values()), lr=self.lr)
                   if self.er_opt == "sgd" else
                   torch.optim.Adam(list(free.values()), lr=self.lr,
                                    eps=self.er_eps))
            mv = self.MV * self.M
            nv = mv.sum().clamp(min=1.0)
            nq = self.M.sum().clamp(min=1.0)
            for _ in range(self.n_itr):
                v_hat, q_hat, _, comp = self._roll_fixed(free)
                lv = (((self.V - v_hat).pow(2).sum(-1) / self.m.v_dim)
                      * mv).sum() / nv
                lq = (((self.Q - q_hat).pow(2).sum(-1) / self.m.q_dim)
                      * self.M).sum() / nq
                E = self.lam_v * lv + self.lam_q * lq + comp
                opt.zero_grad(set_to_none=True)
                E.backward()
                opt.step()
            for k in PredVLA.LAYERS:
                self.c[k] = free[k].detach()
            self.m.w = self.w_train

        with torch.no_grad():
            v_hat, q_hat, D, _, HS, DS, PA, PQ = self._roll_fixed(
                self.c, want_states=True)
            self._cache = (HS, DS, PA, PQ)
            a = self.m.action_point(D[:, idx])
            if self.MV[0, idx] > 0:
                self.last_eps_v = float(
                    (self.V[:, idx] - v_hat[:, idx]).pow(2).sum(-1).mean()
                    / self.m.v_dim)
            self.last_eps_q = float(
                (self.Q[:, idx] - q_hat[:, idx]).pow(2).sum(-1).mean()
                / self.m.q_dim)
        self.t += 1
        return a[0].cpu().numpy()

    # ================= 既定モード =================
    @torch.no_grad()
    def _append_prior_c(self):
        """末尾に足す c の事前値。a_init='prior' なら階層を順に評価して ĉ を作る。

        階層なので同 step 内に依存がある: T を進めて d^T が決まってから V の事前値、
        V を進めて PB が決まってから A_up の事前値、という順序で作る。
        """
        n = self.c["t"].shape[1]
        if self.a_init == "zero":
            add = {k: torch.zeros(1, 1, self.m.r_of[k], device=self.dev)
                   for k in PredVLA.LAYERS}
        else:
            if n == 0:
                hs, ds = list(self.head_h), list(self.head_d)
                pa, pq = self.head_a, self.head_q
            else:
                _, _, _, _, hs, ds, pa, pq = self._roll(self.c, n)
            m = self.m
            ht, hv, hu, hl = hs
            dt, dv, du, dl = ds
            ct = m.T.prior(None, dt)
            _, dt2 = m.T.step(ht, dt, ct, None, self.lang_t)
            cv = m.V.prior(dt2, dv)
            # ablation フラグ（ab_no_av_bridge / ab_no_pb）は model 側と同じ扱いにする
            v_bridge_in = () if getattr(m, "ab_no_av_bridge", False) else (pa, pq)
            _, dv2 = m.V.step(hv, dv, cv, dt2, None, v_bridge_in)
            up_bridge_in = () if getattr(m, "ab_no_pb", False) else (m.W_pb(dv2),)
            cu = m.A_up.prior(dt2, du)
            _, du2 = m.A_up.step(hu, du, cu, dt2, None, up_bridge_in)
            cl = m.A_low.prior(du2, dl)
            add = {"t": ct[:, None], "v": cv[:, None],
                   "up": cu[:, None], "low": cl[:, None]}
        for k in PredVLA.LAYERS:
            self.c[k] = torch.cat([self.c[k], add[k]], dim=1)

    @torch.no_grad()
    def _commit_head(self):
        """窓が溢れたので先頭 1 step を確定し、head 状態を進める。"""
        first = {k: self.c[k][:, :1] for k in PredVLA.LAYERS}
        _, _, _, _, hs, ds, pa, pq = self._roll(first, 1)
        self.head_h = [x.detach().clone() for x in hs]
        self.head_d = [x.detach().clone() for x in ds]
        self.head_a, self.head_q = pa.detach().clone(), pq.detach().clone()
        for k in PredVLA.LAYERS:
            self.c[k] = self.c[k][:, 1:]
        self.buf_v.pop(0); self.buf_q.pop(0); self.buf_mv.pop(0)

    def step(self, v_t, q_t):
        """1 制御 step。戻り値は env に送る行動 (a_dim,) の numpy 配列。"""
        if self.fixed:
            return self._step_fixed(v_t, q_t)
        self.buf_q.append(q_t.to(self.dev)[None])
        if v_t is None:
            self.buf_v.append(self.buf_v[-1] if self.buf_v
                              else torch.zeros(1, self.m.v_dim, device=self.dev))
            self.buf_mv.append(0.0)
        else:
            self.buf_v.append(v_t.to(self.dev)[None])
            self.buf_mv.append(1.0)
        self._append_prior_c()
        if self.c["t"].shape[1] > self.W:
            self._commit_head()

        n = self.c["t"].shape[1]
        V = torch.cat(self.buf_v, 0)[None]
        Q = torch.cat(self.buf_q, 0)[None]
        MV = torch.tensor(self.buf_mv, device=self.dev)[None]
        if self.n_itr > 0:
            self.m.w = self.w_er          # ER 専用のメタプライアに切り替える
            free = {k: self.c[k].clone().requires_grad_(True)
                    for k in PredVLA.LAYERS}
            opt = (torch.optim.SGD(list(free.values()), lr=self.lr)
                   if self.er_opt == "sgd" else
                   torch.optim.Adam(list(free.values()), lr=self.lr,
                                    eps=self.er_eps))
            for _ in range(self.n_itr):
                v_hat, q_hat, _, comp, _, _, _, _ = self._roll(free, n)
                nv = MV.sum().clamp(min=1.0)
                lv = (((V - v_hat).pow(2).sum(-1) / self.m.v_dim) * MV).sum() / nv
                lq = ((Q - q_hat).pow(2).sum(-1) / self.m.q_dim).mean()
                E = self.lam_v * lv + self.lam_q * lq + comp
                opt.zero_grad(set_to_none=True)
                E.backward()
                opt.step()
            for k in PredVLA.LAYERS:
                self.c[k] = free[k].detach()
            self.m.w = self.w_train       # 事前値の生成などは学習時の値に戻す
        with torch.no_grad():
            v_hat, q_hat, D, _, _, _, _, _ = self._roll(self.c, n)
            a = self.m.action_point(D[:, -1])
            if self.buf_mv[-1] > 0:
                self.last_eps_v = float(
                    (V[:, -1] - v_hat[:, -1]).pow(2).sum(-1).mean() / self.m.v_dim)
            self.last_eps_q = float(
                (Q[:, -1] - q_hat[:, -1]).pow(2).sum(-1).mean() / self.m.q_dim)
        self.t += 1
        return a[0].cpu().numpy()
