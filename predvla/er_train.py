"""ER ループを通した学習（2026-08-12）。次の論文の核 B'1 の実装。

## 何を直そうとしているか

現行 PredVLA では、学習時の自由変数 c は系列ごとに **オフラインで** 最適化される
（posterior）。テスト時の ER は c を **観測誤差の勾配で** 動かす。
**この 2 つは別の写像で、後者は学習信号を一切受け取っていない。**

2026-08-12 の測定（`notes/PROJECT_COMPILATION_2026-08-11.md` §8）で
  ・視覚は内部状態に届いている（r 0.12、帰無 0.01、A2 で消える）
  ・しかし分散の 1.4% で、行動を変えるには 2 桁足りない
  ・部分空間の手術（P0a / P1' / P4）は 3 つとも実測で否定
が分かったので、残る道は「どの誤差方向に反応すべきかをモデル自身に学ばせる」こと。

## やり方

学習の各 step で、系列の途中から窓 W step を切り出し、

  ① 窓の手前まで posterior の C で進める（no_grad。状態を作るだけ）
  ② 窓の中で δ=0（= prior）から始め、観測誤差の勾配で δ を n_itr 回 SGD 更新する
     ★この内側ループを微分可能にする（create_graph=True）
  ③ 得られた δ で同じ窓をもう一度回し、**行動 NLL** を取る
  ④ ③ の損失を重みまで逆伝播する

②の目的関数はテスト時の ER と同じ（観測項 + Complexity。行動項は入れない。
テスト時に正解行動が無いため）。③の損失だけが行動を見る。
つまり「観測誤差を下げる向きに δ を動かしたら、正しい行動が出る」ようにモデルを作り替える。

## コスト

内側 n_itr 回それぞれで W step のロールアウトと backward（create_graph 付き）が要る。
Adam n_itr=20 なら素朴には 20 倍以上。**SGD n_itr=10 なら 10 倍程度**（ユーザー試算）。
spatial なら Fujiwara で 15 時間程度。

## 設計上の判断

  ・δ（事前値からのずれ）で持つ。δ=0 が prior と一致するので初期化が自然
    （`model.one_step(..., cs_is_delta=True)`）
  ・内側は SGD 固定 lr。Adam だとモーメント状態の微分まで要る
  ・既存の posterior 損失は **残す**。ER 損失は加算項にする
    （posterior を消すと PV-RNN の生成モデルとしての質が落ちるため）
"""
import torch

from .model import PredVLA

LAY = PredVLA.LAYERS


def _roll_window(model, hs, ds, pa, pq, lang_t, delta, W, v_obs=None, mv=None):
    """窓 W step を δ で回す。戻り値: v_hat, q_hat, d_low の列と comp（B,）。"""
    V, Q, D = [], [], []
    comp = lang_t.new_zeros(lang_t.shape[0])
    for t in range(W):
        cs = {k: delta[k][:, t] for k in LAY}
        hs, ds, v_hat, q_hat, dl, cp = model.one_step(
            hs, ds, cs, lang_t, pa, pq, comp_reduce="none",
            v_obs=(None if v_obs is None else v_obs[:, t]),
            mv_t=(None if mv is None else mv[:, t]),
            cs_is_delta=True)
        V.append(v_hat); Q.append(q_hat); D.append(dl)
        pa = model.action_point(dl)
        pq = q_hat
        comp = comp + cp
    return torch.stack(V, 1), torch.stack(Q, 1), torch.stack(D, 1), comp, hs, ds


def _obs_energy(model, v_hat, q_hat, comp, v, q, m, mv, lam_v, lam_q):
    """テスト時の ER と同じ目的関数（観測項 + Complexity）。行動項は入れない。"""
    nv = (m * mv).sum(1).clamp(min=1.0)
    nq = m.sum(1).clamp(min=1.0)
    lv = (((v - v_hat).pow(2).sum(-1) / model.v_dim) * m * mv).sum(1) / nv
    lq = (((q - q_hat).pow(2).sum(-1) / model.q_dim) * m).sum(1) / nq
    return (lam_v * lv + lam_q * lq + comp).sum()


def er_window_loss(model, C, idx, l, v, q, a, mask, mask_v, rng,
                   W=40, n_itr=10, er_lr=0.05, lam_v=1.0, lam_q=1.0,
                   burn_max=None, first_order=False, sub_b=0, diag=False,
                   er_w=None):
    """ER を n_itr step 展開し、その後の行動 NLL を返す（重みまで微分可能）。

    C は学習中の自由変数（posterior）。窓の手前までを進めるためだけに使い、
    ★no_grad で回すので C への勾配は出ない（posterior 損失側で別に付く）。

    first_order: 内側の勾配を create_graph 無しで取る（1 次近似）。
      外側の逆伝播が「最後の 1 回のロールアウト」だけになるので大幅に安い。
      失うのは「内側の更新則そのものを通る勾配」。得られる δ は同じなので
      「ER が出した δ で正しい行動が出るように重みを作る」圧力は残る。
      ★実測でここが支配的に効く（内側を全部つなぐと外側の逆伝播が 10×40 段になる）。
    sub_b: >0 ならバッチの先頭 sub_b 本だけ ER に使う（コスト削減）。
    diag: True のとき ★δ=0（prior）での行動 NLL と観測エネルギーも返す。
      外側の損失（δ_ER での NLL）は定義上下がるので、それだけでは
      「ER が効いている」ことの証拠にならない。見るべきは差分:
        er_gain_a = NLL(δ=0) − NLL(δ_ER)    ★正で大きいほど ER が行動を良くしている
        er_gain_E = E(δ=0) − E(δ_ER)        内側ループが観測誤差を下げられているか
      ★失敗の署名は er_gain_a → 0 かつ |δ| → 0（δ を無視する方向に収束）。
    er_w: 内側 ER の Complexity 重み（2026-08-13 追加）。None で学習時の
      model.w（{t 0.05, v 0.02, up 0.02, low 0.01}）をそのまま使う（従来と同一）。
      ★スカラを渡すと全層をその値にする。テスト時 ER が既定で使っている
      er_w=1.0 と学習時をそろえるための対照に使う（eval_batch の --er-w と対応）。
    """
    _w_save = None
    if er_w is not None:
        _w_save = dict(model.w)
        model.w = {k: float(er_w) for k in model.w}
    try:
        if sub_b and sub_b < v.shape[0]:
            sl_b = slice(0, sub_b)
            l, v, q, a = l[sl_b], v[sl_b], q[sl_b], a[sl_b]
            mask, mask_v, idx = mask[sl_b], mask_v[sl_b], idx[sl_b]
        B, T = v.shape[0], v.shape[1]
        dev = v.device
        lang_t = model.T.W_l(l)

        # --- ① 窓の手前まで posterior で進める（状態を作るだけ） ---
        valid = mask.sum(1).long()                       # (B,) 各系列の有効長
        hi = int((valid.min() - W).clamp(min=0).item()) if burn_max is None \
            else min(burn_max, int((valid.min() - W).clamp(min=0).item()))
        off = int(rng.integers(0, hi + 1)) if hi > 0 else 0
        hs, ds = model.init_state(B)
        pa = v.new_zeros(B, model.a_dim)
        pq = v.new_zeros(B, model.q_dim)
        with torch.no_grad():
            for t in range(off):
                cs = {k: C[k][idx][:, t] for k in LAY}
                hs, ds, _, q_hat, dl, _ = model.one_step(
                    hs, ds, cs, lang_t, pa, pq, comp_reduce="none")
                pa = model.action_point(dl)
                pq = q_hat
        hs = [x.detach() for x in hs]
        ds = [x.detach() for x in ds]
        pa, pq = pa.detach(), pq.detach()

        sl = slice(off, off + W)
        vw, qw, aw = v[:, sl], q[:, sl], a[:, sl]
        mw, mvw = mask[:, sl], mask_v[:, sl]

        # --- ② 内側 ER（微分可能）。δ=0 = prior から始める ---
        delta = {k: torch.zeros(B, W, model.r, device=dev, requires_grad=True)
                 for k in LAY}
        for _ in range(n_itr):
            v_hat, q_hat, _, comp, _, _ = _roll_window(
                model, hs, ds, pa, pq, lang_t, delta, W)
            E = _obs_energy(model, v_hat, q_hat, comp, vw, qw, mw, mvw, lam_v, lam_q)
            g = torch.autograd.grad(E, list(delta.values()),
                                    create_graph=not first_order)
            if first_order:
                delta = {k: (delta[k] - er_lr * gi).detach().requires_grad_(True)
                         for k, gi in zip(LAY, g)}
            else:
                delta = {k: delta[k] - er_lr * gi for k, gi in zip(LAY, g)}

        # --- ③ 得られた δ で行動 NLL を取る ---
        _, _, D, _, _, _ = _roll_window(model, hs, ds, pa, pq, lang_t, delta, W)
        nll = model.action_nll(D.reshape(B * W, -1),
                               aw.reshape(B * W, -1)).reshape(B, W)
        loss_a = (nll * mw).sum() / mw.sum().clamp(min=1.0)

        info = {"off": off}
        with torch.no_grad():
            info["delta_norm"] = (sum(float((delta[k] ** 2).sum()) for k in LAY) /
                                  max(B * W, 1)) ** 0.5
            if diag:
                # ★δ=0（prior）で同じ窓を回して比較する。これが無いと
                #   「ER が効いている」ことの証拠にならない。
                z = {k: torch.zeros_like(delta[k]) for k in LAY}
                v0, q0, D0, comp0, _, _ = _roll_window(
                    model, hs, ds, pa, pq, lang_t, z, W)
                nll0 = model.action_nll(D0.reshape(B * W, -1),
                                        aw.reshape(B * W, -1)).reshape(B, W)
                a0 = float((nll0 * mw).sum() / mw.sum().clamp(min=1.0))
                E0 = float(_obs_energy(model, v0, q0, comp0, vw, qw, mw, mvw,
                                       lam_v, lam_q)) / max(B, 1)
                vE, qE, _, compE, _, _ = _roll_window(
                    model, hs, ds, pa, pq, lang_t, delta, W)
                E1 = float(_obs_energy(model, vE, qE, compE, vw, qw, mw, mvw,
                                       lam_v, lam_q)) / max(B, 1)
                info["a0"] = a0
                info["gain_a"] = a0 - float(loss_a.detach())   # ★正なら ER が行動を良くした
                info["gain_E"] = E0 - E1                       # 内側が誤差を下げた量
        return loss_a, info
    finally:
        if _w_save is not None:
            model.w = _w_save
