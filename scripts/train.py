"""predvla の学習。自由変数 c を系列ごとに持ち、重みと一緒に Adam で更新する。

LibPvrnn の学習と同じ扱い:
  A（ここでは c）は (系列, 時刻, r) の自由パラメータで optimizer に登録される
  h0 も学習パラメータ
  自由エネルギー = 再構成誤差 + Σ_l w_l Σ_t (事前値からの逸脱)

既存の pc_infer のような入れ子最適化が無いので torch.compile が素直に効く見込み。

使い方:
  python scripts/train.py --config configs/predvla_spatial.yaml --device cuda
"""
import argparse
import glob
import re
import os
import sys
import time

import numpy as np
import torch

# torch.compile(inductor)のメモリ計画は各グラフノードを再帰で辿る。
# 誤差の前向き投射(err_to_low)を入れると 1 step あたりのノードが増え、
# T=200 では既定の再帰上限 1000 を超えて RecursionError で落ちる(2026-08-08 実測)。
sys.setrecursionlimit(50000)
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # リポジトリ直下
sys.path.insert(0, HERE)

from predvla.model import PredVLA  # noqa: E402
from predvla.data import SeqDataset  # noqa: E402
from src.utils import compat       # noqa: E402  ★本リポジトリ固有

compat.apply()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto",
                    help="auto / cuda / mps / cpu。auto は cuda > mps > cpu")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", default=None,
                    help="再開元の ckpt。'auto' で run_dir の最新から")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:                        # 例: train.total_steps=1000
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        # YAML 1.1 は小数点のない指数表記を float と見なさない。
        #   yaml.safe_load("3e-5")   -> 文字列 '3e-5'
        #   yaml.safe_load("1.2e-4") -> float  0.00012
        # このため --set train.lr=3e-5 が文字列で入り、lr スイープの 3 本が
        # 「ValueError: Unknown format code 'g' for object of type 'str'」で即死した
        # （2026-08-02 01:16 発覚。config ファイル側は全て float なので無影響）。
        if isinstance(v, str):
            try:
                v = float(v) if any(c in v for c in ".eE") else int(v)
            except ValueError:
                pass
        cfg[sec][key] = v

    tr, dc = cfg["train"], cfg["data"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    dev = compat.pick_device(args.device)   # ★本リポジトリ固有（CPU 環境対応）

    _cr = cfg["paths"]["cache_root"]
    while _cr.startswith("../"):        # ★本リポジトリ固有（階層が 1 段浅い）
        _cr = _cr[3:]
    cache = dc and (_cr if os.path.isabs(_cr) else os.path.join(HERE, _cr))
    ds = SeqDataset(cache, dc["suite"], dc["T"], dc["vision_stride"],
                    tasks=dc["tasks"], max_demos=dc["max_demos"]).to(dev)
    N, T = len(ds), ds.T
    print(f"[predvla] dataset: {ds.summary()}")

    model = PredVLA(cfg).to(dev)
    # ★no_c（2026-08-13、ユーザー提案の RNN-VLA 対照）
    #   自由変数 c を持たず、自由エネルギーも使わない。毎 step の事前値 ĉ をそのまま
    #   使い（= use_prior、Complexity は恒等的に 0）、観測の予測誤差の逆伝播だけで
    #   重みを更新する。層・τ・ブリッジ（PB と efference copy）・ヘッド・GMM・
    #   データ・マスク・lr・step 数・パラメータ数は PredVLA と完全に同一なので、
    #   PredVLA NI0 との差が「c を持ち自由エネルギーで推論するか否か」だけになる。
    no_c = bool(tr.get("no_c", False))
    if no_c:
        # C は使わないので (1,1,r) のダミーだけ置く（ckpt の形式は変えない）
        C = torch.nn.ParameterDict({
            nm: torch.nn.Parameter(torch.zeros(1, 1, model.r_of[nm], device=dev))
            for nm in PredVLA.LAYERS})
        print(f"[predvla] ★RNN-VLA（no_c）: 自由変数 c 無し・自由エネルギー無し。"
              f"重み {model.num_params()/1e6:.3f}M のみを予測誤差で学習  device={dev}",
              flush=True)
    else:
        # 自由変数 c: 層ごとに (N, T, r)。PV-RNN の A に対応
        C = torch.nn.ParameterDict({
            nm: torch.nn.Parameter(torch.zeros(N, T, model.r_of[nm], device=dev))
            for nm in PredVLA.LAYERS})
        rsum = sum(model.r_of.values())
        print(f"[predvla] 重み {model.num_params()/1e6:.3f}M  "
              f"自由変数 c {N*T*rsum/1e6:.1f}M 個 ({N*T*rsum*4/2**20:.0f}MB)  "
              f"r 層別 {model.r_of}  device={dev}")

    # ★自由変数 C への weight decay（2026-08-27 追加、ユーザー指示）。既定 0.0 = 従来と完全に同一。
    #   なぜ足したか: C は (系列, 時刻) ごとの独立パラメータで容量が事実上無制限、
    #   しかも減衰がかかっていなかった。実測（e300k の [prior] ログ）では、
    #   posterior の mse_a が 0.0000 まで落ちる（= 学習デモを完全に暗記）一方で
    #   prior は 31k→182k で改善ゼロ（v 0.62→0.71 と悪化）、prior/posterior 比が
    #   1.94→4.30 と単調に開き続けた。テスト時に効くのは prior なので、
    #   「暗記の担い手だけが減衰の対象外」という非対称が汎化を止めている疑いがある。
    opt = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": tr["lr"],
          "weight_decay": tr["weight_decay"]},
         {"params": list(C.parameters()), "lr": tr["lr_c"],
          "weight_decay": float(tr.get("weight_decay_c", 0.0))}])
    if float(tr.get("weight_decay_c", 0.0)) > 0:
        print(f"[predvla] ★自由変数 C に weight decay {tr['weight_decay_c']:g}")
    sched = None
    if str(tr.get("lr_schedule", "none")).lower() == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=tr["total_steps"])
        print("[predvla] lr cosine")
    else:
        print(f"[predvla] lr 一定（重み {tr['lr']:g} / c {tr['lr_c']:g}）")

    fe = model.free_energy
    if tr.get("compile", False) and dev == "cuda":
        torch._dynamo.config.cache_size_limit = 64
        # 既定は reduce-overhead（cudagraph）。200 step の小さな行列積の連続なので
        # カーネル起動が支配的で、default mode では eager と同速だった（実測）。
        mode = os.environ.get("PredVLA_COMPILE_MODE", "reduce-overhead")
        # ★er_train のときは default に落とす（cudagraph のバッファ再利用が
        #   ER の内側 autograd と衝突しうる。2026-08-12）
        if tr.get("er_train", False) and mode == "reduce-overhead":
            mode = "default"
        fe = torch.compile(model.free_energy, mode=None if mode == "default" else mode)
        print(f"[predvla] torch.compile 有効（{mode}）")

    run_dir = os.path.join(HERE, "results", tr["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(tr["seed"])
    t0 = time.time()
    ema = None
    start_step = 0
    # --- 再開（2026-08-01 追加）---
    # --resume auto なら run_dir の最大 step の ckpt から、パス指定ならそれから再開する。
    # 旧 ckpt（opt を持たない）でも model と C だけ読んで続けられる（その旨を警告する）。
    if args.resume:
        rp = args.resume
        if rp == "auto":
            cks = sorted(glob.glob(os.path.join(run_dir, "step_*.pt")),
                         key=lambda x: int(re.search(r"step_(\d+)", x).group(1)))
            rp = cks[-1] if cks else None
            # 軽量ローリング ckpt(last.pt、2026-08-06 追加)があり、それが新しければ優先。
            # OOM 死からの再開粒度を 1000 step にするためのもの。
            lp = os.path.join(run_dir, "last.pt")
            if os.path.exists(lp):
                try:
                    ls = int(torch.load(lp, map_location="cpu",
                                        weights_only=False)["step"])
                    best = (int(re.search(r"step_(\d+)", rp).group(1))
                            if rp else -1)
                    if ls > best:
                        rp = lp
                except Exception as e:
                    print(f"[predvla] last.pt 読めず({e})、step_*.pt から再開")
        if rp and os.path.exists(rp):
            ck = torch.load(rp, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            with torch.no_grad():
                for k in PredVLA.LAYERS:
                    C[k].copy_(ck["C"][k].to(dev))
            start_step = int(ck["step"])
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
                rng.bit_generator.state = ck["rng"]
                torch.set_rng_state(ck["torch_rng"].cpu()
                                    if hasattr(ck["torch_rng"], "cpu") else ck["torch_rng"])
                if "cuda_rng" in ck and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([t.cpu() for t in ck["cuda_rng"]])
                if sched is not None and "sched" in ck:
                    sched.load_state_dict(ck["sched"])
                ema = ck.get("ema")
                print(f"[predvla] 完全再開 {rp} step {start_step}", flush=True)
            else:
                print(f"[predvla] 警告: {rp} に opt が無い。model/C のみ復元して "
                      f"step {start_step} から続行（最適化器の履歴は失われる）", flush=True)
        else:
            print(f"[predvla] 再開対象が無いので最初から学習する", flush=True)
    c_no_action = bool(tr.get("c_no_action", False))
    if c_no_action:
        print("[predvla] ★C への勾配から行動 NLL を外す（λv・λq・comp のみ）", flush=True)
    er_train_on = bool(tr.get("er_train", False))
    if er_train_on:
        from predvla.er_train import er_window_loss
        _ew = tr.get("er_w")
        print(f"[predvla] ★ER ループを通した学習 有効: 窓 {tr.get('er_window',40)} "
              f"n_itr {tr.get('er_itr',10)} er_lr {tr.get('er_lr',0.05)} "
              f"損失重み {tr.get('er_loss_w',1.0)} "
              f"内側 Complexity 重み {'学習時の model.w' if _ew is None else _ew}",
              flush=True)
        # ★主損失だけ compile を残せる（ER の内側は free_energy を通らず
        #   one_step を直接呼ぶため）。compile を切ると 13 倍遅くなるので残す価値が高い。
        #   ただし reduce-overhead(CUDA graphs)はバッファ再利用で autograd と
        #   衝突した前例があるので、er_train 時は default モードに落とす。

    for step in range(start_step + 1, tr["total_steps"] + 1):
        idx = torch.from_numpy(rng.choice(N, size=tr["batch_size"],
                                          replace=False)).to(dev)
        v, q, a, l, mask, mask_v = ds.batch(idx)
        Cb = ({k: C[k] for k in PredVLA.LAYERS} if no_c
              else {k: C[k][idx] for k in PredVLA.LAYERS})
        # lambda_v / lambda_q は既定 1.0（現行と同一）。lambda_v=0 が視覚予測の ablation。
        # no_c なら use_prior=True（C は参照されず comp は恒等的に 0）
        out = fe(Cb, l, v, q, a, mask, mask_v, tr["lambda_a"], no_c,
                 float(tr.get("lambda_v", 1.0)), float(tr.get("lambda_q", 1.0)))
        opt.zero_grad(set_to_none=True)
        # ★視覚のみで c を駆動する学習(2026-08-10、ユーザー提案)
        #   c_vision_only_p の確率で、**c への勾配を「視覚誤差 + Complexity」だけ**にする。
        #   行動 NLL と固有感覚誤差は **重みへの勾配としては残す**(予測目標としては残す)。
        #   → 「視覚誤差を最小にする c」が「正しい行動を出す c」と一致するように
        #     重みが作り替わる、という圧力をかける。
        #   損失から λq を落とす案(私の (G))と違い、q̂ の精度は保たれる。
        # c_vision_only_p は数値(固定確率)またはリスト [開始, 終了]。
        # 後者は学習を通じて線形に焼きなます = (G)「推論目的関数から λq を段々に外す」。
        _pv = tr.get("c_vision_only_p", 0.0)
        if isinstance(_pv, (list, tuple)):
            _f = min(1.0, step / max(1, tr["total_steps"]))
            p_vo = float(_pv[0]) + (float(_pv[1]) - float(_pv[0])) * _f
        else:
            p_vo = float(_pv)
        if p_vo > 0.0 and rng.random() < p_vo:
            # C を detach した 2 回目の前向きから行動・固有感覚の項を取る。
            #   ・C への勾配 … 1 回目の λv·lv + comp だけ
            #   ・重みへの勾配 … 両方から(= full total と同じ)
            # autograd.grad を 2 回呼ぶ実装は torch.compile(reduce-overhead)の
            # CUDA graphs がバッファを再利用するため
            #   "variable ... modified by an inplace operation" で落ちる(2026-08-10 実測)。
            # 前向き 2 回 + backward 1 回にすれば同じ勾配が安全に取れる。
            Cd = {k: Cb[k].detach() for k in PredVLA.LAYERS}
            out2 = fe(Cd, l, v, q, a, mask, mask_v, tr["lambda_a"], False,
                      float(tr.get("lambda_v", 1.0)), float(tr.get("lambda_q", 1.0)))
            loss = (float(tr.get("lambda_v", 1.0)) * out["loss_v"] + out["comp"]
                    + float(tr.get("lambda_q", 1.0)) * out2["loss_q"]
                    + tr["lambda_a"] * out2["loss_a"])
            loss.backward()
        elif c_no_action:
            # ★C から「行動 NLL の勾配だけ」を外す（2026-08-12、ユーザー提案）。
            #   狙い: 学習時の c を「観測（v と q）を説明する c」にして、
            #   テスト時の ER の目的関数（観測項 + comp、行動項なし）と揃える。
            #   ★② c_vision_only_p=1.0 との違いは λq を残すこと。
            #     ② は λa と λq の両方を外して NI0 が 72 → 22 に壊滅した。
            #     どちらが原因だったかは分離できていないので、この条件がその切り分けになる。
            #   ★予想: q は行動の積分なので、λq を残す限り c は行動軌道を説明し続ける。
            #     つまり効果は小さいかもしれない。それ自体が「λq が本体」の証拠になる。
            Cd = {k: Cb[k].detach() for k in PredVLA.LAYERS}
            out2 = fe(Cd, l, v, q, a, mask, mask_v, tr["lambda_a"], False,
                      float(tr.get("lambda_v", 1.0)), float(tr.get("lambda_q", 1.0)))
            loss = (float(tr.get("lambda_v", 1.0)) * out["loss_v"]
                    + float(tr.get("lambda_q", 1.0)) * out["loss_q"] + out["comp"]
                    + tr["lambda_a"] * out2["loss_a"])
            loss.backward()
        else:
            out["total"].backward()
        # ★ER ループを通した学習（2026-08-12、次の論文 B'1）。
        #   窓の中で δ=0(=prior) から観測誤差の勾配で n_itr 回 SGD 更新し、
        #   その δ で出る行動の NLL を重みまで逆伝播する。
        #   これで「誤差 → Δc → 行動」という写像そのものが訓練される。
        #   既定 er_train=false なら従来と完全に同一。
        er_info = None
        if er_train_on:
            l_er, er_info = er_window_loss(
                model, C, idx, l, v, q, a, mask, mask_v, rng,
                W=int(tr.get("er_window", 40)),
                n_itr=int(tr.get("er_itr", 10)),
                er_lr=float(tr.get("er_lr", 0.05)),
                lam_v=float(tr.get("lambda_v", 1.0)),
                lam_q=float(tr.get("lambda_q", 1.0)),
                burn_max=tr.get("er_burn_max"),
                first_order=bool(tr.get("er_first_order", True)),
                sub_b=int(tr.get("er_batch", 0)),
                # ★内側 ER の Complexity 重み（2026-08-13）。None（既定）で学習時の
                #   model.w。スカラを入れるとテスト時 ER の --er-w と同じ意味になる。
                er_w=tr.get("er_w"),
                diag=(step % tr["log_every"] == 0 or step == 1))
            (float(tr.get("er_loss_w", 1.0)) * l_er).backward()
            er_info["loss"] = float(l_er.detach())
        gw = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip_w"])
        gc = torch.nn.utils.clip_grad_norm_(C.parameters(), tr["clip_c"])
        opt.step()
        if sched is not None:
            sched.step()
        tot = float(out["total"].detach())
        ema = tot if ema is None else 0.99 * ema + 0.01 * tot
        if step % tr["log_every"] == 0 or step == 1:
            print(f"step {step:6d}/{tr['total_steps']} | total {tot:8.4f} "
                  f"(ema {ema:8.4f}) | v {float(out['loss_v']):.4f} "
                  f"q {float(out['loss_q']):.4f} a {float(out['loss_a']):8.4f} "
                  f"comp {float(out['comp']):.4f}"
                  f"(t {float(out['comp_t']):.3f} v {float(out['comp_v']):.3f} "
                  f"u {float(out['comp_up']):.3f} l {float(out['comp_low']):.3f}) "
                  f"mse_a {float(out['mse_a']):.4f} | "
                  f"g_w {float(gw):.1f} g_c {float(gc):.1f} "
                  f"| {(time.time()-t0)/step*1000:.0f} ms/step"
                  + ("" if er_info is None else
                     f" | ★ER a {er_info['loss']:.4f}"
                     + (f" (δ=0 {er_info['a0']:.4f}, ★利得 {er_info['gain_a']:+.4f}"
                        f", E 低下 {er_info['gain_E']:+.4f})"
                        if "gain_a" in er_info else "")
                     + f" |δ| {er_info['delta_norm']:.4f} off {er_info['off']}"),
                  flush=True)
        if step % tr.get("prior_every", 1000) == 0:
            # prior 生成（c=ĉ、推論なし）の誤差。posterior との差が w の調整指標。
            with torch.no_grad():
                pr = model.free_energy(Cb, l, v, q, a, mask, mask_v,
                                       tr["lambda_a"], use_prior=True)
            print(f"    [prior] v {float(pr['loss_v']):.4f} q {float(pr['loss_q']):.4f} "
                  f"mse_a {float(pr['mse_a']):.4f}   "
                  f"[posterior] v {float(out['loss_v']):.4f} "
                  f"q {float(out['loss_q']):.4f} mse_a {float(out['mse_a']):.4f}",
                  flush=True)
        if step % tr["ckpt_every"] == 0 or step == tr["total_steps"]:
            p = os.path.join(run_dir, f"step_{step}.pt")
            # 再開に必要な情報をすべて入れる（2026-08-01、ユーザー指示）。
            #   opt   Adam のモーメント。C の分が支配的（4 層 × 500×200×64 × 2 = 205MB）
            #   rng   バッチ抽出の numpy Generator
            #   torch_rng / cuda_rng  ノイズや初期化の再現用
            #   ema   ログ用の指数移動平均
            # 既存の読み手（eval / probe）は model と cfg しか見ないのでキー追加は無害。
            ck = {"model": model.state_dict(),
                  "C": {k: v_.detach().cpu() for k, v_ in C.items()},
                  "cfg": cfg, "step": step,
                  "opt": opt.state_dict(),
                  "rng": rng.bit_generator.state,
                  "torch_rng": torch.get_rng_state(),
                  "ema": ema}
            if torch.cuda.is_available():
                ck["cuda_rng"] = torch.cuda.get_rng_state_all()
            if sched is not None:
                ck["sched"] = sched.state_dict()
            torch.save(ck, p)
            print(f"[predvla] saved {p} ({os.path.getsize(p)/1e6:.0f}MB)", flush=True)
    print(f"[predvla] done in {time.time()-t0:.1f}s  final ema {ema:.4f}")


if __name__ == "__main__":
    main()
