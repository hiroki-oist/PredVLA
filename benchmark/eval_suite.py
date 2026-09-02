"""Closed-loop eval across a whole suite (§6.1) — M2 gate.

Runs N rollouts per task for every task in the suite and reports per-task and
mean success rate.

Example:
  python scripts/eval_suite.py --ckpt results/spatial_s0/step_100000.pt --n 20
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from src.utils import compat
from src.eval.rollout import Rollout

compat.apply()


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PROJ, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default=None, help="default: cfg.data.suite")
    ap.add_argument("--n", type=int, default=20, help="rollouts per task")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="", help="suffix for the output csv name "
                    "(e.g. step40000) so evals of different ckpts don't clobber")
    # test-time latent updates (predictive-coding inference; see Rollout)
    ap.add_argument("--lu-k", type=int, default=0, help="Phase A: h_v refinement steps")
    ap.add_argument("--lu-eta", type=float, default=0.05)
    ap.add_argument("--lu-beta", type=float, default=0.1)
    ap.add_argument("--lu-pb-eta", type=float, default=0.0, help="Phase B: PB delta lr (0=off)")
    ap.add_argument("--lu-pb-clip", type=float, default=1.0)
    ap.add_argument("--pc-eta", type=float, default=-1.0,
                    help="rollout 側の内側推論のステップ幅を上書き(-1=学習時設定)。"
                         "既定 0.1 は FE を 0.17%% しか下げないので実質無効。実効域は 1-10")
    ap.add_argument("--pc-rollout-k", type=int, default=-1,
                    help="rollout 時の窓 PC 推論の反復数: -1=学習時のまま, 0=切る, K>0=強制")
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="rollout 時の行動振幅スケール(運動6自由度のみ, [-1,1]にクリップ)")
    ap.add_argument("--scale-gripper", action="store_true",
                    help="グリッパ次元もスケールする")
    ap.add_argument("--perturb", default="none", choices=["none", "intervene"],
                    help="ロバストネス評価: intervene = 途中でランダム行動を注入")
    ap.add_argument("--perturb-at", type=int, default=60)
    ap.add_argument("--perturb-len", type=int, default=10)
    ap.add_argument("--obs-noise", type=float, default=0.0, help="視覚特徴 v に乗せるガウスノイズ")
    ap.add_argument("--act-delay", type=int, default=0, help="行動を d step 遅らせる")
    ap.add_argument("--img-perturb", default="none",
                    choices=["none", "noise", "blur", "bright", "occlude", "shift"],
                    help="画像空間の摂動(実機のカメラ劣化に対応)")
    ap.add_argument("--img-strength", type=float, default=0.0,
                    help="摂動強度: noise=σ(0-255) / blur=カーネル / bright=相対 / occlude=面積比 / shift=画素")
    ap.add_argument("--fe-terms", default="both", choices=["both", "v", "q"],
                    help="rollout 時の内部推論が最小化する Accuracy の構成。"
                         "both=視覚+固有感覚(既定) / v=視覚のみ / q=固有感覚のみ。"
                         "学習不要の ablation")
    ap.add_argument("--pc-delta-warm", default=None,
                    choices=["0", "1"],
                    help="窓推論の delta を前 tick の値から始めるか。未指定なら ckpt の "
                         "model.pc_delta_warm に従う(pcv2 系は true)")
    ap.add_argument("--pc-vars", default="", choices=["", "both", "hv", "delta"],
                    help="推論で動かす自由変数の上書き。''=ckpt のまま / hv=内部状態のみ "
                         "(PB への自由オフセット delta を殺す) / delta=PB 補正のみ")
    ap.add_argument("--tasks", type=int, nargs="+", default=None,
                    help="評価するタスク id を限定する(既定: 全タスク)")
    # --- 開ループ評価(2026-08-01)。ベースライン(bcrnn/bctf)の generic rollout で有効 ---
    # PredVLA の n_itr=0（観測がモデルに一切入らない開ループ）に対応する条件を
    # ベースライン側にも作るため。重みは同じで評価時だけの ablation。
    ap.add_argument("--freeze-vision", action="store_true",
                    help="v を最初の 1 回だけ観測し、以降更新しない")
    ap.add_argument("--chunk-ensemble", action="store_true",
                    help="ACT 式の temporal ensembling を使う（chunk>1 のときのみ有効）")
    ap.add_argument("--chunk-ensemble-m", type=float, default=0.01,
                    help="指数重みの減衰。ACT 既定 0.01")
    ap.add_argument("--proprio-pred", action="store_true",
                    help="q を観測ではなく自分の予測 q̂ に置き換えて自走する")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ro = Rollout(args.ckpt, device=args.device,
                 lu_k=args.lu_k, lu_eta=args.lu_eta, lu_beta=args.lu_beta,
                 lu_pb_eta=args.lu_pb_eta, lu_pb_clip=args.lu_pb_clip,
                 pc_rollout_k=args.pc_rollout_k, pc_eta=args.pc_eta,
                 action_scale=args.action_scale, scale_gripper=args.scale_gripper,
                 perturb=args.perturb, perturb_at=args.perturb_at,
                 perturb_len=args.perturb_len, obs_noise=args.obs_noise,
                 act_delay=args.act_delay, perturb_seed=0,
                 img_perturb=args.img_perturb, img_strength=args.img_strength,
                 fe_terms=args.fe_terms, pc_vars=args.pc_vars,
                 pc_delta_warm=(None if args.pc_delta_warm is None
                                else args.pc_delta_warm == "1"))
    # 開ループ ablation。Rollout の __init__ を変えずに属性で渡す(既定 False で従来通り)
    ro.ol_freeze_vision = bool(args.freeze_vision)
    ro.ol_proprio_pred = bool(args.proprio_pred)

    ro.chunk_ensemble = bool(args.chunk_ensemble)

    ro.chunk_ensemble_m = float(args.chunk_ensemble_m)
    if args.freeze_vision or args.proprio_pred:
        print(f"[eval-suite] 開ループ: 視覚凍結={args.freeze_vision} "
              f"固有感覚を自己予測={args.proprio_pred}")
    # multi-suite training configs ("a+b") default to evaluating the first suite
    suite_name = args.suite or ro.cfg["data"]["suite"].split("+")[0]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    print(f"[eval-suite] suite={suite_name} tasks={suite.n_tasks} n={args.n} "
          f"max_steps={args.max_steps}  ckpt={args.ckpt}")

    rows = []
    per_task = []
    t0 = time.time()
    tids = args.tasks if args.tasks is not None else list(range(suite.n_tasks))
    for tid in tids:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        inits = suite.get_task_init_states(tid)
        l_vec = ro.language_latent(task.language)
        succ = 0
        n = min(args.n, len(inits))
        for r in range(n):
            res = ro.run_episode(env, l_vec, inits[r], warmup=args.warmup, max_steps=args.max_steps)
            succ += int(res["success"])
            rows.append({"task_id": tid, "rollout": r, **res})
        env.close()
        rate = succ / max(n, 1)
        per_task.append(rate)
        # ★scripts/aggregate.py が読む形（"task N success S/T"）で必ず 1 行出す。
        #   ベースラインと PredVLA 本線を同じ集計器にかけるための約束。
        print(f"  task {tid:2d} success {succ:2d}/{n:<2d} = {rate*100:5.1f}%  "
              f"[{task.name[:44]}]  (elapsed {time.time()-t0:.0f}s)")

    mean = float(np.mean(per_task))
    n_tasks_done = len(per_task)
    run_name = ro.cfg["train"]["run_name"]
    out_dir = _abs(os.path.join("results", run_name))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(out_dir, f"eval_suite_{suite_name}{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "rollout", "success", "steps",
                                          "reward_max", "eef_path", "act_absmean"])
        w.writeheader()
        w.writerows(rows)

    # ★"全体" の行も aggregate.py の形に合わせる。エピソード単位の成功数で出す
    #   （タスク平均ではない。タスクごとの n が揃っていれば同じ値になる）。
    n_succ = sum(int(r["success"]) for r in rows)
    n_try = len(rows)
    print(f"\n  全体 {100.0*n_succ/max(n_try,1):.2f}%  ({n_succ}/{n_try})")
    print(f"\n[eval-suite] MEAN success rate over {n_tasks_done} tasks: {mean*100:.1f}%")
    print("[eval-suite] per-task: " + " ".join(f"{r*100:.0f}" for r in per_task))
    print(f"[eval-suite] total {time.time()-t0:.0f}s -> {csv_path}")


if __name__ == "__main__":
    main()
