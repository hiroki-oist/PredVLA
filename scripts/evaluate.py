#!/usr/bin/env python
"""評価をコマンド 1 本で回す。プロトコルの取り違えを防ぐのが主目的。

論文の数値は「どのプロトコルで測ったか」で ±10pt 動く。素の eval_batch.py には
20 個以上のつまみがあり、キューを手で書くと必ず取り違える。ここでは
predvla/protocol.py の定数だけから引数を組み立てるので、間違えようがない。

    # 1 つの ckpt を表①′ プロトコルで
    python scripts/evaluate.py --suite libero_spatial --ckpt results/predvla_spatial_s1/step_30000.pt

    # ディレクトリ内の 7 シードを並列 6 本で
    python scripts/evaluate.py --suite libero_spatial \
        --ckpt-dir checkpoints/main --seeds 0-6 --jobs 6

    # Long（★1 プロセス 1 タスクに自動で割る）
    python scripts/evaluate.py --suite libero_10 --ckpt-dir checkpoints/main --seeds 1-7

    # 何が走るか見るだけ
    python scripts/evaluate.py --suite libero_goal --ckpt-dir checkpoints/main --seeds 0-6 --dry-run

スイートごとに自動で変わるもの（predvla/protocol.py が持つ）
  3 スイート  n=50/タスク・B=16・max_steps 600・1 プロセスで 10 タスク
  libero_10   n=5/タスク・B=1・タスクごとに「最長デモ×1.2」で打ち切り・
              ★1 プロセス 1 タスク（複数の OffScreenRenderEnv を 1 プロセスに
              持つと描画コンテキストを共有し、別シーンのカメラ姿勢で描かれる）
  er_w        3 スイートは 1.0、Long は 0.8（掃引の最良値）

出力は results/logs/ev_<tag>.log。完了したセル（ログに「全体」がある）は飛ばすので、
途中で止めて再実行してよい。集計は scripts/aggregate.py。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import protocol as P  # noqa: E402

LOG_DIR = os.path.join(ROOT, "results", "logs")
# ★既定は main = 確定プロトコル（Adam n_itr10 / er_lr 0.05 / er_w 1.0 / 窓 40）。
#   sgd10 は旧看板で、無傷のモデルでは主表と区別できないが★アブレーションと
#   組み合わせると壊れる例がある（A3 が SGD10 でだけ 1.00 に崩壊）。
PROTOCOLS = {"main": P.MAIN, "sgd10": P.SGD10, "ni0": P.NI0}


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def vpy() -> str:
    p = os.path.join(ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def done(tag: str) -> bool:
    """そのセルが完了しているか（ログに「全体」の行があるか）。"""
    p = os.path.join(LOG_DIR, f"ev_{tag}.log")
    if not os.path.exists(p):
        return False
    try:
        with open(p, errors="ignore") as f:
            return "全体" in f.read()
    except OSError:
        return False


def free_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 1e9


def build_cells(args) -> list[tuple[str, list[str]]]:
    """(tag, コマンド) の一覧を作る。"""
    er = PROTOCOLS[args.protocol]
    is_long = args.suite == P.SUITE_LONG
    # ★確定プロトコルは全スイート er_w=1.0（Long も 0.8 ではない）。
    er_w = args.er_w if args.er_w is not None else P.ER_W

    # --- ckpt を集める ---
    ckpts: list[tuple[str, str]] = []      # (名前, パス)
    if args.ckpt:
        for p in args.ckpt:
            for q in sorted(glob.glob(p)) or [p]:
                # 名前の付け方は 2 通りある。取り違えるとログ名が衝突して
                # シードどうしが上書きされる（実際に踏んだ）。
                #   .../<run_name>/step_30000.pt  -> run_name を使う
                #   .../predvla_spatial_s1.pt        -> ファイル名の幹を使う
                stem = os.path.splitext(os.path.basename(q))[0]
                if re.fullmatch(r"step_\d+", stem) or stem in ("last", "final"):
                    name = os.path.basename(os.path.dirname(q)) or stem
                else:
                    name = stem
                ckpts.append((name, q))
    if args.ckpt_dir:
        seeds = parse_seeds(args.seeds) if args.seeds else list(
            P.SEEDS_LONG if is_long else P.SEEDS_MAIN)
        for s in seeds:
            # 対応する 2 つの置き方をどちらも探す:
            #   <ckpt_dir>/<name>_<suite>_s<S>.pt        （配布物の平置き）
            #   <ckpt_dir>/<name>_s<S>/step_30000.pt     （学習の出力ディレクトリ）
            pats = [
                os.path.join(args.ckpt_dir, f"{args.name}_{P.SHORT[args.suite]}_s{s}.pt"),
                os.path.join(args.ckpt_dir, f"{args.name}_{P.SHORT[args.suite]}_s{s}",
                             f"step_{P.TRAIN_STEPS}.pt"),
                os.path.join(args.ckpt_dir, f"{args.name}_s{s}",
                             f"step_{P.TRAIN_STEPS}.pt"),
            ]
            hit = next((p for p in pats if os.path.exists(p)), None)
            if hit is None:
                print(f"  [欠品] シード {s} の ckpt が無い。探した先:")
                for p in pats:
                    print(f"          {os.path.relpath(p, ROOT)}")
                continue
            ckpts.append((f"{args.name}_{P.SHORT[args.suite]}_s{s}", hit))
    if not ckpts:
        raise SystemExit("[eval] 評価する ckpt が無い。--ckpt か --ckpt-dir を指定する。")

    cells: list[tuple[str, list[str]]] = []
    for name, path in ckpts:
        base = [vpy(), "-u", os.path.join(ROOT, "scripts", "eval_batch.py"),
                "--ckpt", path, "--device", args.device,
                "--window", str(P.WINDOW),
                "--n-itr", str(er["n_itr"]), "--er-opt", er["er_opt"],
                "--er-lr", str(er["er_lr"]), "--er-w", str(er_w)]
        if args.compile:
            base.append("--compile")
        if args.cache_root:
            base += ["--cache-root", args.cache_root]
        base += shlex.split(args.extra)
        tag0 = f"{args.tag_prefix}{name}" if args.tag_prefix else name
        n_roll = args.n if args.n else (P.ROLLOUTS_LONG if is_long
                                        else P.ROLLOUTS_MAIN)
        tasks = args.tasks if args.tasks else list(range(10))
        if is_long:
            # ★1 プロセス 1 タスク。ここを破ると 2026-08-10 に撤回した誤りを繰り返す。
            for t in tasks:
                cells.append((f"{tag0}_t{t}",
                              base + ["--n", str(n_roll), "--B", "1",
                                      "--max-steps-auto", str(P.MAX_STEPS_AUTO_LONG),
                                      "--tasks", str(t),
                                      "--tag", f"{tag0}_t{t}"]))
        else:
            cell = base + ["--n", str(n_roll), "--B", str(args.B),
                           "--max-steps", str(P.MAX_STEPS_MAIN),
                           "--tag", tag0]
            if args.tasks:
                cell += ["--tasks"] + [str(t) for t in args.tasks]
            cells.append((tag0, cell))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--suite", required=True, choices=list(P.ALL_SUITES))
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="ckpt を直接指定（glob 可）")
    ap.add_argument("--ckpt-dir", default=None,
                    help="ここから --name と --seeds で ckpt を探す")
    ap.add_argument("--name", default="predvla",
                    help="--ckpt-dir 内の系列名（既定 main）")
    ap.add_argument("--seeds", default=None,
                    help="例 0-6 / 1,3,5。既定は protocol.py のシード")
    ap.add_argument("--protocol", default="main", choices=list(PROTOCOLS),
                    help="main=確定プロトコル Adam n_itr10 er_lr0.05（既定） / "
                         "sgd10=旧看板 SGD n_itr10（★アブレーションには使わない） / "
                         "ni0=ER 無し（開ループ）")
    ap.add_argument("--er-w", type=float, default=None,
                    help="ER の Complexity 重み。★既定 1.0（全スイート共通）。"
                         "2026-08-22 の掃引で 1.0 が頂点と確認済み")
    ap.add_argument("--B", type=int, default=16,
                    help="3 スイートの同時枠数（Long は常に 1）")
    ap.add_argument("--device", default="cpu",
                    help="★既定 cpu のまま論文の数値が出る（GPU は描画に使う）")
    ap.add_argument("--compile", action="store_true",
                    help="ER のロールを torch.compile する（CUDA のときだけ効く）")
    ap.add_argument("--cache-root", default=None,
                    help="ckpt の cfg にある PCA 基底を上書きする（通常は不要）")
    ap.add_argument("--jobs", type=int, default=0,
                    help="並列本数。0 で自動（コア数−2 と RAM から決める）")
    ap.add_argument("--tag-prefix", default="",
                    help="ログのタグに付ける接頭辞（測り直しを区別する）")
    ap.add_argument("--extra", default="",
                    help="eval_batch.py にそのまま渡す追加引数。"
                         "★1 つの文字列で渡す（例 --extra=\"--er-lambda-v 0\"）。"
                         "argparse が - 始まりを引数と誤認するのを避けるため")
    # --- 以下 2 つは★動作確認用。論文の数値を出すときは使わない ---
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="★タスクを絞る（動作確認用。論文の数値は 10 タスク全部）")
    ap.add_argument("--n", type=int, default=None,
                    help="★1 タスクあたりのロールアウト数を上書き"
                         "（動作確認用。論文は 3 スイート 50 / Long 5）")
    ap.add_argument("--redo", action="store_true",
                    help="完了済みのセルも測り直す")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    cells = build_cells(args)
    todo = [c for c in cells if args.redo or not done(c[0])]
    skip = len(cells) - len(todo)

    # 並列数: 1 セル 1 コア・約 6GB（B=16）/ 約 3GB（Long, B=1）
    per_gb = 3.0 if args.suite == P.SUITE_LONG else 6.0
    if args.jobs > 0:
        jobs = args.jobs
    else:
        import multiprocessing
        by_cpu = max(1, multiprocessing.cpu_count() - 2)
        by_ram = max(1, int((free_gb() - 4) / per_gb))
        jobs = max(1, min(by_cpu, by_ram, 12))
    print(f"[eval] スイート {args.suite}  プロトコル {args.protocol} "
          f"({PROTOCOLS[args.protocol]})")
    print(f"[eval] er_w {args.er_w if args.er_w is not None else P.ER_W}"
          f"  窓 {P.WINDOW}  device {args.device}")
    if args.suite == P.SUITE_LONG:
        print(f"[eval] ★Long: 1 プロセス 1 タスク・n={P.ROLLOUTS_LONG}/タスク・"
              f"打ち切りは最長デモ×{P.MAX_STEPS_AUTO_LONG}")
    print(f"[eval] セル {len(cells)} 件（完了済み {skip} 件を飛ばす）  "
          f"並列 {jobs} 本  1 セル約 {per_gb:.0f}GB  空き RAM {free_gb():.0f}GB")
    if args.tasks or args.n:
        print("[eval] " + "★" * 28)
        print(f"[eval] ★動作確認モード: tasks={args.tasks or '全部'} "
              f"n={args.n or '規定値'}")
        print("[eval] ★この結果は論文の数値と比較してはいけない"
              "（プロトコルが違う）")
        print("[eval] " + "★" * 28)

    if args.dry_run:
        for tag, cmd in todo:
            print(f"  $ {' '.join(cmd)}")
        print(f"[eval] --dry-run なので実行しない（{len(todo)} セル）")
        return 0
    if not todo:
        print("[eval] 全部終わっている。集計は scripts/aggregate.py。")
        return 0

    running: list[tuple[str, subprocess.Popen, object]] = []
    q = list(todo)
    t0 = time.time()
    n_done = 0
    try:
        while q or running:
            while q and len(running) < jobs:
                tag, cmd = q.pop(0)
                lp = os.path.join(LOG_DIR, f"ev_{tag}.log")
                fh = open(lp, "w")
                pr = subprocess.Popen(cmd, cwd=ROOT, stdout=fh,
                                      stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL)
                running.append((tag, pr, fh))
                print(f"  [{time.time()-t0:6.0f}s] 起動 {tag}  "
                      f"（稼働 {len(running)}/{jobs}, 残 {len(q)}）", flush=True)
            time.sleep(5)
            for item in list(running):
                tag, pr, fh = item
                if pr.poll() is None:
                    continue
                fh.close()
                running.remove(item)
                n_done += 1
                ok = done(tag)
                print(f"  [{time.time()-t0:6.0f}s] 完了 {tag}  "
                      f"rc={pr.returncode} {'' if ok else '★「全体」行が無い'}"
                      f"  ({n_done}/{len(todo)})", flush=True)
    except KeyboardInterrupt:
        print("\n[eval] 中断。走っているセルを止める。")
        for tag, pr, fh in running:
            pr.terminate()
            fh.close()
        return 130

    nok = sum(1 for tag, _ in todo if not done(tag))
    print(f"\n[eval] 終了 {time.time()-t0:.0f}s。"
          f"未完了 {nok} 件。ログ {LOG_DIR}")
    print(f"[eval] 集計: python scripts/aggregate.py --suite {args.suite}")
    return 1 if nok else 0


if __name__ == "__main__":
    sys.exit(main())
