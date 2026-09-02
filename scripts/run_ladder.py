#!/usr/bin/env python
"""「はしご」（PredVLA → BC-LSTM）をコマンド 1 本で回す。

段の定義は `predvla/ladder.py` にある。各段は上の段の設定を引き継いで、次の機構を
1 つ落とす。パラメータ数は全段 675,732 ± 1% に揃えてある（段ごとに層幅 d が違う）。

    L0  無傷の PredVLA               本線 ckpt を確定プロトコルで評価するだけ（再学習不要）
    L1  − オンライン ER              本線 ckpt を n_itr=0 で評価するだけ（再学習不要）
    L2  − 自由変数 c と自由エネルギー  再学習が要る
    L3  + 観測を前向きに入れる（予測は保つ）  〃
    L4  − 予測（予測 head と予測損失を外す）  〃
    L5  − 多時定数（全層 τ=1）        〃
    L6  − PV-RNN セル（LSTM に）      〃
    L7  独立実装の BC-LSTM             benchmark/ 側（手順だけ出す）

  側枝: L2b（生の視覚を A_low へ）/ L2-s1 / L2b-s1

  ★段番号は 2026-09-01 に付け替えてある。旧 L3'（予測を保ったまま直入力）が
    新 L3 になり、旧 L3〜L6 が 1 つずつ繰り上がった（predvla/ladder.py の対応表を見る）。

使い方

    # 何が走るか見るだけ
    python scripts/run_ladder.py --rung all --suite libero_spatial --seeds 1-7 --dry-run

    # 段の config を書き出すだけ（configs/ladder/ に出る）
    python scripts/run_ladder.py --rung all --suite libero_spatial --seeds 1-7 --mode config

    # パラメータ数が本線に揃っているか確認する
    python scripts/run_ladder.py --rung all --check-params

    # 1 段を学習から評価まで
    python scripts/run_ladder.py --rung L3 --suite libero_spatial --seeds 1-7 --mode both

    # 学習済みの段を評価だけ
    python scripts/run_ladder.py --rung L4,L5 --suite libero_goal --seeds 1-7 --mode rollout

★評価プロトコルは `scripts/evaluate.py` が `predvla/protocol.py` から組み立てる。
  このスクリプトは **L1 以降で `--protocol ni0`（n_itr=0）を強制する**だけで、
  ロールアウト数・打ち切り・Long の 1 プロセス 1 タスクには触らない。
  L2 以降は自由変数 c を持たないので ER（n_itr>0）は定義できない。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import ladder as L  # noqa: E402
from predvla import protocol as P  # noqa: E402


def vpy() -> str:
    p = os.path.join(ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


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


def run(cmd: list[str], dry: bool) -> int:
    print("    $ " + " ".join(cmd), flush=True)
    return 0 if dry else subprocess.call(cmd, cwd=ROOT)


def base_cfg(suite: str) -> dict:
    p = os.path.join(ROOT, "configs", f"predvla_{P.SHORT[suite]}.yaml")
    if not os.path.exists(p):
        sys.exit(f"本線 config が無い: {p}")
    return yaml.safe_load(open(p))


def run_name(rung: str, suite: str, seed: int) -> str:
    return f"ladder_{rung}_{P.SHORT[suite]}_s{seed}"


def write_cfg(rung: str, suite: str, seed: int, steps: int, dry: bool) -> str:
    """段の config を configs/ladder/ に書き出してパスを返す。"""
    cfg = L.make_cfg(rung, base_cfg(suite), seed,
                     run_name(rung, suite, seed), steps)
    path = L.config_path(ROOT, rung, P.SHORT[suite], seed)
    if dry:
        print(f"    [config] {os.path.relpath(path, ROOT)}（--dry-run なので書かない）")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head = (f"# はしご {rung}: {L.RUNGS[rung]['desc']}\n"
            f"#   {suite} / seed {seed}\n"
            f"#   ★生成元 scripts/run_ladder.py（手で編集しない）\n"
            f"#   ★評価は n_itr=0（自由変数 c を持たないので ER が定義できない）\n")
    with open(path, "w") as f:
        f.write(head)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    # 書いたものを読み直して型を確認する（YAML 1.1 は "6e-05" を文字列として読む）
    back = yaml.safe_load(open(path))
    assert isinstance(back["train"]["lr"], float), \
        f"lr が float として読めない: {back['train']['lr']!r}"
    return path


def find_main_ckpt(ckpt_dir: str, suite: str, seed: int) -> str:
    short = P.SHORT[suite]
    for p in (os.path.join(ckpt_dir, f"predvla_{short}_s{seed}.pt"),
              os.path.join(ckpt_dir, f"predvla_{short}_s{seed}",
                           f"step_{P.TRAIN_STEPS}.pt"),
              os.path.join(ROOT, "results", f"predvla_{short}_s{seed}",
                           f"step_{P.TRAIN_STEPS}.pt")):
        if os.path.exists(p):
            return p
    return ""


def ladder_ckpt(rung: str, suite: str, seed: int, steps: int,
                ckpt_dir: str = "") -> str:
    """学習した段の ckpt を探す。

    配布 ckpt の平置き（checkpoints/<run 名>.pt）と学習の出力
    （results/<run 名>/step_<N>.pt）の両方を見る。本線 ckpt の探し方
    （find_main_ckpt）と揃えてある。
    """
    name = run_name(rung, suite, seed)
    cands = [os.path.join(ROOT, "results", name, f"step_{steps}.pt")]
    if ckpt_dir:
        cands = [os.path.join(ckpt_dir, f"{name}.pt"),
                 os.path.join(ckpt_dir, name, f"step_{steps}.pt")] + cands
    for p in cands:
        if os.path.exists(p):
            return p
    # 見つからないときは「学習の出力があるべき場所」を返す（欠品の表示に使う）
    return cands[-1]


def check_params(rungs: list[str], suite: str) -> int:
    """各段のパラメータ数を数えて本線（675,732）と比べる。"""
    cfg0 = base_cfg(suite)
    n0 = L.count_params(cfg0)
    print(f"  {'段':<8}{'d':>5}{'params':>12}{'本線比':>10}")
    print(f"  {'L0 本線':<8}{cfg0['model']['d']:>5}{n0:>12,}{1.0:>9.3f}x")
    bad = 0
    for r in rungs:
        spec = L.RUNGS[r]
        if spec["kind"] == "baseline":
            print(f"  {r:<8}{'—':>5}{'（別実装）':>12}")
            continue
        cfg = L.make_cfg(r, cfg0, 1, "x", P.TRAIN_STEPS)
        n = L.count_params(cfg)
        rel = n / L.PARAMS_TARGET
        note = spec.get("params_note")
        ok = abs(rel - 1.0) <= L.PARAMS_TOL or note is not None
        bad += 0 if ok else 1
        mark = "" if abs(rel - 1.0) <= L.PARAMS_TOL else (
            f"  ※{note}" if note else "  ★±1% を外れている")
        print(f"  {r:<8}{cfg['model']['d']:>5}{n:>12,}{rel:>9.3f}x{mark}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--rung", default="all",
                    help="段の id をカンマ区切りで / all（本線＋側枝）/ "
                         "main（本線だけ）/ side（側枝だけ）。"
                         f"選べるのは: {', '.join(L.RUNGS)}")
    ap.add_argument("--suite", default="libero_spatial", choices=list(P.ALL_SUITES))
    ap.add_argument("--seeds", default="1-7", help="例 1-7 / 1,3,5")
    ap.add_argument("--mode", choices=["config", "train", "rollout", "both"],
                    default="both",
                    help="config=config を書くだけ / train=学習のみ / "
                         "rollout=既存 ckpt で評価のみ / both=両方（既定）")
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"),
                    help="本線 ckpt の置き場（L1 が読む）")
    ap.add_argument("--steps", type=int, default=P.TRAIN_STEPS)
    ap.add_argument("--device", default="auto", help="学習の device")
    ap.add_argument("--eval-device", default="cpu", help="評価の device")
    ap.add_argument("--jobs", type=int, default=0, help="評価の並列本数（0 で自動）")
    ap.add_argument("--check-params", action="store_true",
                    help="パラメータ数が本線に揃っているかだけ見る")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.rung == "all":
        rungs = list(L.ALL)
    elif a.rung == "main":
        rungs = list(L.MAIN_LINE)
    elif a.rung == "side":
        rungs = list(L.SIDE)
    else:
        rungs = []
        for x in a.rung.split(","):
            try:
                rungs.append(L.resolve(x))
            except KeyError:
                sys.exit(f"未知の段: {x}\n  選べるのは: {', '.join(L.RUNGS)} / "
                         f"all / main / side")

    if a.check_params:
        return 1 if check_params(rungs, a.suite) else 0

    seeds = parse_seeds(a.seeds)
    print("=" * 78)
    print(f" はしご  スイート {a.suite}  シード {seeds}  mode={a.mode}")
    print(f" 段: {', '.join(rungs)}")
    print(" ★評価プロトコルは段ごと: L0 は確定プロトコル、L1 以降は n_itr=0")
    if len(seeds) < 7:
        print(f" ★シード {len(seeds)} 本。σ≈4〜11pt あるので、確定を主張するなら 7 本以上")
    print("=" * 78)

    rc = 0
    for r in rungs:
        spec = L.RUNGS[r]
        print(f"\n=== {r}  [{spec['kind']}]  {spec['desc']} ===")

        if spec["kind"] == "baseline":
            print("  L6 は独立実装のベースライン。benchmark/ 側で回す:")
            print(f"    $ {os.path.basename(vpy())} tools/configs_from_ckpt.py --baselines")
            print(f"    $ {os.path.basename(vpy())} benchmark/train.py "
                  f"--config configs/baseline_bc_lstm_{P.SHORT[a.suite]}.yaml \\")
            print(f"        --set train.run_name=bcrnn_{P.SHORT[a.suite]}_s1 train.seed=1")
            print(f"    $ {os.path.basename(vpy())} benchmark/eval_suite.py "
                  f"--ckpt results/bcrnn_{P.SHORT[a.suite]}_s1/step_50000.pt --n 50")
            print("  ★ベースラインは 50,000 step（PredVLA は 30,000）")
            continue

        # ---------------- config ----------------
        cfgs: dict[int, str] = {}
        if spec["kind"] == "train":
            for s in seeds:
                cfgs[s] = write_cfg(r, a.suite, s, a.steps, a.dry_run)
            if a.mode == "config":
                continue
        elif a.mode == "config":
            print("  （L1 は再学習が要らないので config は作らない）")
            continue

        # ---------------- 学習 ----------------
        if spec["kind"] == "train" and a.mode in ("train", "both"):
            for s in seeds:
                rc |= run([vpy(), "-u", os.path.join(ROOT, "scripts", "train.py"),
                           "--config", cfgs[s], "--device", a.device], a.dry_run)
        if a.mode == "train":
            continue

        # ---------------- 評価 ----------------
        ckpts, missing = [], []
        for s in seeds:
            ck = (find_main_ckpt(a.ckpt_dir, a.suite, s) if spec["kind"] == "test"
                  else ladder_ckpt(r, a.suite, s, a.steps, a.ckpt_dir))
            if ck and (a.dry_run or os.path.exists(ck)):
                ckpts.append(ck)
            else:
                missing.append(s)
        if missing:
            print(f"  [欠品] ckpt が無いシード {missing}")
            if spec["kind"] == "train":
                print("          --mode both を付けると学習からやり直す")
            else:
                print(f"          本線 ckpt を {a.ckpt_dir} に置く"
                      f"（predvla_{P.SHORT[a.suite]}_s<S>.pt）")
        if not ckpts:
            continue

        # 評価プロトコルは段の定義（predvla/ladder.py の protocol 欄）が決める。
        # L0 だけ確定プロトコル（ER あり）、L1 以降は n_itr=0。
        cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "evaluate.py"),
               "--suite", a.suite, "--protocol", spec.get("protocol") or "ni0",
               "--device", a.eval_device,
               # 学習した段は run_name にすでに ladder_<段> が入っているので
               # 接頭辞を重ねない。L1 だけは本線 ckpt を使うので段名を足す。
               "--tag-prefix", (f"ladder_{r}__" if spec["kind"] == "test" else ""),
               "--ckpt"] + ckpts
        if a.jobs:
            cmd += ["--jobs", str(a.jobs)]
        if a.dry_run:
            cmd.append("--dry-run")
        rc |= run(cmd, False if a.dry_run else a.dry_run)

    if not a.dry_run:
        print("\n=== 集計 ===")
        run([vpy(), "-u", os.path.join(ROOT, "scripts", "aggregate.py"),
             "--match", "ladder_"], a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
