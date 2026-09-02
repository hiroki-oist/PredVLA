#!/usr/bin/env python
"""アブレーション（表③）をコマンド 1 本で回す。

4 つのアブレーションは 2 種類に分かれる。ここを混同すると無駄に学習し直す。

  学習時 (train)  そのフラグを付けて **再学習が要る**
      no_efference_a2v   A→V の efference（前 step の â, q̂ を V に渡す経路）を切る
      no_pb_v2a          V→A の PB（W_pb·d^V, 256→32）を切る
  テスト時 (test)  既存 ckpt に **評価フラグを足すだけ**。再学習は不要
      no_er_vision       ER 時のみ視覚予測誤差の経路を切る（★学習時は使う）
      no_er_action       ER 時のみ固有感覚予測誤差の経路を切る（★学習時は使う）
      no_online_er       オンライン ER 自体を切る（NI0、n_itr=0）

使い方
    # 全部（テスト時 3 つは既存 ckpt で即、学習時 2 つは専用 ckpt があれば評価だけ）
    python scripts/run_ablation.py --ablation all --suite libero_spatial \
        --seeds 0-6 --ckpt-dir checkpoints

    # 学習時アブレーションを学習からやり直して評価まで
    python scripts/run_ablation.py --ablation no_pb_v2a --suite libero_spatial \
        --seeds 0-2 --mode both

    # 何が走るか見るだけ
    python scripts/run_ablation.py --ablation all --suite libero_spatial \
        --seeds 0-6 --dry-run

★評価プロトコル・ロールアウト数・Long の 1 プロセス 1 タスクは scripts/evaluate.py が
  predvla/protocol.py から組み立てる。このスクリプトはアブレーションのフラグを足すだけで
  プロトコルには触らない（取り違えの余地を作らない）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


def train_one(abl: str, spec: dict, suite: str, seed: int, device: str,
              dry: bool) -> tuple[int, str]:
    """アブレーション付きで 1 シード学習する。(rc, ckpt パス) を返す。"""
    run_name = f"{abl}_{P.SHORT[suite]}_s{seed}"
    cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "train.py"),
           "--config", os.path.join(ROOT, "configs",
                                    f"predvla_{P.SHORT[suite]}.yaml"),
           "--device", device, "--set",
           f"train.run_name={run_name}", f"train.seed={seed}",
           f"train.total_steps={P.TRAIN_STEPS}"]
    for k, v in spec.get("set", {}).items():
        cmd.append(f"{k}={str(v).lower() if isinstance(v, bool) else v}")
    rc = run(cmd, dry)
    return rc, os.path.join(ROOT, "results", run_name,
                            f"step_{P.TRAIN_STEPS}.pt")


def find_ckpt(ckpt_dir: str, abl: str, suite: str, seed: int, kind: str) -> str:
    """テスト時アブレーションは本線 ckpt、学習時アブレーションは専用 ckpt を使う。"""
    tag = "predvla" if kind == "test" else abl
    short = P.SHORT[suite]
    for p in (os.path.join(ckpt_dir, f"{tag}_{short}_s{seed}.pt"),
              os.path.join(ckpt_dir, f"{tag}_{short}_s{seed}",
                           f"step_{P.TRAIN_STEPS}.pt"),
              os.path.join(ROOT, "results", f"{tag}_{short}_s{seed}",
                           f"step_{P.TRAIN_STEPS}.pt")):
        if os.path.exists(p):
            return p
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ablation", required=True,
                    help="/".join(P.ABLATIONS) + " / all")
    ap.add_argument("--suite", default="libero_spatial", choices=list(P.ALL_SUITES))
    ap.add_argument("--seeds", default="0-6", help="例 0-6 / 1,3,5")
    ap.add_argument("--mode", choices=["train", "rollout", "both"],
                    default="rollout",
                    help="train=再学習のみ / rollout=既存 ckpt で評価のみ（既定） / both")
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"),
                    help="本線 ckpt の置き場（テスト時アブレーションが読む）")
    ap.add_argument("--device", default="auto", help="学習の device")
    ap.add_argument("--eval-device", default="cpu", help="評価の device")
    ap.add_argument("--protocol", default="main", choices=["main", "sgd10"],
                    help="main=確定プロトコル Adam n_itr10（既定）。"
                         "★sgd10 は使わない: A3 が SGD10 でだけ 1.00 に崩壊する")
    ap.add_argument("--jobs", type=int, default=0, help="評価の並列本数（0 で自動）")
    ap.add_argument("--no-aggregate", action="store_true", help="最後の集計を出さない")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.ablation == "all":
        abls = list(P.ABLATIONS)
    else:
        abls = []
        for x in a.ablation.split(","):
            try:
                abls.append(P.resolve_ablation(x.strip()))
            except KeyError:
                sys.exit(f"未知のアブレーション: {x}\n"
                         f"  選べるのは: {', '.join(P.ABLATIONS)} / all\n"
                         f"  旧名も受ける: {', '.join(P.ABLATION_ALIASES)}")
    seeds = parse_seeds(a.seeds)
    is_long = a.suite == P.SUITE_LONG

    print("=" * 78)
    print(f" アブレーション  スイート {a.suite}  シード {seeds}")
    print(f" mode={a.mode}  protocol={a.protocol}  eval_device={a.eval_device}")
    if len(seeds) < 7:
        print(f" ★シード {len(seeds)} 本。この系列は n=3 の ±5pt が n=7 で消えた実例が"
              f" 4 件ある。確定を主張するなら 7 本以上にする")
    if is_long:
        print(" ★Long: evaluate.py が 1 プロセス 1 タスクに割る")
    print("=" * 78)

    rc = 0
    for abl in abls:
        spec = P.ABLATIONS[abl]
        print(f"\n=== {abl}  [{spec['kind']}]  {spec['desc']} ===")

        if spec["kind"] == "test" and a.mode == "train":
            print("  （テスト時アブレーションなので再学習は不要。何もしない）")
            continue

        # ---------------- 学習 ----------------
        trained: dict[int, str] = {}
        if spec["kind"] == "train" and a.mode in ("train", "both"):
            for s in seeds:
                r, ck = train_one(abl, spec, a.suite, s, a.device, a.dry_run)
                rc |= r
                trained[s] = ck
        if a.mode == "train":
            continue

        # ---------------- 評価 ----------------
        ckpts: list[str] = []
        missing: list[int] = []
        for s in seeds:
            ck = trained.get(s) or find_ckpt(a.ckpt_dir, abl, a.suite, s,
                                             spec["kind"])
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

        cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "evaluate.py"),
               "--suite", a.suite, "--protocol", a.protocol,
               "--device", a.eval_device, "--tag-prefix", f"{abl}__",
               "--ckpt"] + ckpts
        if a.jobs:
            cmd += ["--jobs", str(a.jobs)]
        flags = spec.get("eval_flags", {})
        if flags:
            # evaluate.py の --extra は 1 つの文字列で受ける（- 始まりを argparse に
            # 引数と誤認させないため）。= で繋ぐのも同じ理由。
            cmd += ["--extra=" + " ".join(f"{k} {v}" for k, v in flags.items())]
        if a.dry_run:
            cmd.append("--dry-run")
        rc |= run(cmd, False)      # evaluate.py 自身が --dry-run を解釈する

    # ---------------- 集計 ----------------
    if not a.no_aggregate and not a.dry_run and a.mode != "train":
        print("\n" + "=" * 78)
        print(" 集計")
        print("=" * 78)
        for abl in abls:
            subprocess.call([vpy(), "-u",
                             os.path.join(ROOT, "scripts", "aggregate.py"),
                             "--match", f"{abl}__"], cwd=ROOT)
        m, sd, n = P.MAIN_TABLE[a.suite]
        print(f"\n--- 論文の参考値（{a.suite}、確定プロトコル MAIN）---")
        print(f"  {'主表（アブレーション無し）':32s} {m:6.2f} ± {sd:.2f} (n={n})")
        for x in abls:
            v = P.ABLATION_TABLE.get(x, {}).get(a.suite)
            if v is None:
                old = P.ABLATION_ERW09_NOT_FOR_CITATION.get(x, {}).get(a.suite)
                extra = (f"（er_w=0.9 の旧値 {old:.2f}。★引用禁止）"
                         if old is not None else "")
                print(f"  {x:32s} 確定プロトコルでは未取得 {extra}")
            else:
                print(f"  {x:32s} {v[0]:6.2f} ± {v[1]:.2f} (n={v[2]})")
        print("  ※ ロールアウトのラン間ノイズが ±1〜4.5pt あるので "
              "±2pt 程度で乗れば再現できている")
    return rc


if __name__ == "__main__":
    sys.exit(main())
