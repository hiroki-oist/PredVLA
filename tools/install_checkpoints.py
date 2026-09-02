#!/usr/bin/env python
"""学習済み ckpt を `checkpoints/` に置く（評価に要らない中身を落として小さくする）。

## なぜ「slim 化」するか

学習の ckpt は再開できるよう全部入りで、**1 本 315MB** ある。内訳の大半は評価では
使わないものである。

    model   675,732 パラメータ                        約   2.7MB   ★評価が読む
    cfg     設定                                      約     数 KB ★評価が読む
    C       自由変数 c（系列 × 時刻 × r × 4 層）      約 100MB     学習の再開用
    opt     Adam のモーメント（C の分が支配的）        約 205MB     学習の再開用
    rng     numpy / torch の乱数状態                   小           学習の再開用

評価（`scripts/eval_batch.py` / `benchmark/eval_suite.py`）が触るのは `model` と
`cfg` と `step` だけである。**テスト時の自由変数 c は毎エピソード事前値から張り直す**
ので、学習時の C を持っていても使わない。だから配布用は 3 つだけ残せばよく、
315MB → 約 2.7MB（1/115）になる。

★slim 化した ckpt では **学習の再開ができない**（C も opt も無い）。再開したいときは
  元の全部入りを使う。`--keep-full` を付ければそのままコピーする。

## 使い方

    # 何をどこに置くか見るだけ
    python tools/install_checkpoints.py --list

    # 代表セット（既定）をまとめて入れる。<dir> の中に <名前>.pt か
    # <名前>/step_<N>.pt があるものを拾う
    python tools/install_checkpoints.py --from <ckpt の置き場>

    # 自分で学習した run を 1 つ入れる
    python tools/install_checkpoints.py --src results/predvla_spatial_s1/step_30000.pt \\
        --name predvla_spatial_s1

    # 手元の ckpt をまとめて小さくする（置き換え）
    python tools/install_checkpoints.py --slim results/*/step_30000.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import protocol as P  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints")

# 評価が読むキーだけ残す。★増やすときは eval 側が本当に読むか確かめてから。
KEEP = ("model", "cfg", "step")

SUITES = ("spatial", "goal", "object", "long")

# 代表セット: (名前, その run の step)
#   名前は predvla/protocol.py の命名規則そのもの（<系列>_<スイート>_s<シード>）。
#   --from に渡すディレクトリの中から、この名前で
#       <名前>.pt              平置き
#       <名前>/step_<step>.pt  学習の出力ディレクトリ
#   のどちらかを探す。無ければ欠品として報告して飛ばす。
#
#   ★別の命名で置かれた ckpt を入れたいときは --src と --name で 1 本ずつ渡す。
def default_set() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    # 本線: 4 スイート × シード 1
    for s in SUITES:
        out.append((f"predvla_{s}_s1", P.TRAIN_STEPS))
    # ベースライン: 4 スイート × シード 0
    for s in SUITES:
        out.append((f"bc_lstm_{s}_s0", P.BASELINE_STEPS))
        out.append((f"bc_transformer_{s}_s0", P.BASELINE_STEPS))
    # はしご（spatial × シード 1）。本線 L2〜L6 と側枝 3 本。
    #   L0 / L1 は本線 ckpt を、L7 は BC-LSTM をそのまま使うので置かない。
    for rung in ("L2", "L2b", "L2-s1", "L2b-s1", "L3", "L4", "L5", "L6"):
        out.append((f"ladder_{rung}_spatial_s1", P.TRAIN_STEPS))
    # アブレーション（spatial × シード 0）
    #   A4〜A6 は本線 ckpt に評価フラグを足すだけなので置かない。
    for abl in ("A1_uniform_tau", "A2_no_pb", "A3_no_efference"):
        out.append((f"{abl}_spatial_s0", P.TRAIN_STEPS))
    return out


def locate(src_dir: str, name: str, step: int) -> str:
    """<名前>.pt と <名前>/step_<step>.pt の両方を探す。"""
    for p in (os.path.join(src_dir, f"{name}.pt"),
              os.path.join(src_dir, name, f"step_{step}.pt")):
        if os.path.exists(p):
            return p
    return ""


def slim(ck: dict) -> dict:
    return {k: ck[k] for k in KEEP if k in ck}


def size_mb(p: str) -> float:
    return os.path.getsize(p) / 1e6


def install_one(src: str, dst: str, keep_full: bool, dry: bool) -> bool:
    if not os.path.exists(src):
        print(f"  [欠品] {os.path.relpath(src, ROOT) if src.startswith(ROOT) else src}")
        return False
    if dry:
        print(f"  [dry] {src}\n        -> {os.path.relpath(dst, ROOT)}")
        return True
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = ck if keep_full else slim(ck)
    missing = [k for k in ("model", "cfg") if k not in out]
    if missing:
        print(f"  [不正] {src} に {missing} が無い。飛ばす")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(out, dst)
    print(f"  {os.path.basename(dst):40s} {size_mb(src):8.1f}MB -> "
          f"{size_mb(dst):6.2f}MB  step {out.get('step','?')}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--from", dest="src_dir", default=None,
                    help="ckpt の置き場。代表セットの名前でここから探す")
    ap.add_argument("--src", default=None, help="ckpt を 1 つだけ入れる")
    ap.add_argument("--name", default=None,
                    help="--src と一緒に使う。置く名前（拡張子なし）")
    ap.add_argument("--slim", nargs="*", default=None, metavar="CKPT",
                    help="渡した ckpt を小さくして同じ場所に書き戻す")
    ap.add_argument("--list", action="store_true",
                    help="代表セットの一覧を出すだけ")
    ap.add_argument("--keep-full", action="store_true",
                    help="小さくせずそのままコピーする（学習を再開したいとき）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.list:
        print(f"{'名前':38s}{'step':>7s}   {'いま入っているか':>8s}")
        for name, step in default_set():
            have = os.path.exists(os.path.join(CKPT_DIR, f"{name}.pt"))
            mark = (f"{size_mb(os.path.join(CKPT_DIR, f'{name}.pt')):.2f}MB"
                    if have else "—")
            print(f"{name:38s}{step:>7d}   {mark:>8s}")
        n_have = sum(os.path.exists(os.path.join(CKPT_DIR, f"{n}.pt"))
                     for n, _ in default_set())
        print(f"\n代表セット {len(default_set())} 本 / いま入っている {n_have} 本。"
              f"slim 化すると 1 本 2〜3MB（元は 1 本 315MB）。")
        print("L0 / L1 は本線 ckpt を、L7 は BC-LSTM の ckpt をそのまま使うので"
              "別に置く必要はない。")
        print("A4 / A5 / A6 も本線 ckpt に評価フラグを足すだけなので置かない。")
        return 0

    if a.slim is not None:
        n = 0
        for pat in a.slim:
            for p in sorted(glob.glob(pat)) or [pat]:
                if not os.path.exists(p):
                    print(f"  [欠品] {p}")
                    continue
                before = size_mb(p)
                if a.dry_run:
                    print(f"  [dry] {p}  {before:.1f}MB")
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                torch.save(slim(ck), p)
                print(f"  {p}  {before:.1f}MB -> {size_mb(p):.2f}MB")
                n += 1
        print(f"{n} 本を小さくした")
        return 0

    if a.src:
        if not a.name:
            sys.exit("--src には --name が要る")
        ok = install_one(a.src, os.path.join(CKPT_DIR, f"{a.name}.pt"),
                         a.keep_full, a.dry_run)
        return 0 if ok else 1

    if not a.src_dir:
        sys.exit("--from か --src か --slim か --list のどれかを指定する。"
                 "\n  例: python tools/install_checkpoints.py --list")

    print(f"元: {a.src_dir}")
    print(f"先: {os.path.relpath(CKPT_DIR, ROOT)}"
          + ("（そのままコピー）" if a.keep_full else "（slim 化する）"))
    ok = miss = 0
    for name, step in default_set():
        src = locate(a.src_dir, name, step) or os.path.join(
            a.src_dir, name, f"step_{step}.pt")
        if install_one(src, os.path.join(CKPT_DIR, f"{name}.pt"),
                       a.keep_full, a.dry_run):
            ok += 1
        else:
            miss += 1
    print(f"\n{ok} 本を入れた / {miss} 本は元が見つからなかった")
    if not a.dry_run and os.path.isdir(CKPT_DIR):
        total = sum(os.path.getsize(os.path.join(CKPT_DIR, f))
                    for f in os.listdir(CKPT_DIR)
                    if f.endswith(".pt")
                    and os.path.isfile(os.path.join(CKPT_DIR, f)))
        print(f"checkpoints/ の合計 {total/1e6:.0f}MB")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
