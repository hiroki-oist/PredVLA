#!/usr/bin/env python
"""データを用意する。3 段構えで、下に行くほど計算が要る。

  (A) 凍結特徴キャッシュを配布物から置く   ★推奨。約 250MB。数分
  (B) LIBERO のデモから自分で作る          約 32GB のダウンロード + GPU 数時間
  (C) LIBERO のデモだけ落として (B) をやる

    python tools/prepare_data.py --check                 いま何が揃っているか
    python tools/prepare_data.py --download              LIBERO のデモを落とす
    python tools/prepare_data.py --build main            3 スイート用のキャッシュを作る
    python tools/prepare_data.py --build long            Long 用のキャッシュを作る
    python tools/prepare_data.py --build all             両方

★PCA 基底がスイート群で違う（取り違えると数値が再現しない）
--------------------------------------------------------------------------
2026-08-22 に実キャッシュを検証して判った実態は次のとおり。

  data/cache_l64      視覚 PCA と q の正規化統計を **libero_spatial のデモだけで**
                      当て、それを spatial / goal / object / libero_10 / libero_90 の
                      符号化に使い回している（= 3 スイート共通の基底）
  data/cache_ps_sp    上と **同じ手順をもう一度**回した結果。視覚基底と正規化統計は
                      cache_l64 と**バイト単位で同一**、言語 PCA だけ再当てはめのぶん
                      5e-5 ずれる。spatial の特徴の差は最大 2e-6（float32 の丸め相当）。
                      → 実質は同じもの。論文が spatial にこちらを使っているので
                        ディレクトリ構成を再現する意味で別に作る。
  data/cache_l64_lg   視覚 PCA と正規化統計を **libero_10 のデモで張り直した** もの。
                      これは cache_l64 と本当に違う（基底の最大差 0.14）。Long 専用。

したがって作り直す単位は 2 つだけである（main と long）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import protocol as P  # noqa: E402

# 作り直しの単位。(cache_root, 基底を当てるスイート, そのキャッシュに入れるスイート)
PROFILES = {
    # 3 スイート共通。基底は libero_spatial のデモで当てる。
    "main": dict(cache_root="data/cache_l64",
                 fit="libero_spatial",
                 encode=["libero_spatial", "libero_goal", "libero_object"]),
    # spatial 用（論文のディレクトリ構成の再現。中身は main と実質同じ）
    "spatial_dir": dict(cache_root="data/cache_ps_sp",
                        fit="libero_spatial",
                        encode=["libero_spatial"]),
    # Long 専用。基底を libero_10 で張り直す。
    "long": dict(cache_root="data/cache_l64_lg",
                 fit="libero_10",
                 encode=["libero_10"]),
}
BASIS_FILES = ("pca_agentview.npz", "pca_eye.npz", "pca_language.npz",
               "norm_stats.npz")


def vpy() -> str:
    p = os.path.join(ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def check() -> int:
    print("=" * 68)
    print(" データの状況")
    print("=" * 68)
    bad = 0

    print("\n--- LIBERO のデモ hdf5（(B)(C) で必要。(A) だけなら不要）---")
    for s in P.ALL_SUITES:
        d = os.path.join(ROOT, "data", "libero", s)
        n = len([f for f in os.listdir(d)
                 if f.endswith(".hdf5")]) if os.path.isdir(d) else 0
        print(f"  {'OK ' if n >= 10 else '無 '} {s:16s} {n}/10 タスク  {d}")

    print("\n--- 凍結特徴キャッシュ（★学習と評価に必須）---")
    for name, pf in PROFILES.items():
        cr = os.path.join(ROOT, pf["cache_root"])
        basis = sum(os.path.exists(os.path.join(cr, b)) for b in BASIS_FILES)
        print(f"  [{name}] {pf['cache_root']}  基底 {basis}/4")
        for s in pf["encode"]:
            d = os.path.join(cr, s)
            n = len([f for f in os.listdir(d)
                     if f.endswith(".h5")]) if os.path.isdir(d) else 0
            ok = (n >= 10 and basis == 4)
            bad += 0 if ok else 1
            print(f"      {'OK ' if ok else '★  '} {s:16s} {n}/10 タスク")

    print("\n--- 学習・評価が実際に読む先（predvla/protocol.py の CACHE_ROOT）---")
    for s in P.ALL_SUITES:
        cr = P.CACHE_ROOT[s]
        d = os.path.join(ROOT, cr, s)
        n = len([f for f in os.listdir(d)
                 if f.endswith(".h5")]) if os.path.isdir(d) else 0
        print(f"  {'OK ' if n >= 10 else '★  '} {s:16s} -> {cr}  {n}/10")

    print("\n" + "=" * 68)
    if bad:
        print(f" 足りないもの {bad} 件 → --build all で作る"
              f"（または配布キャッシュを data/ に展開する）")
    else:
        print(" すべて揃っている。")
    print("=" * 68)
    return bad


def download() -> int:
    """LIBERO 付属のダウンローダを呼ぶ。約 32GB（libero_90 を除く 4 スイート）。"""
    lib = os.path.join(ROOT, "third_party", "LIBERO")
    dl = os.path.join(lib, "benchmark_scripts", "download_libero_datasets.py")
    if not os.path.exists(dl):
        print(f"★{dl} が無い。先に bash setup.sh を走らせて LIBERO を clone する。")
        return 1
    tgt = os.path.join(ROOT, "data", "libero")
    os.makedirs(tgt, exist_ok=True)
    print(f"[prepare] LIBERO のデモを落とす -> {tgt}")
    print("  ★約 32GB。回線次第で数十分〜数時間かかる。")
    rc = 0
    for suite in ("libero_spatial", "libero_goal", "libero_object", "libero_10"):
        cmd = [vpy(), dl, "--datasets", suite, "--save-dir", tgt]
        print(f"  $ {' '.join(cmd)}")
        rc |= subprocess.call(cmd, cwd=lib)
    return rc


def build(profile: str, force: bool, device: str, max_demos: int) -> int:
    pf = PROFILES[profile]
    cr = pf["cache_root"]
    cfg = os.path.join(ROOT, "configs", "preprocess.yaml")
    entry = os.path.join(ROOT, "tools", "_preprocess_cli.py")
    print(f"\n[prepare] === profile={profile}  cache_root={cr} ===")
    print(f"[prepare] 基底を当てるスイート: {pf['fit']}")

    # 基底を当てるスイートを **必ず最初に** 回す。preprocess.run() は基底ファイルが
    # 無いときだけ当てはめ、あれば読むので、順番がそのまま「どのスイートで基底を
    # 当てたか」を決める。ここを間違えると数値が再現しない。
    order = [pf["fit"]] + [s for s in pf["encode"] if s != pf["fit"]]
    rc = 0
    for i, suite in enumerate(order):
        cmd = [vpy(), "-u", entry, "--config", cfg, "--suite", suite,
               "--device", device, "--set", f"paths.cache_root={cr}"]
        # 1 本目だけ --force を通す（基底を当て直す指定のとき）
        if force and i == 0:
            cmd.append("--force")
        if max_demos:
            cmd += ["--max-demos", str(max_demos)]
        print(f"  $ {' '.join(cmd)}")
        rc |= subprocess.call(cmd, cwd=ROOT)
        if rc:
            print(f"  ★{suite} の前処理が失敗した（rc={rc}）")
            return rc
        if i == 0:
            miss = [b for b in BASIS_FILES
                    if not os.path.exists(os.path.join(ROOT, cr, b))]
            if miss:
                print(f"  ★基底ファイルが出来ていない: {miss}")
                return 1
            print(f"  基底を {os.path.join(cr)} に作った（以降のスイートはこれを使う）")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="状況を見るだけ")
    ap.add_argument("--download", action="store_true", help="LIBERO のデモを落とす")
    ap.add_argument("--build", default=None,
                    choices=list(PROFILES) + ["all"],
                    help="キャッシュを作る（main / spatial_dir / long / all）")
    ap.add_argument("--force", action="store_true",
                    help="★PCA 基底と正規化統計を当て直す（既存の数値と変わる）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-demos", type=int, default=0,
                    help="1 タスクあたりのデモ数を絞る（動作確認用。0 で全部）")
    a = ap.parse_args()

    if not (a.check or a.download or a.build):
        return check()
    rc = 0
    if a.check:
        rc |= check()
    if a.download:
        rc |= download()
    if a.build:
        names = (["main", "spatial_dir", "long"] if a.build == "all"
                 else [a.build])
        for n in names:
            rc |= build(n, a.force, a.device, a.max_demos)
        print("\n[prepare] 作り終わり。--check で確認する。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
