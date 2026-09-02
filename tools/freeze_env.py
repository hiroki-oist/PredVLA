#!/usr/bin/env python
"""requirements.lock.txt を「直接依存からの推移閉包」だけで生成する。

`pip freeze` をそのまま使うと、この機の venv には ROS2 の system site-packages が
漏れて入ってくるので、無関係なパッケージが 100 件以上混ざる（実測 284 件中）。
そこで pyproject.toml の直接依存を根として、インストール済みメタデータの
Requires-Dist を辿った閉包だけを書き出す。

    python tools/freeze_env.py            # requirements.lock.txt を更新
    python tools/freeze_env.py --stdout   # 標準出力に出すだけ

★extra（[test] など）は辿らない。環境マーカ（; sys_platform == ...）は現在の
  環境で評価して、成り立つものだけ辿る。
"""
from __future__ import annotations

import argparse
import os
import sys

import importlib.metadata as md
from packaging.requirements import Requirement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# pyproject.toml を読まずに済むよう根をここに書く（pyproject と一致させること）。
ROOTS = [
    "torch", "torchvision", "numpy", "h5py", "PyYAML", "scikit-learn",
    "opencv-python", "pillow", "sentence-transformers", "transformers",
    "robosuite", "robomimic", "mujoco", "bddl", "easydict", "einops",
    "imageio", "packaging",
]
# third_party から editable で入れるので lock には出さない
EXCLUDE = {"libero"}


def norm(n: str) -> str:
    return n.lower().replace("_", "-").replace(".", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()

    installed: dict[str, md.Distribution] = {}
    for d in md.distributions():
        n = d.metadata.get("Name")
        if n:
            installed.setdefault(norm(n), d)

    seen: set[str] = set()
    missing: list[str] = []
    stack = [norm(r) for r in ROOTS]
    while stack:
        key = stack.pop()
        if key in seen or key in {norm(x) for x in EXCLUDE}:
            continue
        d = installed.get(key)
        if d is None:
            missing.append(key)
            continue
        seen.add(key)
        for spec in (d.requires or []):
            try:
                req = Requirement(spec)
            except Exception:
                continue
            if req.marker is not None:
                # extra を要求するマーカ（[test] 等）は辿らない
                if "extra" in str(req.marker):
                    continue
                try:
                    if not req.marker.evaluate():
                        continue
                except Exception:
                    continue
            stack.append(norm(req.name))

    lines = [
        "# PredVLA 再現環境の凍結。★手で編集しない（tools/freeze_env.py が生成する）。",
        "# 生成元: 2026-08-22 に論文の数値を出している実機の venv。",
        f"#   Python {sys.version.split()[0]}",
        "# 直接依存（tools/freeze_env.py の ROOTS）からの推移閉包のみ。",
        "# pip freeze をそのまま使うと ROS2 の system site-packages が混ざるので使わない。",
        "#",
        "# ★torch / torchvision は CUDA ビルドなので setup.sh が --index-url 付きで先に入れる。",
        "#   CPU だけの機では setup.sh が CPU ビルドの index を選ぶ。",
        "# ★libero は third_party/LIBERO から editable で入れる（ここには出さない）。",
    ]
    for key in sorted(seen):
        d = installed[key]
        lines.append(f"{d.metadata['Name']}=={d.version}")
    out = "\n".join(lines) + "\n"

    if missing:
        sys.stderr.write("[freeze] ★未インストールの根/依存: "
                         + ", ".join(sorted(set(missing))) + "\n")
    if a.stdout:
        sys.stdout.write(out)
    else:
        p = os.path.join(ROOT, "requirements.lock.txt")
        with open(p, "w") as f:
            f.write(out)
        print(f"[freeze] {p} に {len(seen)} 件を書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
