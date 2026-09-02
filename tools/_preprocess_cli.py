#!/usr/bin/env python
"""凍結前段のキャッシュを作る CLI（src/data/preprocess.run の薄い皮）。

普段は tools/prepare_data.py 経由で呼ぶ。直に叩くのは 1 スイートだけ作り直したいとき。

    python tools/_preprocess_cli.py --config configs/preprocess.yaml \
        --suite libero_spatial --set paths.cache_root=data/cache_l64

★--force を付けると PCA 基底と q の正規化統計を当て直す。既存キャッシュとの
  互換が切れる（学習済み ckpt がその基底に依存している）ので、通常は付けない。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils import compat            # noqa: E402  MUJOCO_GL などを先に決める

compat.apply()

from src.utils.config import load_config, apply_overrides  # noqa: E402
from src.data import preprocess          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=os.path.join(ROOT, "configs", "preprocess.yaml"))
    ap.add_argument("--suite", default=None)
    ap.add_argument("--max-demos", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true",
                    help="★PCA 基底と正規化統計を当て直す")
    ap.add_argument("--set", nargs="*", default=[], help="key.sub=val")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg = apply_overrides(cfg, a.set)
    if a.suite:
        cfg["preprocess"]["suite"] = a.suite
        cfg.setdefault("data", {})["suite"] = a.suite
    preprocess.run(cfg, force=a.force, max_demos=a.max_demos, device=a.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
