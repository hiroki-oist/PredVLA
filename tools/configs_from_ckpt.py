#!/usr/bin/env python
"""公開 ckpt に入っている cfg から yaml を機械的に書き出す。

論文の run の一部は yaml を書かず `--set` の羅列で組まれていた。手で書き写すと
必ずどこか間違うので、ckpt の中の cfg をそのまま吐かせる。

    # 1 つ見る
    python tools/configs_from_ckpt.py --ckpt <ckpt> --show

    # ベースラインの config をまとめて作る（既定は同梱 ckpt から）
    python tools/configs_from_ckpt.py --baselines

★書き出した yaml の paths.cache_root は本リポジトリの階層に直す（先頭の ../ を剥がす）。
★train.run_name は消す（使うときに --set で与えるもの）。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (出力する config 名, ckpt の名前)
#   既定では checkpoints/ に同梱してある ckpt から復元する。
#   --runs-dir を渡せば別の置き場（<名前>/step_*.pt という形）からも引ける。
BASELINES = [
    ("baseline_bc_lstm_spatial",        "bc_lstm_spatial_s0"),
    ("baseline_bc_lstm_goal",           "bc_lstm_goal_s0"),
    ("baseline_bc_lstm_object",         "bc_lstm_object_s0"),
    ("baseline_bc_lstm_long",           "bc_lstm_long_s0"),
    ("baseline_bc_transformer_spatial", "bc_transformer_spatial_s0"),
    ("baseline_bc_transformer_goal",    "bc_transformer_goal_s0"),
    ("baseline_bc_transformer_object",  "bc_transformer_object_s0"),
    ("baseline_bc_transformer_long",    "bc_transformer_long_s0"),
]

HEADER = """# ★このファイルは tools/configs_from_ckpt.py が公開 ckpt の cfg から生成した。
#   手で編集しない。元 ckpt: {run}（{ck}）
#
# 論文のベースラインは yaml を書かず --set の羅列で組まれていたので、
# ckpt に保存された cfg をそのまま書き出すのが唯一確実な復元手段である。
# ★手を入れているのは 3 か所だけ: paths.cache_root の先頭の "../" を剥がす /
#   device を auto にする / model.type を現在の呼び名に直す。
# 使い方:
#   python benchmark/train.py --config configs/{name}.yaml \\
#       --set train.run_name=<名前> train.seed=<S>
"""


def load_cfg(ck_path: str) -> dict:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or ck.get("config")
    if cfg is None:
        raise SystemExit(f"[cfg] {ck_path} に cfg が無い（キー: {list(ck.keys())[:8]}）")
    return cfg


def normalize(cfg: dict) -> dict:
    cfg = dict(cfg)
    p = dict(cfg.get("paths", {}))
    cr = p.get("cache_root", "")
    while isinstance(cr, str) and cr.startswith("../"):
        cr = cr[3:]
    if cr:
        p["cache_root"] = cr
    if p:
        cfg["paths"] = p
    tr = dict(cfg.get("train", {}))
    tr.pop("run_name", None)          # 使うときに --set で与える
    if tr:
        cfg["train"] = tr
    # ★元 ckpt は Mac で組まれたものがあり device: mps が焼き込まれている。
    #   再現環境では自動判定にする（cuda > mps > cpu）。
    cfg["device"] = "auto"
    # ★model.type だけ現在の呼び名に直す（bcrnn -> bc_lstm、bctf -> bc_transformer）。
    #   中身は変えない。src/models/registry.py は旧名も受けるので、この書き換えが
    #   無くても動くが、生成した config が新旧混在の表記になるのを避ける。
    m = dict(cfg.get("model", {}))
    if "type" in m:
        from src.models import registry
        m["type"] = registry.normalize(m["type"])
        cfg["model"] = m
    return cfg


def find_ckpt(runs_dir: str, run: str) -> str:
    """平置きの <run>.pt と、<run>/step_*.pt の両方を探す。"""
    flat = os.path.join(runs_dir, f"{run}.pt")
    if os.path.exists(flat):
        return flat
    d = os.path.join(runs_dir, run)
    if not os.path.isdir(d):
        return ""
    cks = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    return os.path.join(d, cks[-1]) if cks else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--show", action="store_true", help="標準出力に出すだけ")
    ap.add_argument("--baselines", action="store_true",
                    help="ベースラインの config をまとめて作る")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "checkpoints"),
                    help="ckpt の置き場（既定 checkpoints/）")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "configs"))
    a = ap.parse_args()

    if a.ckpt:
        cfg = normalize(load_cfg(a.ckpt))
        txt = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        if a.show:
            print(txt)
        else:
            print(txt)
        return 0

    if not a.baselines:
        ap.print_help()
        return 2

    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for name, run in BASELINES:
        ck = find_ckpt(a.runs_dir, run)
        if not ck:
            print(f"  [欠品] {run} が {a.runs_dir} に無い → {name}.yaml は作らない")
            print(f"          $ python tools/install_checkpoints.py --list")
            continue
        cfg = normalize(load_cfg(ck))
        out = os.path.join(a.out_dir, f"{name}.yaml")
        with open(out, "w") as f:
            f.write(HEADER.format(
                run=run,
                ck=(os.path.relpath(ck, ROOT) if ck.startswith(ROOT) else ck),
                name=name))
            f.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        mt = cfg.get("model", {}).get("type", "?")
        print(f"  書いた {os.path.relpath(out, ROOT)}  "
              f"(type={mt}, cache_root={cfg.get('paths', {}).get('cache_root')}, "
              f"total_steps={cfg.get('train', {}).get('total_steps')})")
        n += 1
    print(f"[cfg] {n} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
