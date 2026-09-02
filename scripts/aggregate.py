#!/usr/bin/env python
"""results/logs/ の評価ログを読んで表にする。

    # スイートの全系列を一覧
    python scripts/aggregate.py

    # 系列を絞って、タスク別も出す
    python scripts/aggregate.py --match 'predvla_spatial' --per-task

    # 論文の参考値と突き合わせる
    python scripts/aggregate.py --match 'predvla_spatial' --reference libero_spatial

    # 2 つの系列を対応 t 検定で比べる（同じシードどうしを引き算する）
    python scripts/aggregate.py --compare predvla_spatial A2_no_pb_spatial

★シードは平均だけでなく必ず ±SD と n を出す。この系列はシード分散が σ≈6〜11pt あり、
  n=3 の結論は n=7 でしばしば符号が反転する（実測で 4 例）。
★Long は 1 シードが 10 個のログ（_t0.._t9）に分かれる。10 個揃っていないシードは
  「部分」と印を付けて平均から外す。部分値は成功エピソードが先に終わるため
  上向きに偏るので、混ぜてはいけない。
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import protocol as P  # noqa: E402

LOG_DIR = os.path.join(ROOT, "results", "logs")
RE_TOTAL = re.compile(r"全体\s+([0-9.]+)%\s+\((\d+)/(\d+)\)")
RE_TASK = re.compile(r"task\s+(\d+)\s+success\s+(\d+)/(\d+)")
RE_TAG = re.compile(r"^ev_(?P<tag>.+)\.log$")
RE_SEED = re.compile(r"^(?P<name>.+?)_s(?P<seed>\d+)(?:_t(?P<task>\d+))?$")


def read_log(path: str):
    """(全体%, 成功数, 試行数, {task: (成功, 試行)}) を返す。未完了なら None。"""
    try:
        with open(path, errors="ignore") as f:
            txt = f.read()
    except OSError:
        return None
    m = RE_TOTAL.search(txt)
    if not m:
        return None
    per = {int(a): (int(b), int(c)) for a, b, c in RE_TASK.findall(txt)}
    return float(m.group(1)), int(m.group(2)), int(m.group(3)), per


def collect(match: str | None):
    """ログを走査して {系列名: {シード: 集計}} を作る。"""
    if not os.path.isdir(LOG_DIR):
        return {}, []
    # 系列 -> シード -> {"succ":, "n":, "cells": set(task), "per": {task:(s,n)}}
    series: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(
        lambda: dict(succ=0, n=0, cells=set(), per={})))
    broken: list[str] = []
    for fn in sorted(os.listdir(LOG_DIR)):
        m = RE_TAG.match(fn)
        if not m:
            continue
        tag = m.group("tag")
        if match and match not in tag:
            continue
        r = read_log(os.path.join(LOG_DIR, fn))
        if r is None:
            broken.append(tag)
            continue
        _, s, n, per = r
        ms = RE_SEED.match(tag)
        if not ms:
            name, seed, task = tag, -1, None
        else:
            name = ms.group("name")
            seed = int(ms.group("seed"))
            task = ms.group("task")
        d = series[name][seed]
        d["succ"] += s
        d["n"] += n
        d["per"].update(per)
        d["cells"].add(int(task) if task is not None else -1)
    return series, broken


def stats(vals: list[float]) -> tuple[float, float, int]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, sd, n


def seed_rates(name: str, sd: dict, expect_cells: int) -> dict[int, float]:
    """シード -> 成功率。★セルが揃っていないシードは外す。"""
    out: dict[int, float] = {}
    for seed, d in sorted(sd.items()):
        if len(d["cells"]) < expect_cells:
            continue
        if d["n"] == 0:
            continue
        out[seed] = 100.0 * d["succ"] / d["n"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=None, help="タグに含まれる文字列で絞る")
    ap.add_argument("--suite", default=None, choices=list(P.ALL_SUITES),
                    help="Long かどうかでセル数の期待値が変わる。指定が無ければ"
                         "タグに _t0 があるかで自動判定する")
    ap.add_argument("--per-task", action="store_true", help="タスク別も出す")
    ap.add_argument("--reference", default=None, choices=list(P.ALL_SUITES),
                    help="論文の参考値と突き合わせる")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"),
                    help="2 系列を対応 t 検定で比べる")
    a = ap.parse_args()

    series, broken = collect(a.match)
    if not series:
        print(f"[aggregate] {LOG_DIR} に該当するログが無い")
        return 1

    print("=" * 78)
    print(f" 評価ログの集計   {LOG_DIR}")
    print("=" * 78)
    if broken:
        print(f"\n★未完了（「全体」行が無い）{len(broken)} 件: "
              + ", ".join(broken[:6]) + (" ..." if len(broken) > 6 else ""))

    rows = []
    for name in sorted(series):
        sd = series[name]
        # Long は 1 シード 10 セル、3 スイートは 1 セル
        is_long = any(c >= 0 for d in sd.values() for c in d["cells"])
        expect = 10 if is_long else 1
        rates = seed_rates(name, sd, expect)
        partial = [s for s, d in sd.items() if len(d["cells"]) < expect]
        m, s, n = stats(list(rates.values()))
        rows.append((name, m, s, n, rates, partial, is_long, sd))

    print(f"\n{'系列':46s} {'平均':>7s} {'±SD':>7s} {'n':>3s}  シード別")
    print("-" * 78)
    for name, m, s, n, rates, partial, is_long, _ in rows:
        if n == 0:
            print(f"{name:46s} {'—':>7s} {'—':>7s} {0:3d}  "
                  f"（揃ったシード無し。部分 {sorted(partial)}）")
            continue
        per = " ".join(f"s{k}:{v:.0f}" for k, v in sorted(rates.items()))
        sds = f"{s:7.2f}" if n >= 2 else f"{'—':>7s}"
        note = ""
        if n < 3:
            note = "  ★暫定(n<3)"
        elif n < 7:
            note = "  ★候補(n<7。この系列は n=3→7 で符号が反転した実例が 4 件ある)"
        if partial:
            note += f"  部分:{sorted(partial)}"
        print(f"{name:46s} {m:7.2f} {sds} {n:3d}  {per}{note}")

    if a.per_task:
        print("\n--- タスク別 ---")
        for name, m, s, n, rates, partial, is_long, sd in rows:
            agg = defaultdict(lambda: [0, 0])
            for seed, d in sd.items():
                if seed not in rates:
                    continue
                for t, (sc, nn) in d["per"].items():
                    agg[t][0] += sc
                    agg[t][1] += nn
            if not agg:
                continue
            print(f"\n  [{name}]  n={n} シードを合算")
            for t in sorted(agg):
                sc, nn = agg[t]
                print(f"    task {t:2d}  {100.0*sc/max(nn,1):6.2f}%  ({sc}/{nn})")

    if a.reference:
        suite = a.reference
        m, sd, n = P.MAIN_TABLE[suite]
        print(f"\n--- 論文の参考値との突き合わせ（{suite}）---")
        print(f"  表① 主表（確定プロトコル MAIN）      {m:6.2f} ± {sd:.2f}  (n={n})")
        print(f"  旧看板 SGD10                        "
              f"{P.SGD10_TABLE[suite]:6.2f}   ★プロトコルが違うので混ぜない")
        print(f"\n  アブレーション（{suite}）:")
        for abl, per in P.ABLATION_TABLE.items():
            v = per.get(suite)
            if v is None:
                old = P.ABLATION_ERW09_NOT_FOR_CITATION.get(abl, {}).get(suite)
                extra = (f"   （er_w=0.9 の旧値 {old:.2f}。★引用禁止）"
                         if old is not None else "")
                print(f"    {abl:20s} 確定プロトコルでは未取得{extra}")
            else:
                print(f"    {abl:20s} {v[0]:6.2f} ± {v[1]:.2f}  (n={v[2]})")
        print("\n  ※ 表の平均がこれらに ±2pt 程度で乗れば再現できている"
              "（ロールアウトのラン間ノイズが ±1〜4.5pt ある）")
        print("  ※ シード分散は σ≈4〜6pt（3 スイート）/ 6.4pt（Long）。"
              "n を揃えずに平均だけ比べてはいけない")

    if a.compare:
        A, B = a.compare
        ra = next((r for r in rows if r[0] == A), None)
        rb = next((r for r in rows if r[0] == B), None)
        if not ra or not rb:
            print(f"\n★--compare の系列が見つからない: "
                  f"{A if not ra else ''} {B if not rb else ''}")
            return 1
        common = sorted(set(ra[4]) & set(rb[4]))
        print(f"\n--- 対応比較 {A} − {B} ---")
        if len(common) < 2:
            print(f"  共通シードが {len(common)} 個しかないので検定できない")
            return 0
        d = [ra[4][s] - rb[4][s] for s in common]
        m, sd_, n = stats(d)
        se = sd_ / math.sqrt(n)
        t = m / se if se > 0 else float("inf")
        print(f"  共通シード {common}")
        print(f"  差 " + " ".join(f"s{s}:{ra[4][s]-rb[4][s]:+.0f}" for s in common))
        print(f"  Δ = {m:+.2f} ± {sd_:.2f}  (n={n}, SE={se:.2f}, t={t:+.2f})")
        print(f"  ★n={n} の Δ は参考値。この系列は n=3 の ±5pt が n=7 で消える")
    return 0


if __name__ == "__main__":
    sys.exit(main())
