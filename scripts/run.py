#!/usr/bin/env python
"""PredVLA の実験を 1 本のコマンドで回す唯一の入口。

論文に載る 4 つの系列（本線 / ベースライン / はしご / アブレーション）を、
**渡す変数を変えるだけ**で学習から評価・集計まで通す。

    python scripts/run.py --track <トラック> [選択肢] --suite <スイート> \\
        --seeds <シード> --stage <段階>

トラック
    main       PredVLA 本線
    baseline   BC-LSTM / BC-Transformer（--model で選ぶ。既定は両方）
    ladder     はしご L0〜L7（--rung で選ぶ。既定は L0-L7 + 側枝 L2b など）
    ablation   表③のアブレーション A1〜A6（--id で選ぶ。既定は全部）
    all        上の 4 つを順に

段階（--stage）
    train      学習だけ
    eval       既存 ckpt の評価だけ
    aggregate  集計だけ
    all        学習 → 評価 → 集計（既定）

例

    # 何が走るか見るだけ（★最初に必ずこれを見る）
    python scripts/run.py --track all --suite libero_spatial --seeds 1-3 --dry-run

    # 同梱 ckpt で本線 4 スイートを評価して表にする（学習しない）
    python scripts/run.py --track main --suite all --seeds 1 --stage eval

    # ベースラインを 1 つ学習して評価
    python scripts/run.py --track baseline --model bc_lstm \\
        --suite libero_goal --seeds 0-6 --stage all

    # はしごを L0 から L7 まで（L2 以降は学習が要る）
    python scripts/run.py --track ladder --rung L0-L7 \\
        --suite libero_spatial --seeds 1-7 --stage all

    # アブレーションのうち再学習が要らないものだけ
    python scripts/run.py --track ablation --id A4,A5,A6 \\
        --suite libero_spatial --seeds 0-6 --stage eval

★このスクリプト自身はプロトコルを決めない。ロールアウト数・打ち切り・ER の設定・
  Long の「1 プロセス 1 タスク」は `predvla/protocol.py` にあり、評価は
  `scripts/evaluate.py` がそこから組み立てる。ここがやるのは
  「どの系列のどのシードを、どの順で、どのスクリプトに渡すか」だけである。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from predvla import ladder as L  # noqa: E402
from predvla import protocol as P  # noqa: E402

LOG_DIR = os.path.join(ROOT, "results", "logs")

# 「L0-L7」のような本線の範囲指定（側枝の id には - が入るので L<数字> に限る）
RE_RUNG_RANGE = re.compile(r"[Ll]\d+-[Ll]\d+")


# --------------------------------------------------------------------------
# 小道具
# --------------------------------------------------------------------------
def vpy() -> str:
    p = os.path.join(ROOT, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def parse_seeds(s: str) -> list[int]:
    """'1-7' / '1,3,5' / '0-2,7' を受ける。"""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_list(s: str, expand: dict[str, list[str]] | None = None) -> list[str]:
    """'A1,A2' / 'L0-L6' / 'all' をリストに開く。"""
    expand = expand or {}
    out: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part in expand:
            out += expand[part]
        else:
            out.append(part)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(title: str) -> None:
    say()
    say("=" * 78)
    say(f" {title}")
    say("=" * 78)


def run(cmd: list[str], dry: bool, log: str | None = None) -> int:
    """1 本のサブプロセスを回す。log を渡すとそこへ標準出力を落とす。"""
    shown = " ".join(cmd)
    say(f"    $ {shown}" + (f"  > {os.path.relpath(log, ROOT)}" if log else ""))
    if dry:
        return 0
    if log is None:
        return subprocess.call(cmd, cwd=ROOT)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.call(cmd, cwd=ROOT, stdout=fh,
                               stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)


def cell_done(tag: str) -> bool:
    """その評価セルが済んでいるか（ログに「全体」の行があるか）。"""
    p = os.path.join(LOG_DIR, f"ev_{tag}.log")
    if not os.path.exists(p):
        return False
    try:
        with open(p, errors="ignore") as f:
            return "全体" in f.read()
    except OSError:
        return False


def find_ckpt(series: str, suite: str, seed: int, steps: int,
              ckpt_dir: str) -> str:
    """配布 ckpt（平置き）と学習の出力ディレクトリの両方を探す。"""
    name = P.run_name(series, suite, seed)
    for p in (os.path.join(ckpt_dir, f"{name}.pt"),
              os.path.join(ROOT, "results", name, f"step_{steps}.pt"),
              os.path.join(ckpt_dir, name, f"step_{steps}.pt")):
        if os.path.exists(p):
            return p
    return ""


def parse_rungs(arg: str) -> list[str]:
    """はしごの段の指定を解く。

        all      本線 + 側枝
        main     本線だけ（L0 〜 L7）
        side     側枝だけ
        L0-L7    ★本線の範囲。番号を付け替えても壊れないよう、その場で
                 MAIN_LINE を切り出す（`L0-L6` のように固定で書かない）
        L2,L4    カンマ区切りの直接指定（側枝の id もここに書ける）
    """
    out: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            out += list(L.ALL)
        elif part == "main":
            out += list(L.MAIN_LINE)
        elif part == "side":
            out += list(L.SIDE)
        elif RE_RUNG_RANGE.fullmatch(part):
            a_, b_ = (L.resolve(x) for x in part.split("-", 1))
            if a_ not in L.MAIN_LINE or b_ not in L.MAIN_LINE:
                sys.exit(f"範囲で指定できるのは本線の段だけ: {part}"
                         f"（本線は {', '.join(L.MAIN_LINE)}）")
            i, j = L.MAIN_LINE.index(a_), L.MAIN_LINE.index(b_)
            if i > j:
                i, j = j, i
            out += L.MAIN_LINE[i:j + 1]
        else:
            out.append(part)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def suites_of(arg: str) -> list[str]:
    if arg == "all":
        return list(P.ALL_SUITES)
    if arg == "main":
        return list(P.SUITES_MAIN)
    return parse_list(arg)


# --------------------------------------------------------------------------
# トラック: 本線
# --------------------------------------------------------------------------
def track_main(a, suite: str, seeds: list[int]) -> int:
    rc = 0
    series = P.series_name("main")
    cfg = os.path.join(ROOT, "configs", f"predvla_{P.SHORT[suite]}.yaml")
    if not os.path.exists(cfg):
        say(f"  [欠品] config が無い: {os.path.relpath(cfg, ROOT)}")
        return 1

    if a.stage in ("train", "all"):
        for s in seeds:
            name = P.run_name(series, suite, s)
            rc |= run([vpy(), "-u", os.path.join(ROOT, "scripts", "train.py"),
                       "--config", cfg, "--device", a.device,
                       "--resume", "auto", "--set",
                       f"train.run_name={name}", f"train.seed={s}",
                       f"train.total_steps={a.steps}"], a.dry_run)

    if a.stage in ("eval", "all"):
        ckpts = [find_ckpt(series, suite, s, a.steps, a.ckpt_dir) for s in seeds]
        miss = [s for s, c in zip(seeds, ckpts) if not c]
        ckpts = [c for c in ckpts if c]
        if miss:
            say(f"  [欠品] ckpt が無いシード {miss}")
            say(f"          checkpoints/{series}_{P.SHORT[suite]}_s<S>.pt か "
                f"results/{series}_{P.SHORT[suite]}_s<S>/step_{a.steps}.pt を置く")
        if ckpts:
            rc |= eval_ckpts(a, suite, ckpts, protocol="main", tag_prefix="")
    return rc


# --------------------------------------------------------------------------
# トラック: ベースライン
# --------------------------------------------------------------------------
def track_baseline(a, suite: str, seeds: list[int], models: list[str]) -> int:
    rc = 0
    steps = P.BASELINE_STEPS
    for model in models:
        if model not in P.BASELINES:
            say(f"  [不明] ベースライン {model}（使えるのは {list(P.BASELINES)}）")
            rc |= 1
            continue
        head(f"ベースライン {P.BASELINES[model]}  /  {suite}")
        cfg = os.path.join(ROOT, "configs",
                           f"baseline_{model}_{P.SHORT[suite]}.yaml")
        if not os.path.exists(cfg):
            say(f"  [欠品] config が無い: {os.path.relpath(cfg, ROOT)}")
            say(f"          $ {os.path.basename(vpy())} tools/configs_from_ckpt.py "
                f"--baselines")
            rc |= 1
            continue

        if a.stage in ("train", "all"):
            for s in seeds:
                name = P.run_name(model, suite, s)
                rc |= run([vpy(), "-u", os.path.join(ROOT, "benchmark", "train.py"),
                           "--config", cfg, "--device", a.device, "--set",
                           f"train.run_name={name}", f"train.seed={s}",
                           f"train.total_steps={steps}"], a.dry_run)

        if a.stage in ("eval", "all"):
            for s in seeds:
                ck = find_ckpt(model, suite, s, steps, a.ckpt_dir)
                if not ck:
                    say(f"  [欠品] {P.run_name(model, suite, s)} の ckpt が無い")
                    continue
                rc |= eval_baseline_ckpt(a, suite, ck,
                                         P.run_name(model, suite, s))
    return rc


def eval_baseline_ckpt(a, suite: str, ckpt: str, tag: str) -> int:
    """ベースライン 1 本を閉ループ評価する。

    ★本線と同じロールアウト数・打ち切り・Long の 1 プロセス 1 タスクを守る。
      `benchmark/eval_suite.py` は `--max-steps-auto` を持たないので、Long では
      タスクごとの打ち切りを `protocol.MAX_STEPS_LONG_PER_TASK` から直接渡す。
    ★出力は本線と同じ `results/logs/ev_<tag>.log` に落とす。`eval_suite.py` は
      集計器が読む「task N success S/T」と「全体」の行を出すようにしてある。
    """
    rc = 0
    is_long = suite == P.SUITE_LONG
    base = [vpy(), "-u", os.path.join(ROOT, "benchmark", "eval_suite.py"),
            "--ckpt", ckpt, "--suite", suite, "--device", a.eval_device]
    if is_long:
        n = P.ROLLOUTS_LONG
        for t in range(10):
            cell = f"{tag}_t{t}"
            if cell_done(cell) and not a.redo:
                say(f"    [済] {cell}")
                continue
            rc |= run(base + ["--n", str(n), "--tasks", str(t),
                              "--max-steps", str(P.MAX_STEPS_LONG_PER_TASK[t]),
                              "--tag", cell],
                      a.dry_run, os.path.join(LOG_DIR, f"ev_{cell}.log"))
    else:
        if cell_done(tag) and not a.redo:
            say(f"    [済] {tag}")
            return 0
        rc |= run(base + ["--n", str(P.ROLLOUTS_MAIN),
                          "--max-steps", str(P.MAX_STEPS_MAIN),
                          "--tag", tag],
                  a.dry_run, os.path.join(LOG_DIR, f"ev_{tag}.log"))
    return rc


# --------------------------------------------------------------------------
# トラック: はしご
# --------------------------------------------------------------------------
def track_ladder(a, suite: str, seeds: list[int], rungs: list[str]) -> int:
    """L6 だけベースライン側で回し、残りは scripts/run_ladder.py に渡す。"""
    rc = 0
    rungs = [L.resolve(r) for r in rungs]
    ladder_rungs = [r for r in rungs if L.RUNGS[r]["kind"] != "baseline"]
    has_l6 = any(L.RUNGS[r]["kind"] == "baseline" for r in rungs)

    if ladder_rungs:
        cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "run_ladder.py"),
               "--rung", ",".join(ladder_rungs), "--suite", suite,
               "--seeds", ",".join(str(s) for s in seeds),
               "--mode", {"train": "train", "eval": "rollout",
                          "all": "both"}.get(a.stage, "both"),
               "--ckpt-dir", a.ckpt_dir, "--steps", str(a.steps),
               "--device", a.device, "--eval-device", a.eval_device]
        if a.jobs:
            cmd += ["--jobs", str(a.jobs)]
        if a.dry_run:
            cmd.append("--dry-run")
        rc |= run(cmd, False)

    if has_l6:
        head(f"はしご L6（独立実装の BC-LSTM）  /  {suite}")
        say("  ★L6 は同じコードベースの再学習ではなく、別実装のベースラインである。")
        rc |= track_baseline(a, suite, seeds, ["bc_lstm"])
    return rc


# --------------------------------------------------------------------------
# トラック: アブレーション
# --------------------------------------------------------------------------
def track_ablation(a, suite: str, seeds: list[int], ids: list[str]) -> int:
    ids = [P.resolve_ablation(x) for x in ids]
    cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "run_ablation.py"),
           "--ablation", ",".join(ids), "--suite", suite,
           "--seeds", ",".join(str(s) for s in seeds),
           "--mode", {"train": "train", "eval": "rollout",
                      "all": "both"}.get(a.stage, "both"),
           "--ckpt-dir", a.ckpt_dir, "--device", a.device,
           "--eval-device", a.eval_device, "--protocol", "main"]
    if a.jobs:
        cmd += ["--jobs", str(a.jobs)]
    if a.dry_run:
        cmd.append("--dry-run")
    return run(cmd, False)


# --------------------------------------------------------------------------
# 評価の共通経路（本線・はしご・アブレーションは evaluate.py を通る）
# --------------------------------------------------------------------------
def eval_ckpts(a, suite: str, ckpts: list[str], protocol: str,
               tag_prefix: str) -> int:
    cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "evaluate.py"),
           "--suite", suite, "--protocol", protocol,
           "--device", a.eval_device]
    if tag_prefix:
        cmd += ["--tag-prefix", tag_prefix]
    if a.jobs:
        cmd += ["--jobs", str(a.jobs)]
    if a.redo:
        cmd.append("--redo")
    if a.dry_run:
        cmd.append("--dry-run")
    cmd += ["--ckpt"] + ckpts
    return run(cmd, False)


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------
def aggregate(a, matches: list[str]) -> int:
    rc = 0
    for m in matches:
        head(f"集計  --match {m}")
        cmd = [vpy(), "-u", os.path.join(ROOT, "scripts", "aggregate.py"),
               "--match", m]
        rc |= run(cmd, a.dry_run)
    return rc


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--track", default="main",
                    choices=["main", "baseline", "ladder", "ablation", "all"],
                    help="回す系列（既定 main）")
    ap.add_argument("--suite", default="libero_spatial",
                    help="スイート名 / all（4 つ）/ main（Long 以外の 3 つ）/ "
                         "カンマ区切り")
    ap.add_argument("--seeds", default="1",
                    help="例 1-14 / 0-6 / 1,3,5（既定 1）")
    ap.add_argument("--stage", default="all",
                    choices=["train", "eval", "aggregate", "all"],
                    help="どこまでやるか（既定 all）")
    # トラック別の選択肢
    ap.add_argument("--model", default="all",
                    help="baseline: bc_lstm / bc_transformer / all（既定）")
    ap.add_argument("--rung", default="all",
                    help="ladder: all（本線 + 側枝）/ main（本線 L0〜L7）/ "
                         "side（側枝だけ）/ 範囲（L0-L7, L2-L5）/ "
                         "カンマ区切り（例 L2,L4,L2b）")
    ap.add_argument("--id", default="all",
                    help="ablation: A1〜A6 / all（既定）/ train（A1〜A3）/ "
                         "test（A4〜A6）/ カンマ区切り")
    # 実行の条件
    ap.add_argument("--device", default="auto", help="学習の device（既定 auto）")
    ap.add_argument("--eval-device", default="cpu",
                    help="評価の device（既定 cpu。B=16 は CPU の方が速い）")
    ap.add_argument("--jobs", type=int, default=0,
                    help="評価の並列本数（0 でコア数と空き RAM から自動）")
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"),
                    help="配布 ckpt の置き場（既定 checkpoints/）")
    ap.add_argument("--steps", type=int, default=P.TRAIN_STEPS,
                    help=f"本線・はしご・アブレーションの学習 step 数"
                         f"（既定 {P.TRAIN_STEPS}。ベースラインは常に "
                         f"{P.BASELINE_STEPS}）")
    ap.add_argument("--redo", action="store_true",
                    help="完了済みの評価セルもやり直す")
    ap.add_argument("--dry-run", action="store_true",
                    help="★何が走るかだけ出す。まずこれを見る")
    a = ap.parse_args()

    suites = suites_of(a.suite)
    bad = [s for s in suites if s not in P.ALL_SUITES]
    if bad:
        sys.exit(f"未知のスイート {bad}。使えるのは {list(P.ALL_SUITES)} / all / main")
    seeds = parse_seeds(a.seeds)
    tracks = (["main", "baseline", "ladder", "ablation"]
              if a.track == "all" else [a.track])

    models = (list(P.BASELINES) if a.model == "all"
              else parse_list(a.model))
    rungs = parse_rungs(a.rung)
    abl_ids = parse_list(a.id, expand={
        "all": list(P.ABLATIONS),
        "train": [k for k, v in P.ABLATIONS.items() if v["kind"] == "train"],
        "test": [k for k, v in P.ABLATIONS.items() if v["kind"] == "test"]})

    head("PredVLA 実験ランナー")
    say(f"  トラック  {', '.join(tracks)}")
    say(f"  スイート  {', '.join(suites)}")
    say(f"  シード    {seeds}")
    say(f"  段階      {a.stage}" + ("   ★dry-run" if a.dry_run else ""))
    if "baseline" in tracks:
        say(f"  ベースライン {', '.join(P.BASELINES[m] for m in models if m in P.BASELINES)}"
            f"（{P.BASELINE_STEPS:,} step）")
    if "ladder" in tracks:
        say(f"  はしごの段  {', '.join(rungs)}")
    if "ablation" in tracks:
        names = ", ".join(P.resolve_ablation(i) for i in abl_ids)
        say(f"  アブレーション {names}")
    if len(seeds) < 7 and a.stage != "train":
        say("  ★シードが 7 本未満。σ≈4〜11pt あるので、確定を主張するなら 7 本以上")

    # そのトラック・スイートの結果を aggregate.py に拾わせる --match の文字列。
    #   ★ログのタグと同じ規則（protocol.run_name）から作る。ここを直に書くと
    #     系列名を変えたときに集計だけ黙って空になる。
    def matches_for(tr: str, suite: str) -> list[str]:
        if tr == "main":
            return [f"{P.series_name('main')}_{P.SHORT[suite]}"]
        if tr == "baseline":
            return [f"{m}_{P.SHORT[suite]}" for m in models if m in P.BASELINES]
        if tr == "ladder":
            return ["ladder_"]
        if tr == "ablation":
            return [f"{P.resolve_ablation(i)}__" for i in abl_ids]
        return []

    rc = 0
    if a.stage != "aggregate":
        for suite in suites:
            for tr in tracks:
                head(f"{tr}  /  {suite}")
                if tr == "main":
                    rc |= track_main(a, suite, seeds)
                elif tr == "baseline":
                    rc |= track_baseline(a, suite, seeds, models)
                elif tr == "ladder":
                    rc |= track_ladder(a, suite, seeds, rungs)
                elif tr == "ablation":
                    rc |= track_ablation(a, suite, seeds, abl_ids)

    if a.stage in ("aggregate", "all", "eval"):
        seen, uniq = set(), []
        for suite in suites:
            for tr in tracks:
                for m in matches_for(tr, suite):
                    if m not in seen:
                        seen.add(m)
                        uniq.append(m)
        rc |= aggregate(a, uniq)

    say()
    say("終了コード " + str(rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
