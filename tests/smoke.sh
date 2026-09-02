#!/usr/bin/env bash
# clone 直後の確認。学習・評価・集計が通るところまでを 10 分程度で見る。
#
#   bash tests/smoke.sh            CPU で回す（どの環境でも通るはず）
#   bash tests/smoke.sh --cuda     device を auto にして GPU も使う
#
# ★これは「動くか」の確認であって、論文の数値の再現ではない。
#   ロールアウト数もタスク数も絞ってあるので、出た成功率を論文と比べてはいけない。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

# ---------------------------------------------------------------------------
# ★排他ロック。このスクリプトは固定パスを使い回す:
#     results/smoke_train_s1/        学習の出力（毎回消してから作る）
#     results/logs/ev_smoke_*.log    評価ログ（毎回消してから作る）
#     /tmp/smoke_*.log               各段の生ログ
#   2 本同時に走らせると、後から始まった方の rm が先に走っている方の出力を消す。
#   消されても評価プロセス自体は rc=0 で終わるので、
#   **「全体」行が無いという分かりにくい FAIL** になって原因を追いにくい
#   （2026-09-01 に実際に踏んだ）。並行実行はここで止める。
# ---------------------------------------------------------------------------
LOCK="$ROOT/results/.smoke.lock"
mkdir -p "$ROOT/results"
exec 9>"$LOCK" || { echo "ロックファイルを作れない: $LOCK"; exit 1; }
if ! flock -n 9; then
  echo "★このリポジトリで tests/smoke.sh が既に走っている（ロック $LOCK）。"
  echo "  固定パスを共有するので同時には走らせられない。"
  echo "  先に走っている方の終了を待つか、そちらを止めてから再実行する:"
  echo "      pgrep -af 'tests/smoke.sh'"
  exit 2
fi

DEV=cpu
[ "${1:-}" = "--cuda" ] && DEV=auto

PASS=0; FAIL=0
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[32m  PASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
ng()   { printf '\033[31m  FAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }

step "1/7 健診（シミュレータの起動確認は飛ばす）"
if "$PY" tools/doctor.py --quick > /tmp/smoke_doctor.log 2>&1; then
  ok "doctor.py 致命 0 件"
else
  ng "doctor.py に致命的な項目がある → tail /tmp/smoke_doctor.log"
  grep '★致命' /tmp/smoke_doctor.log | head
fi

step "2/7 同梱 ckpt が読めるか（重みを実際に load_state_dict する）"
if "$PY" - > /tmp/smoke_ckpt.log 2>&1 <<'EOF'
import glob, os, sys, torch
sys.path.insert(0, ".")
from src.utils import compat; compat.apply()
from predvla.model import PredVLA
from src.models import registry
paths = sorted(glob.glob("checkpoints/*.pt"))
assert paths, "checkpoints/ に ckpt が 1 本も無い"
for p in paths:
    name = os.path.basename(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = (registry.build(ck["cfg"]) if name.startswith(("bc_lstm", "bc_transformer"))
         else PredVLA(ck["cfg"]))
    m.load_state_dict(ck["model"])
    print(f"  {name:34s} {sum(x.numel() for x in m.parameters()):>9,} params")
print(f"OK {len(paths)} 本")
EOF
then
  ok "$(tail -1 /tmp/smoke_ckpt.log)"
else
  ng "同梱 ckpt が読めない → tail /tmp/smoke_ckpt.log"
  tail -10 /tmp/smoke_ckpt.log
fi

step "3/7 入口スクリプトが全トラックの計画を出せるか（--dry-run）"
if "$PY" scripts/run.py --track all --suite libero_spatial --seeds 1 \
     --dry-run > /tmp/smoke_plan.log 2>&1; then
  ok "run.py --track all --dry-run が通った"
else
  ng "run.py が失敗 → tail /tmp/smoke_plan.log"
  tail -20 /tmp/smoke_plan.log
fi

step "4/7 import と パラメータ数"
"$PY" - <<'EOF'
import sys, yaml
sys.path.insert(0, ".")
from predvla.model import PredVLA
from predvla import protocol as P
cfg = yaml.safe_load(open("configs/predvla_spatial.yaml"))
m = PredVLA(cfg)
n = m.num_params()
print(f"  パラメータ数 {n:,}")
assert 6.0e5 < n < 7.5e5, f"想定外のパラメータ数 {n}"
print(f"  確定プロトコル {P.MAIN}")
print("  OK")
EOF
[ $? -eq 0 ] && ok "モデルを組めてパラメータ数が想定内" || ng "モデルの構築に失敗"

step "5/7 学習 20 step（1 タスク・8 デモ・T=64、device=$DEV）"
rm -rf results/smoke_train_s1
rm -f results/logs/ev_smoke_*.log     # 前回のスモークのログを残すと次で拾ってしまう
if "$PY" -u scripts/train.py --config configs/predvla_spatial.yaml --device "$DEV" \
     --set train.run_name=smoke_train_s1 train.total_steps=20 train.ckpt_every=20 \
           train.log_every=10 train.compile=false train.seed=1 \
           data.tasks='[0]' data.max_demos=8 data.T=64 train.batch_size=4 \
     > /tmp/smoke_train.log 2>&1 && [ -f results/smoke_train_s1/step_20.pt ]; then
  ok "step_20.pt ができた（$(grep -c '^step' /tmp/smoke_train.log) 本のログ行）"
  grep -E '^step +20' /tmp/smoke_train.log | tail -1 | sed 's/^/      /'
else
  ng "学習が失敗 → tail /tmp/smoke_train.log"
  tail -20 /tmp/smoke_train.log
fi

step "6/7 閉ループ評価（1 タスク × 2 ロールアウト。★論文とは比較不可）"
rm -f results/logs/ev_smoke_eval.log
if "$PY" -u scripts/evaluate.py --suite libero_spatial \
     --ckpt results/smoke_train_s1/step_20.pt --tasks 0 --n 2 --jobs 1 \
     --tag-prefix smoke_ > /tmp/smoke_eval.log 2>&1; then
  # ★タグはこの 1 本に決まる（--tag-prefix + ckpt の run 名）。glob で拾うと
  #   前回のスモークのログを取ってしまう（実際に踏んだ）。
  L=results/logs/ev_smoke_smoke_train_s1.log
  if [ -f "$L" ] && grep -q 全体 "$L"; then
    ok "評価が完走した: $(grep 全体 "$L" | tail -1 | tr -s ' ')"
  else
    ng "評価ログに「全体」が無い → $L"
    tail -15 "$L" 2>/dev/null
  fi
else
  ng "評価が失敗 → tail /tmp/smoke_eval.log"
  tail -25 /tmp/smoke_eval.log
fi

step "7/7 集計"
if "$PY" scripts/aggregate.py --match smoke_ > /tmp/smoke_agg.log 2>&1; then
  ok "aggregate.py が表を出した"
  sed -n '/系列/,$p' /tmp/smoke_agg.log | head -5 | sed 's/^/      /'
else
  ng "aggregate.py が失敗 → tail /tmp/smoke_agg.log"
fi

printf '\n\033[1m========================================\033[0m\n'
printf ' PASS %d / FAIL %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  cat <<'EOS'
 → clone 直後の確認は通った。次は README.md の
   「4. 1 本のコマンドで回す」に進む。
   ★このスモークで出た成功率は論文の数値ではない（プロトコルを絞っている）。
EOS
else
  echo " → 上の FAIL を直す。README.md の「1. 環境構築」を見る。"
fi
printf '\033[1m========================================\033[0m\n'
exit "$FAIL"
