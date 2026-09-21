#!/usr/bin/env bash
# PredVLA 再現環境の構築。Ubuntu 24.04 / 22.04 で CPU 環境でも CUDA 環境でも通る。
#
#   bash setup.sh                CUDA を自動判定して入れる
#   bash setup.sh --cpu          CPU 版 torch を強制する
#   bash setup.sh --cuda cu124   CUDA の index を明示する（cu121/cu124/cu126/cu128/cu130）
#   bash setup.sh --skip-libero  LIBERO の clone/install を飛ばす（既にある場合）
#
# ★システムの Python は触らない。uv が Python 3.10 を取ってきて .venv を作る。
#   Ubuntu 24.04 の既定は 3.12 で、LIBERO の依存（robosuite 1.4 / mujoco 2.3.7）の
#   ホイールが無くビルドに落ちるため、3.10 を明示的に使う。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY_VERSION=3.10
LIBERO_COMMIT=8f1084e            # 2026-08 時点で論文の数値を出しているコミット
LIBERO_REPO=https://github.com/Lifelong-Robot-Learning/LIBERO.git

FORCE_CPU=0
CUDA_TAG=""
SKIP_LIBERO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)          FORCE_CPU=1; shift;;
    --cuda)         CUDA_TAG="$2"; shift 2;;
    --skip-libero)  SKIP_LIBERO=1; shift;;
    -h|--help)      sed -n '2,12p' "$0"; exit 0;;
    *) echo "不明な引数: $1"; exit 2;;
  esac
done

say() { printf '\n\033[1m[setup] %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------
# 0. OS 側のライブラリ（MuJoCo のオフスクリーン描画に必要）
# --------------------------------------------------------------------------
say "OS 側のライブラリを確認する"
NEED_APT=()
for pkg in libegl1 libgl1 libglew-dev libosmesa6-dev patchelf libglfw3 build-essential; do
  dpkg -s "$pkg" >/dev/null 2>&1 || NEED_APT+=("$pkg")
done
if [ ${#NEED_APT[@]} -gt 0 ]; then
  echo "  以下が足りない: ${NEED_APT[*]}"
  echo "  実行してから setup.sh をやり直すこと:"
  echo "      sudo apt-get update && sudo apt-get install -y ${NEED_APT[*]}"
  echo "  ★このスクリプトは sudo を勝手に使わない。"
  exit 1
fi
echo "  OK"

# --------------------------------------------------------------------------
# 1. uv
# --------------------------------------------------------------------------
say "uv を確認する"
if ! command -v uv >/dev/null 2>&1; then
  echo "  uv が無い。入れる:"
  echo "      curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  （conda は使わない。この構成は uv + .venv 前提）"
  exit 1
fi
uv --version

# --------------------------------------------------------------------------
# 2. .venv (Python 3.10)
# --------------------------------------------------------------------------
say ".venv を作る（Python ${PY_VERSION}）"
# 既存が symlink（開発機で本リポジトリの venv を共有している場合）なら触らない
if [ -L .venv ]; then
  echo "  .venv は symlink なのでそのまま使う -> $(readlink .venv)"
elif [ -x .venv/bin/python ]; then
  # 2 回目以降（apt パッケージを入れてやり直す等）は既存の .venv を使う。
  # uv venv は既存ディレクトリがあると止まるので作り直さない（中身は下で上書きされる）。
  echo "  既存の .venv を使う"
else
  uv venv --python "${PY_VERSION}" .venv
fi
VPY="$ROOT/.venv/bin/python"
"$VPY" -V

# --------------------------------------------------------------------------
# 3. torch（CUDA / CPU の切り替え）
# --------------------------------------------------------------------------
say "torch を入れる"
if [ "$FORCE_CPU" = "1" ]; then
  IDX="https://download.pytorch.org/whl/cpu"
  echo "  --cpu 指定 -> CPU 版"
elif [ -n "$CUDA_TAG" ]; then
  IDX="https://download.pytorch.org/whl/${CUDA_TAG}"
  echo "  --cuda ${CUDA_TAG} 指定"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  # ドライバが報告する CUDA 版から index を選ぶ
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  CUDA_RT=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  echo "  NVIDIA GPU を検出（driver ${DRV}, CUDA ${CUDA_RT}）"
  case "$CUDA_RT" in
    13.*) IDX="https://download.pytorch.org/whl/cu130";;
    12.8|12.9) IDX="https://download.pytorch.org/whl/cu128";;
    12.6|12.7) IDX="https://download.pytorch.org/whl/cu126";;
    12.4|12.5) IDX="https://download.pytorch.org/whl/cu124";;
    12.*) IDX="https://download.pytorch.org/whl/cu121";;
    *)    IDX="https://download.pytorch.org/whl/cpu";
          echo "  ★CUDA ${CUDA_RT} に対応する index が判らないので CPU 版にする";;
  esac
else
  IDX="https://download.pytorch.org/whl/cpu"
  echo "  GPU が見えない -> CPU 版"
fi
echo "  index: $IDX"
uv pip install --python "$VPY" --index-url "$IDX" torch torchvision

# --------------------------------------------------------------------------
# 4. 残りの依存
# --------------------------------------------------------------------------
say "残りの依存を入れる"
if [ -f requirements.lock.txt ]; then
  echo "  requirements.lock.txt（実測環境の凍結）を使う"
  # torch/torchvision は上で入れた版を保つため lock から外す
  grep -viE '^(torch|torchvision)==' requirements.lock.txt > /tmp/predvla_lock.$$ || true
  uv pip install --python "$VPY" -r /tmp/predvla_lock.$$ || {
    echo "  ★lock での解決に失敗した。pyproject の範囲指定で入れ直す"
    uv pip install --python "$VPY" -e .
  }
  rm -f /tmp/predvla_lock.$$
else
  uv pip install --python "$VPY" -e .
fi

# --------------------------------------------------------------------------
# 5. LIBERO（PyPI に無いので clone して editable）
# --------------------------------------------------------------------------
if [ "$SKIP_LIBERO" = "0" ]; then
  say "LIBERO を入れる（コミット ${LIBERO_COMMIT} に固定）"
  mkdir -p third_party
  if [ ! -d third_party/LIBERO/.git ]; then
    git clone "$LIBERO_REPO" third_party/LIBERO
  fi
  ( cd third_party/LIBERO && git fetch --all --quiet && git checkout --quiet "$LIBERO_COMMIT" )
  # LIBERO のリポジトリには libero/__init__.py が無く、名前空間が壊れる
  touch third_party/LIBERO/libero/__init__.py
  uv pip install --python "$VPY" --no-deps -e third_party/LIBERO
else
  say "LIBERO は --skip-libero でスキップ"
fi

# --------------------------------------------------------------------------
# 6. robosuite の macro ファイル（警告を消す。挙動には影響しない）
# --------------------------------------------------------------------------
say "robosuite の macro を設定する"
"$VPY" -c "
import os, robosuite
p = os.path.join(os.path.dirname(robosuite.__file__), 'scripts', 'setup_macros.py')
print('  ', p)
" || true
"$VPY" "$("$VPY" -c "import os,robosuite;print(os.path.join(os.path.dirname(robosuite.__file__),'scripts','setup_macros.py'))")" >/dev/null 2>&1 || true

# --------------------------------------------------------------------------
# 7. 健診
# --------------------------------------------------------------------------
say "環境を健診する"
"$VPY" tools/doctor.py || true

cat <<'EOS'

[setup] 完了。次にやること:

  1. LIBERO のデモを置く（既にあれば飛ばす）
         python tools/prepare_data.py --download        # LIBERO のデモを取得
         python tools/prepare_data.py --suite all       # 凍結特徴のキャッシュを作る

  2. 学習
         python scripts/train.py --config configs/predvla_spatial.yaml --device auto

  3. 評価（1 コマンド）
         python scripts/evaluate.py --suite libero_spatial --ckpt <ckpt> --protocol table1p

  4. アブレーション（1 コマンド）
         python scripts/run_ablation.py --ablation all --suite libero_spatial --seeds 0-6

  詳細は README.md を読むこと。
EOS
