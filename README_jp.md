# PredVLA — 再現用リポジトリ

**PredVLA: Predictive Sensorimotor Modeling for Sub-Million-Parameter Robot Manipulation** の学習・評価・アブレーションを再現するためのリポジトリです。

PredVLA は、凍結した視覚・言語特徴（ResNet18 + MiniLM + PCA）を用いる、675,732 パラメータの階層的予測符号化ポリシーです。LIBERO の各スイート上で学習し、テスト時にはオンライン Error Regression (ER) により自由変数 `c` を更新します。

論文で用いた主要な評価設定は `predvla/protocol.py` に集約されています。

## 論文の主結果

| Suite | Success rate |
|---|---:|
| LIBERO-Spatial | **83.19 ± 6.10** |
| LIBERO-Goal | **88.40 ± 3.83** |
| LIBERO-Object | **89.24 ± 6.13** |
| LIBERO-Long (`libero_10`) | **40.57 ± 6.44** |

主要評価プロトコル:

```text
Adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / ER window=40
```

---

## 1. 環境構築

### 必要環境

- Linux
- Python 3.10
- `uv`
- MuJoCo のオフスクリーン描画に必要なシステムライブラリ

Ubuntu では以下をインストールしてください。

```bash
sudo apt-get update
sudo apt-get install -y \
    libegl1 libgl1 libglew-dev libosmesa6-dev patchelf libglfw3 build-essential
```

`uv` が未導入の場合:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### セットアップ

```bash
git clone https://github.com/CyberneticHumanity/PredVLA.git PredVLA_reproduce
cd PredVLA_reproduce

bash setup.sh
```

CPU 版 PyTorch を使う場合:

```bash
bash setup.sh --cpu
```

CUDA の index を明示する場合:

```bash
bash setup.sh --cuda cu124
```

`setup.sh` は Python 3.10 の仮想環境作成、依存関係の導入、LIBERO の取得、環境チェックまで行います。

### 動作確認

```bash
bash tests/smoke.sh
```

GPU も確認する場合:

```bash
bash tests/smoke.sh --cuda
```

より詳しい環境診断:

```bash
.venv/bin/python tools/doctor.py
```

再現性のため、以下の主要依存は固定されています。

```text
robosuite==1.4.0
mujoco==2.3.7
robomimic==0.2.0
bddl==3.6.0
numpy<2
```

---

## 2. データ準備

現在のデータ状態を確認:

```bash
.venv/bin/python tools/prepare_data.py --check
```

### 推奨: 同梱の凍結特徴キャッシュを使用

凍結特徴キャッシュ（LIBERO デモの ResNet18 + MiniLM + PCA 特徴、計 ≈870 MB）は **このリポジトリに同梱**しています（`data/` 以下）。`git clone` だけで学習と同梱 ckpt の評価ができます。

```text
data/
├── cache_l64/            # shared PCA basis (fit on libero_spatial demos); used for goal / object
│   ├── libero_spatial/
│   ├── libero_goal/
│   ├── libero_object/
│   ├── libero_10/
│   └── libero_90/
├── cache_ps_sp/          # per-suite refit; used for spatial
│   └── libero_spatial/
└── cache_l64_lg/         # PCA refit on libero_10 demos; used for long
    └── libero_10/
```

この方法では LIBERO のデモデータ全体や、ResNet/MiniLM/PCA 特徴の再計算は不要です。

### 特徴を自分で再生成する場合

LIBERO デモを取得:

```bash
.venv/bin/python tools/prepare_data.py --download
```

全特徴キャッシュを生成:

```bash
.venv/bin/python tools/prepare_data.py --build all
```

---

## 3. 最短で動かす

学習・評価・集計はすべて `scripts/run.py` から実行できます。

基本形:

```bash
.venv/bin/python scripts/run.py \
    --track <track> \
    --suite <suite> \
    --seeds <seeds> \
    --stage <stage>
```

### Track

| Track | 内容 |
|---|---|
| `main` | PredVLA 本体 |
| `baseline` | BC-LSTM / BC-Transformer |
| `ladder` | PredVLA から BC-LSTM までの mechanism ladder |
| `ablation` | 各種アブレーション |
| `all` | 全トラック |

### 主な引数

| 引数 | 例 |
|---|---|
| `--suite` | `libero_spatial`, `libero_goal`, `libero_object`, `libero_10`, `main`, `all` |
| `--seeds` | `1`, `1-7`, `1-14`, `1,3,5` |
| `--stage` | `train`, `eval`, `aggregate`, `all` |
| `--device` | `auto`, `cuda`, `cpu` |
| `--eval-device` | `cpu`, `cuda` |
| `--jobs` | 評価の並列数 |
| `--redo` | 完了済み評価も再実行 |
| `--dry-run` | 実行内容のみ表示 |

実行前に `--dry-run` で確認できます。

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 1 \
    --stage eval \
    --dry-run
```

---

## 4. 学習済みチェックポイントで評価する

`checkpoints/` に代表的な学習済みチェックポイントが含まれているため、学習を行わず評価から開始できます。

例: PredVLA を全スイートで 1 seed 評価

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite all \
    --seeds 1 \
    --stage eval
```

評価ログは以下に保存されます。

```text
results/logs/
```

集計:

```bash
.venv/bin/python scripts/aggregate.py
```

特定系列のみ集計:

```bash
.venv/bin/python scripts/aggregate.py \
    --match predvla_spatial \
    --reference libero_spatial
```

---

## 5. PredVLA を再学習する

### 1 suite / 1 seed

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 2 \
    --stage train \
    --device auto
```

### 論文の main result を再現

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite all \
    --seeds 1-14 \
    --stage all
```

PredVLA は基本的に 30,000 steps 学習します。

Suite ごとの設定:

| Suite | Config |
|---|---|
| Spatial | `configs/predvla_spatial.yaml` |
| Goal | `configs/predvla_goal.yaml` |
| Object | `configs/predvla_object.yaml` |
| Long | `configs/predvla_long.yaml` |

Long ではより長い時間スケールを用います。

---

## 6. ベースライン

同じ凍結前段特徴を使った BC-LSTM と BC-Transformer を再現できます。

```bash
.venv/bin/python scripts/run.py \
    --track baseline \
    --suite main \
    --seeds 0-6 \
    --stage all
```

片方だけ評価する例:

```bash
.venv/bin/python scripts/run.py \
    --track baseline \
    --model bc_lstm \
    --suite libero_goal \
    --seeds 0 \
    --stage eval
```

| Model | Parameters |
|---|---:|
| BC-LSTM | 674,276 |
| BC-Transformer | 646,628 |

ベースラインは 50,000 steps 学習します。

---

## 7. Mechanism Ladder

PredVLA の構成要素を段階的に置き換え、最終的に BC-LSTM に近づける比較です。

| ID | 構成 |
|---|---|
| `L0` | PredVLA |
| `L1` | online ER なし |
| `L2` | 自由変数 `c` / free-energy optimization なし |
| `L3` | 観測を feed-forward 入力 |
| `L4` | prediction objective なし |
| `L5` | multi-timescale dynamics なし |
| `L6` | PV-RNN cell を LSTM に置換 |
| `L7` | BC-LSTM |

全段を実行:

```bash
.venv/bin/python scripts/run.py \
    --track ladder \
    --rung L0-L7 \
    --suite libero_spatial \
    --seeds 1-7 \
    --stage all
```

側枝を含む全設定:

```bash
.venv/bin/python scripts/run.py \
    --track ladder \
    --rung all \
    --suite libero_spatial \
    --seeds 1-7 \
    --stage all
```

各段の詳細な定義は `predvla/ladder.py` にあります。

---

## 8. Ablation

| ID | Ablation | 再学習 |
|---|---|---|
| `A1` | uniform timescale | 必要 |
| `A2` | predictive bottleneck なし | 必要 |
| `A3` | efference copy なし | 必要 |
| `A4` | ER から visual prediction error を除外 | 不要 |
| `A5` | ER から proprioceptive prediction error を除外 | 不要 |
| `A6` | online ER なし | 不要 |

再学習不要のアブレーション:

```bash
.venv/bin/python scripts/run.py \
    --track ablation \
    --id A4,A5,A6 \
    --suite libero_spatial \
    --seeds 0-6 \
    --stage eval
```

再学習が必要なアブレーション:

```bash
.venv/bin/python scripts/run.py \
    --track ablation \
    --id A1,A2,A3 \
    --suite libero_spatial \
    --seeds 0-6 \
    --stage all
```

---

## 9. 全実験をまとめて実行

まず実行計画を確認してください。

```bash
.venv/bin/python scripts/run.py \
    --track all \
    --suite all \
    --seeds 1-7 \
    --dry-run
```

実行:

```bash
.venv/bin/python scripts/run.py \
    --track all \
    --suite all \
    --seeds 1-7
```

全実験は計算量が大きいため、通常は track / suite / seed を分けて実行することを推奨します。

---

## 10. ディレクトリ構成

```text
scripts/
  run.py              # 学習・評価・集計の統一入口
  train.py            # PredVLA 学習
  evaluate.py         # PredVLA 評価
  aggregate.py        # 評価ログ集計
  run_ladder.py       # ladder
  run_ablation.py     # ablation

predvla/
  model.py            # PredVLA 本体
  er.py               # Error Regression
  er_batch.py         # batched ER
  data.py             # 学習データ
  frontend.py         # frozen frontend
  protocol.py         # 評価プロトコル
  ladder.py           # ladder 定義

benchmark/
  train.py            # baseline 学習
  eval_suite.py       # baseline 評価

configs/
  predvla_*.yaml      # PredVLA 設定

checkpoints/           # 配布済み学習済みモデル
data/                  # データ・特徴キャッシュ
results/               # 学習・評価結果
tests/smoke.sh         # 最小動作確認
tools/                 # 環境確認・データ準備など
```

---

## 11. 再現時の注意

- `numpy<2` を使用してください。
- LIBERO / robosuite / MuJoCo のバージョンは固定してください。
- Long (`libero_10`) の評価はリポジトリ標準の runner を使用してください。
- Ladder の評価設定は `predvla/ladder.py` に従ってください。
- 学習済み checkpoint から得られる単一 seed の結果と、論文中の複数 seed 平均を直接比較しないでください。
- 学習・評価の低レベルスクリプトを直接呼ぶより、`scripts/run.py` の使用を推奨します。

---

## 推奨する再現手順

最小構成では、以下の順で確認してください。

```bash
# 1. setup
bash setup.sh

# 2. smoke test
bash tests/smoke.sh

# 3. data check
.venv/bin/python tools/prepare_data.py --check

# 4. pretrained checkpoint の評価
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 1 \
    --stage eval

# 5. 1 seed を再学習
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 2 \
    --stage all

# 6. 必要に応じて全 seed / baseline / ladder / ablation を実行
```

まずはこの最小手順が通ることを確認したあと、論文中の各表に対応する実験へ拡張してください。
