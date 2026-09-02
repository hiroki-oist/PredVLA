# PredVLA — 再現用リポジトリ

**PredVLA: a sub-million-parameter predictive-coding policy for language-conditioned
robot manipulation** の学習・評価・アブレーション・はしごを、このディレクトリだけで再現する。

675,732 パラメータの予測符号化ポリシー（PV-RNN 準拠の階層的生成モデル）を、
凍結した前段特徴（ResNet18 + MiniLM + PCA）の上で学習し、LIBERO の 4 スイートで
閉ループ評価する。テスト時にはオンライン誤差回帰（ER）で自由変数 `c` を更新する。

| スイート | 成功率（14 シード） | エピソード/シード |
|---|---:|---:|
| LIBERO-Spatial | **83.19 ± 6.10** | 500 |
| LIBERO-Goal | **88.40 ± 3.83** | 500 |
| LIBERO-Object | **89.24 ± 6.13** | 500 |
| LIBERO-Long (libero_10) | **40.57 ± 6.44** | 50 ★1/10 |

確定プロトコル: **Adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / 窓 40**。
数値と設定の対応は `predvla/protocol.py` に 1 か所へ集約してある。**論文の数値を引くときは
そこから引く**（キュースクリプトに散らすと必ず取り違える、というのが元リポジトリの教訓）。

## 入口は 1 つ

学習も評価も集計も、**`scripts/run.py` に渡す変数を変えるだけ**で回る。

```bash
.venv/bin/python scripts/run.py --track <トラック> [選択肢] \
    --suite <スイート> --seeds <シード> --stage <段階>
```

| トラック | 中身 | 対応する表 |
|---|---|---|
| `main` | PredVLA 本線 | 表① 主表 |
| `baseline` | BC-LSTM / BC-Transformer | 表② ベースライン |
| `ladder` | はしご L0〜L7（側枝 L2b などを含む） | 表④ はしご |
| `ablation` | アブレーション A1〜A6 | 表③ アブレーション |
| `all` | 上の 4 つを順に | — |

代表的な学習済み ckpt は `checkpoints/` に**同梱してある**（23 本、61MB）。
学習を回さずに評価から始められる。

---

## 目次

1. [環境構築](#1-環境構築)
2. [データ](#2-データ)
3. [同梱チェックポイント](#3-同梱チェックポイント)
4. [1 本のコマンドで回す](#4-1-本のコマンドで回す)
5. [順を追った再現手順](#5-順を追った再現手順)
6. [トラックごとの詳細](#6-トラックごとの詳細)
7. [集計と数値の読み方](#7-集計と数値の読み方)
8. [ディレクトリ](#8-ディレクトリ)
9. [再現するときの落とし穴](#9-再現するときの落とし穴)

---

## 1. 環境構築

### 前提

- Linux（Ubuntu 24.04 / 22.04 で確認）。CPU だけでも CUDA でも動く
- `uv`（conda は使わない）
- OS 側のライブラリ（MuJoCo のオフスクリーン描画に必要）

```bash
sudo apt-get update && sudo apt-get install -y \
    libegl1 libgl1 libglew-dev libosmesa6-dev patchelf libglfw3 build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv が無いとき
```

### 構築

```bash
git clone <this repo> PredVLA_reproduce && cd PredVLA_reproduce
bash setup.sh                  # CUDA を自動判定
# bash setup.sh --cpu          # CPU 版 torch を強制
# bash setup.sh --cuda cu124   # index を明示（cu121/cu124/cu126/cu128/cu130）
```

`setup.sh` がやること:

1. OS 側ライブラリの確認（足りなければ `apt-get` の 1 行を出して止まる。sudo は勝手に使わない）
2. **`uv` に Python 3.10 を取ってこさせて `.venv` を作る**
   — Ubuntu 24.04 の既定は 3.12 だが、LIBERO の依存（robosuite 1.4 / mujoco 2.3.7 /
   robomimic 0.2）のホイールが 3.10 までしか無く、3.12 ではソースビルドに落ちて失敗する。
   システムの Python は触らない
3. `nvidia-smi` が報告する CUDA 版から torch の index を選ぶ（GPU が無ければ CPU 版）
4. `requirements.lock.txt`（実測環境の凍結、95 件）で残りを入れる
5. LIBERO を **コミット `8f1084e` に固定して** clone し editable で入れる
6. `tools/doctor.py` で健診

### clone 直後の確認（10 分）

```bash
bash tests/smoke.sh            # CPU で回す（どの環境でも通るはず）
bash tests/smoke.sh --cuda     # GPU も使う
```

7 項目を通す:

| # | 見るもの |
|---:|---|
| 1 | 健診（`doctor.py --quick`） |
| 2 | **同梱 ckpt 23 本が実際に `load_state_dict` できるか** |
| 3 | `run.py --track all --dry-run` が全トラックの計画を出せるか |
| 4 | モデルを組めてパラメータ数が想定内か（675,732） |
| 5 | 20 step の学習が通るか |
| 6 | 1 タスク × 2 ロールアウトの閉ループ評価が通るか |
| 7 | 集計が表を出せるか |

**★ここで出る成功率は論文の数値ではない**（20 step のモデルなので 0% になる）。
7 項目すべて PASS なら次に進む。

このスクリプトは `results/smoke_train_s1/` と `results/logs/ev_smoke_*.log` を
毎回消してから作り直すので、**同時に 2 本走らせられない**（後から始まった方の `rm` が
先に走っている方の出力を消し、「全体」行が無いという分かりにくい FAIL になる）。
`results/.smoke.lock` で排他してあり、2 本目は終了コード 2 で即座に止まる。

### 健診

```bash
.venv/bin/python tools/doctor.py
```

Python / torch / device / 依存の版 / MuJoCo の描画バックエンド（EGL・OSMesa・GLFW の
どれが使えるか）/ LIBERO / デモ hdf5 / 凍結特徴キャッシュ を見て、
**LIBERO 環境を 1 つ実際に起動して 1 枚描画するところまで**確認する。
致命的な項目の数が終了コードになる。

**版を厳密に合わせるべきもの**（違うと成功判定や物理が変わる）:
`robosuite==1.4.0` / `mujoco==2.3.7` / `robomimic==0.2.0` / `bddl==3.6.0` / `numpy<2`。
`doctor.py` はこれらだけ別扱いで警告する。

---

## 2. データ

3 段構えで、下に行くほど計算が要る。

```bash
.venv/bin/python tools/prepare_data.py --check      # いま何が揃っているか
```

### (A) 配布された凍結特徴キャッシュを置く ★推奨

`data/` に展開するだけ。**約 250MB**、数分。LIBERO のデモ（32GB）も ResNet の
特徴抽出（GPU 数時間）も要らない。学習と評価はキャッシュしか読まない。

```
data/cache_l64/{libero_spatial,libero_goal,libero_object}/*.h5   +  基底 4 ファイル
data/cache_ps_sp/libero_spatial/*.h5                             +  基底 4 ファイル
data/cache_l64_lg/libero_10/*.h5                                 +  基底 4 ファイル
```

閉ループ評価には**シミュレータ用に LIBERO 本体（bddl と init_files）も必要**だが、
これは `setup.sh` が clone するので、デモ hdf5（32GB）は (A) では不要。

### (B) LIBERO のデモから自分で作る

```bash
.venv/bin/python tools/prepare_data.py --download        # 約 32GB
.venv/bin/python tools/prepare_data.py --build all       # GPU で数時間
```

### ★PCA 基底の取り違えが一番よくある再現失敗

2026-08-22 に実キャッシュを突き合わせて確かめた実態:

| キャッシュ | 視覚 PCA と q の正規化統計を当てたデモ | 使うスイート |
|---|---|---|
| `data/cache_l64` | **libero_spatial のみ** | spatial / goal / object |
| `data/cache_ps_sp` | 同じ手順をもう一度回したもの（視覚基底は `cache_l64` と**バイト単位で同一**、言語 PCA だけ 5e-5 ずれる） | spatial |
| `data/cache_l64_lg` | **libero_10 で張り直し**（基底の最大差 0.14） | libero_10 |

つまり**張り直しが要るのは Long だけ**で、3 スイートは共通基底である。
`tools/prepare_data.py` は「基底を当てるスイートを必ず最初に回す」順序を守る
（`preprocess.run()` は基底ファイルが無いときだけ当てはめ、あれば読むので、
順序がそのまま「どのデモで基底を当てたか」を決める）。

言語 PCA は 64 次元（キャッシュ名の `l64`）。32 次元では最悪ペアの分離が 1/4 になり
タスクを取り違える。

**どのキャッシュをどのスイートに使うかは `predvla/protocol.py` の `CACHE_ROOT` が決める。**
学習済み ckpt には当時の `cache_root` が焼き込まれていて、評価はそれを使う
（`scripts/eval_batch.py` の `_cache_root` が先頭の `../` を剥がして解決する）。

> ★ベースラインの spatial だけは `data/cache_l64` で学習されている（本線 spatial は
> `data/cache_ps_sp`）。この 2 つは視覚基底がバイト単位で同一で言語 PCA が 5e-5 しか
> 違わないので実質同じものだが、config を手で書き換えないこと。公開 ckpt の設定を
> そのまま復元してある。

---

## 3. 同梱チェックポイント

`checkpoints/` に代表的な学習済み ckpt が **23 本（合計 61MB）** 入っている。
`run.py --stage eval` はここを見るので、**学習を 1 step も回さずに評価から始められる。**

| 系列 | 入っているもの | 本数 |
|---|---|---:|
| 本線 PredVLA | `predvla_{spatial,goal,object,long}_s1` | 4 |
| BC-LSTM | `bc_lstm_{spatial,goal,object,long}_s0` | 4 |
| BC-Transformer | `bc_transformer_{spatial,goal,object,long}_s0` | 4 |
| はしご 本線（spatial・シード 1） | `ladder_{L2,L3,L4,L5,L6}_spatial_s1` | 5 |
| はしご 側枝（spatial・シード 1） | `ladder_{L2b,L2-s1,L2b-s1}_spatial_s1` | 3 |
| アブレーション（spatial・シード 0） | `A1_uniform_tau` / `A2_no_pb` / `A3_no_efference` `_spatial_s0` | 3 |

**入っていないものと、その理由**

- **はしごの L0 / L1** — 本線 ckpt をそのまま使う。段の違いは評価プロトコルだけである
  （L0 は ER あり、L1 は `n_itr=0`）
- **はしごの L7** — 独立実装の BC-LSTM そのものなので、`bc_lstm_*` を使う
- **アブレーションの A4 / A5 / A6** — 本線 ckpt に評価フラグを足すだけで、重みは本線と同一
- **シード 2 本目以降** — 論文の表は 7〜14 シードで出ている。同梱は各系列 1 本だけなので、
  **同梱 ckpt から出た数値は「1 シードの値」であって表の値ではない**（§7）

はしごは **L0〜L7 と側枝 3 本のすべてが ckpt 付きで揃っている**（L0/L1 は本線 ckpt、
L7 は BC-LSTM の ckpt を使う）。`--stage eval` だけで全段を通せる。

### ckpt の中身と大きさ

学習の ckpt は再開できるよう全部入りで **1 本 315MB**（Long は 776MB）ある。
だが評価が読むのは 3 つのキーだけである。

| キー | 大きさ | 評価が読むか |
|---|---:|---|
| `model` 675,732 パラメータ | 約 2.7MB | ★読む |
| `cfg` 設定 | 数 KB | ★読む |
| `step` | — | ★読む |
| `C` 自由変数 c（系列 × 時刻 × r × 4 層） | 約 100MB | 読まない |
| `opt` Adam のモーメント（`C` の分が支配的） | 約 205MB | 読まない |
| `rng` numpy / torch の乱数状態 | 小 | 読まない |

**テスト時の自由変数 `c` は毎エピソード事前値から張り直す**ので、学習時の `C` を
持っていても使わない。同梱 ckpt は `model` / `cfg` / `step` だけを残してあり、
315MB → 2.7MB（1/115）になっている。

```bash
# 何が入っているかの一覧
.venv/bin/python tools/install_checkpoints.py --list

# 自分で学習した run を checkpoints/ に入れる（自動で小さくなる）
.venv/bin/python tools/install_checkpoints.py \
    --src results/predvla_spatial_s2/step_30000.pt --name predvla_spatial_s2

# 手元の全部入り ckpt をまとめて小さくする（置き換え）
.venv/bin/python tools/install_checkpoints.py --slim results/*/step_30000.pt
```

**★小さくした ckpt では学習を再開できない**（`C` も `opt` も無い）。
再開する予定があるなら `results/` の元のファイルを消さない。

---

## 4. 1 本のコマンドで回す

```bash
.venv/bin/python scripts/run.py --track <トラック> [選択肢] \
    --suite <スイート> --seeds <シード> --stage <段階>
```

### 共通の引数

| 引数 | 取る値 | 既定 |
|---|---|---|
| `--track` | `main` / `baseline` / `ladder` / `ablation` / `all` | `main` |
| `--suite` | `libero_spatial` / `libero_goal` / `libero_object` / `libero_10` / `all`（4 つ）/ `main`（Long 以外の 3 つ）/ カンマ区切り | `libero_spatial` |
| `--seeds` | `1-14` / `0-6` / `1,3,5` | `1` |
| `--stage` | `train` / `eval` / `aggregate` / `all` | `all` |
| `--device` | 学習の device（`auto` は cuda > mps > cpu） | `auto` |
| `--eval-device` | 評価の device | `cpu` |
| `--jobs` | 評価の並列本数（0 で自動） | `0` |
| `--ckpt-dir` | ckpt の置き場 | `checkpoints/` |
| `--steps` | 学習 step 数（ベースラインは常に 50,000） | `30000` |
| `--redo` | 完了済みの評価セルもやり直す | 切 |
| `--dry-run` | **何が走るか出すだけ** | 切 |

### トラックごとの選択肢

| 引数 | 効くトラック | 取る値 | 既定 |
|---|---|---|---|
| `--model` | `baseline` | `bc_lstm` / `bc_transformer` / `all` | `all` |
| `--rung` | `ladder` | `all`（本線 + 側枝）/ `main`（本線 L0〜L7）/ `side`（側枝だけ）/ 範囲（`L0-L7`, `L2-L5`）/ カンマ区切り（`L2,L4,L2b`） | `all` |
| `--id` | `ablation` | `A1`〜`A6` / `all` / `train`（A1〜A3）/ `test`（A4〜A6）/ カンマ区切り | `all` |

### まず `--dry-run` を見る

```bash
.venv/bin/python scripts/run.py --track all --suite libero_spatial --seeds 1-7 --dry-run
```

何本のプロセスが、どのスクリプトに、どの引数で投げられるかが全部出る。
**30,000 step の学習を 7 シード分うっかり始める前に、必ずこれを見る。**

### `run.py` は何を決めて、何を決めないか

`run.py` が決めるのは「**どの系列のどのシードを、どの順で、どのスクリプトに渡すか**」
だけである。ロールアウト数・打ち切り・ER の設定・Long の「1 プロセス 1 タスク」は
`predvla/protocol.py` にあり、`scripts/evaluate.py` がそこから組み立てる。

```
run.py
  ├ main      → scripts/train.py        → scripts/evaluate.py → scripts/eval_batch.py
  ├ baseline  → benchmark/train.py      → benchmark/eval_suite.py
  ├ ladder    → scripts/run_ladder.py   → scripts/evaluate.py → scripts/eval_batch.py
  │             （L7 だけ baseline トラックへ回す）
  ├ ablation  → scripts/run_ablation.py → scripts/evaluate.py → scripts/eval_batch.py
  └ 最後に      scripts/aggregate.py
```

評価の出力は経路によらず `results/logs/ev_<タグ>.log` に集まり、`aggregate.py` が
同じ形（「`task N success S/T`」と「`全体 XX% (S/N)`」の行）で読む。
**ベースラインも本線と同じ集計器に載る。**

途中で止めて再実行してよい。ログに「全体」の行があるセルは飛ばす。

---

## 5. 順を追った再現手順

### 手順 0 — 環境とデータを揃える

```bash
bash setup.sh
bash tests/smoke.sh                                  # 7 項目すべて PASS を確認
.venv/bin/python tools/prepare_data.py --check       # キャッシュが揃っているか
```

### 手順 1 — 同梱 ckpt で主表を 1 シードだけ再現する（学習しない）

まず**学習を回さずに評価だけ**を通し、閉ループ評価が自分の環境で動くことを確かめる。

```bash
# 何が走るか
.venv/bin/python scripts/run.py --track main --suite all --seeds 1 \
    --stage eval --dry-run

# 実行（3 スイートは 500 エピソード/シード、Long は 50）
.venv/bin/python scripts/run.py --track main --suite all --seeds 1 --stage eval
```

出た値を `predvla/protocol.py` の `MAIN_TABLE_SEEDS` のシード 1 と突き合わせる。

| スイート | シード 1 の参考値 |
|---|---:|
| spatial | 81.6 |
| goal | 84.8 |
| object | 86.0 |
| libero_10 | 46 |

★ロールアウトにはラン間のノイズがあるので、ぴったり一致はしない。数 pt 以内なら通っている。
**14 シード揃うまで表の値（83.19 など）とは比べない。**

### 手順 2 — 本線を学習する

同梱 ckpt はシード 1 だけなので、表を再現するには残りのシードを学習する。

```bash
# まず 1 スイート・1 シードで所要時間を測る
.venv/bin/python scripts/run.py --track main --suite libero_spatial \
    --seeds 2 --stage train --device auto

# 測った時間から見積もって、残りを流す（表① は 14 シード）
.venv/bin/python scripts/run.py --track main --suite libero_spatial \
    --seeds 3-14 --stage train --device auto
```

- 30,000 step。lr は**一定**（`lr_schedule: none`）。
  ★cosine にすると「飽和した」という誤った結論が出るので変えない
- `torch.compile` は `train.compile: true` かつ device が CUDA のときだけ効く。
  CPU では自動で無視される（数値は変わらない）
- `run.py` は `--resume auto` を付けて呼ぶので、途中で落ちても同じコマンドで再開する

| スイート | config | T | τ (top/v/up/low) | cache_root |
|---|---|---:|---|---|
| spatial | `configs/predvla_spatial.yaml` | 200 | 16/8/5/2 | `data/cache_ps_sp` |
| goal | `configs/predvla_goal.yaml` | 200 | 16/8/5/2 | `data/cache_l64` |
| object | `configs/predvla_object.yaml` | 200 | 16/8/5/2 | `data/cache_l64` |
| long | `configs/predvla_long.yaml` | 500 | **30/14/8/2** | `data/cache_l64_lg` |

Long だけ τ が広い。平均エピソード長 / τ_top ≒ 9.2 が最良という掃引結果による。

学習した ckpt を配布用に小さくして `checkpoints/` に集めるなら:

```bash
for s in $(seq 2 14); do
  .venv/bin/python tools/install_checkpoints.py \
      --src results/predvla_spatial_s${s}/step_30000.pt --name predvla_spatial_s${s}
done
```

### 手順 3 — 本線を全シード評価して表① を出す

```bash
.venv/bin/python scripts/run.py --track main --suite all --seeds 1-14 --stage eval
.venv/bin/python scripts/aggregate.py --match predvla_spatial --reference libero_spatial
```

### 手順 4 — ベースライン（表②）

```bash
# 学習（50,000 step）から評価まで、2 種類 × 3 スイート × 7 シード
.venv/bin/python scripts/run.py --track baseline --suite main --seeds 0-6 --stage all

# 片方だけ・同梱 ckpt の評価だけ
.venv/bin/python scripts/run.py --track baseline --model bc_lstm \
    --suite libero_goal --seeds 0 --stage eval
```

### 手順 5 — はしご（表④）とアブレーション（表③）

```bash
# はしご: L0〜L7 を通す。L0/L1 は本線 ckpt、L7 は BC-LSTM に自動で回る
.venv/bin/python scripts/run.py --track ladder --rung L0-L7 \
    --suite libero_spatial --seeds 1-7 --stage all

# 側枝も含めて全部
.venv/bin/python scripts/run.py --track ladder --rung all \
    --suite libero_spatial --seeds 1-7 --stage all

# アブレーション: 再学習が要らないものだけ先に
.venv/bin/python scripts/run.py --track ablation --id A4,A5,A6 \
    --suite libero_spatial --seeds 0-6 --stage eval

# 再学習が要るもの
.venv/bin/python scripts/run.py --track ablation --id A1,A2,A3 \
    --suite libero_spatial --seeds 0-6 --stage all
```

### 手順 6 — 全部まとめて

環境と時間に余裕があるなら、1 本で全系列を流せる。

```bash
.venv/bin/python scripts/run.py --track all --suite all --seeds 1-7 --dry-run   # ★まず見る
.venv/bin/python scripts/run.py --track all --suite all --seeds 1-7
```

**★これは非常に長い。** 本線 4 スイート × 7 シード（30,000 step）+ ベースライン
2 種 × 4 スイート × 7 シード（50,000 step）+ はしご 8 段 + アブレーション 3 段の学習と、
その全部の閉ループ評価が入る。まず 1 セルの実時間を測り、掛け算して見積もってから流す。

---

## 6. トラックごとの詳細

### 6.1 本線（`--track main`）

`configs/predvla_<スイート>.yaml` で 30,000 step 学習し、確定プロトコルで評価する。
config は公開 ckpt に保存されていた `cfg` から機械的に復元してあるので、
論文の数値を出したときの設定と 1 バイトも違わない（`cache_root` の階層だけ直してある）。

評価でスイートごとに自動で変わるもの:

| | 3 スイート | libero_10 (Long) |
|---|---|---|
| ロールアウト | 50/タスク（500 エピソード） | **5/タスク（50 エピソード）** |
| 同時枠 B | 16 | **1** |
| 打ち切り | 600 step 固定 | **最長デモ × 1.2**（タスクごと） |
| プロセス割り | 1 プロセスで 10 タスク | **★1 プロセス 1 タスク** |

**★Long の「1 プロセス 1 タスク」は破ってはいけない。** 複数の `OffScreenRenderEnv` を
1 プロセスに持つと描画コンテキストを共有し、最後に `reset` した env 以外が別シーンの
カメラ姿勢で描かれる。libero_10 は 10 タスクが 9 シーンにまたがるので影響が出る
（これを踏んで 2026-08-10 に結果を 1 度撤回している）。単一シーンの
spatial/goal/object では起きない。`run.py` と `evaluate.py` は自動で分ける。

- **評価の device は `cpu` が既定**。B=16 のロールアウトは CPU の方が速く、GPU は描画に使う
- 並列数は指定しなければ「コア数 − 2」と空き RAM から決める
  （1 セル 1 コア、B=16 で約 6GB、Long の B=1 で約 3GB）

素の `scripts/eval_batch.py` にはつまみが 20 個以上あり、手でキューを書くと必ず取り違える
（論文の数値は設定違いで ±10pt 動く）。**直に叩かない。**

### 6.2 ベースライン（`--track baseline`）

同じ凍結前段・同規模で比較する対照。

| 呼び名 | 実装 | パラメータ数 |
|---|---|---:|
| **BC-LSTM** | 1 層 `nn.LSTMCell` に `[v; q; lang]` を連結。行動 head は本線と同じ GMM | 674,276 |
| **BC-Transformer** | 因果 Transformer。`[v; q; lang]` を線形埋め込みして位置ごとに同じ head | 646,628 |

- **50,000 step**（本線は 30,000）。学習曲線が寝るまでの step 数が違うだけで、
  凍結前段・特徴・スイートは同一
- 視覚は本線と同じく `vision_stride` の窓の中で保持される（5Hz 相当の情報量）
- config は公開 ckpt の `cfg` から復元してある。作り直すなら
  `.venv/bin/python tools/configs_from_ckpt.py --baselines`
- 評価は `benchmark/eval_suite.py` が担うが、ロールアウト数・打ち切り・Long の
  1 プロセス 1 タスクは `run.py` が `protocol.py` から渡すので本線と揃う
  （`eval_suite.py` は `--max-steps-auto` を持たないので、Long ではタスクごとの
  打ち切りを `protocol.MAX_STEPS_LONG_PER_TASK` から直接渡している）

モデルの型名は `src/models/registry.py` が正規化する。公開 ckpt の `cfg` には
学習当時の表記（`bcrnn` / `bctf`）が焼き込まれているので、**旧名も受ける**。

### 6.3 はしご（`--track ladder`）

アブレーション（§6.4）が「本線から部品を 1 つ落とす」のに対し、**はしご**は上の段の
設定を引き継いだまま次の部品を落としていく列である。最下段は独立実装のベースライン
（BC-LSTM）と同じ構成になるので、「ベースラインの実装が弱いだけではないか」を
**同じコードベースの中だけで**確かめられる。

| id | 何を落とす／足すか | 種別 | `d` | 立てるフラグ | 評価 |
|---|---|---|---:|---|---|
| `L0` | 無傷の PredVLA | test | 256 | — | 確定プロトコル |
| `L1` | − オンライン ER（開ループ生成） | test | 256 | — | `n_itr=0` |
| `L2` | − 自由変数 `c` と自由エネルギー | train | 256 | `train.no_c` | `n_itr=0` |
| `L3` | **+ 生の観測を毎 step V に前向きに入れる（予測 head と予測損失は保つ）** | train | 265 | `model.feedforward`, `model.ff_keep_pred`, `model.ff_delay`, `vision_stride=1` | `n_itr=0` |
| `L4` | − 予測（視覚予測 head と予測損失を外す。直入力は L3 のまま） | train | 301 | `ff_keep_pred` / `ff_delay` を落とす | `n_itr=0` |
| `L5` | − 多時定数（全層 τ=1、層数は保つ） | train | 301 | `+ model.ab_tau_flat` | `n_itr=0` |
| `L6` | − PV-RNN セル（LSTM に置き換える） | train | 142 | `+ model.cell=lstm` | `n_itr=0` |
| `L7` | 独立実装の BC-LSTM（1 層 LSTM、`[v;q;lang]` 連結） | baseline | — | — | ベースラインと同じ |

側枝（本線から 1 手だけ動かして、経路と密度を分けて見るための段）

| id | 中身 | `d` | 立てるフラグ |
|---|---|---:|---|
| `L2b` | L2 + 生の視覚を **A_low** に前向きに入れる（V には入れない） | 256 | `model.obs_to_low` |
| `L2-s1` | L2 の**視覚レートだけ** 4 → 1（前向き経路は無いまま） | 256 | `vision_stride=1` |
| `L2b-s1` | L2b の視覚レートだけ 4 → 1 | 256 | `+ vision_stride=1` |

### ★段番号は 2026-09-01 に付け替えた

旧 `L3′`（予測を保ったまま直入力を足す段）が本線に上がって **新 `L3`** になり、
旧 `L3` 以降が 1 つずつ繰り上がった。`L0` / `L1` / `L2` と側枝はそのままである。

| 旧 | 新 | 中身 |
|---|---|---|
| `L3′` (`L3p`) | **`L3`** | prediction-kept（直入力あり・予測あり） |
| `L3` | **`L4`** | no-prediction |
| `L4` | **`L5`** | tau-flat |
| `L5` | **`L6`** | lstm-cell |
| `L6` | **`L7`** | bc-lstm |

**★この付け替えより前に取った評価ログ（`results/logs/ev_ladder_L*.log`）は旧番号で
書かれている。混ぜて集計しない。** 番号に依らない名前でも指定できるので、スクリプトに
書くときはそちらを勧める（`no-prediction` → `L4`、`tau-flat` → `L5`、`lstm-cell` → `L6`、
`bc-lstm` → `L7`）。旧称 `L3p` / `L3'` は中身が同じ新 `L3` に解決される。

同梱 ckpt もこの番号で置き直してある（中身は `cfg` のフラグで照合して確認済み）。

### 何を 1 段ずつ落としているか

`L3` は「直入力を足す」と「視覚レートを 4 → 1 にする」を同時に変えている。レートの
効果だけを見たいときは側枝の `L2-s1`（L2 の視覚レートだけを 1 にしたもの）と比べる。
`L3` → `L4` は**予測だけ**を落とすので、ここが「予測しているかどうか」の対照になる。

**3 つの約束**

1. **パラメータ数を 675,732 ± 1% に揃える。** 段ごとに層幅 `d` を変えて合わせてある。
   揃えないと「簡単なモデルは params が少ないから負けた」という別解釈が残る。

   ```bash
   .venv/bin/python scripts/run_ladder.py --rung all --check-params
   ```

   ★例外は `L2b` / `L2b-s1` で、L2 と `d` を揃える（1 変数だけ動かす）ことを優先した結果
   **724,884（+7.3%）と多い方向にずれる**。少ない方向ではないので上の別解釈は生じない。
2. **L1 以降の評価は必ず `n_itr=0`。** 自由変数 `c` を持たない段で ER は定義できない
   （`L1` は重みこそ本線と同じだが、ER を切って評価する段である）。
   段ごとの評価プロトコルは `predvla/ladder.py` の `protocol` 欄に書いてあり、
   `run_ladder.py` はそれを読むだけ。手で `--protocol` を渡す必要はない。
3. **`ff_delay` を外さない。** L3′ 以降で前向きに入れる観測は 1 step 遅らせる。遅らせないと
   視覚予測 head が同じ step の観測を入力として受け取ってそれを出力することになり、
   予測ではなく再構成になる。

段の config は `configs/ladder/` に自動生成される（手で編集しない）。書き出すだけなら:

```bash
.venv/bin/python scripts/run_ladder.py --rung all --suite libero_spatial \
    --seeds 1-7 --mode config
```

### 6.4 アブレーション（`--track ablation`）

| ID | 何を落とすか | 種別 | 再学習 |
|---|---|---|---|
| `A1_uniform_tau` | 時間階層（全層の τ を 8 に統一） | train | 要 |
| `A2_no_pb` | PB（V→A、`W_pb·d^V` 256→32） | train | 要 |
| `A3_no_efference` | efference copy（A→V、前 step の â と q̂） | train | 要 |
| `A4_no_er_vision` | ER 時のみ視覚予測誤差（★学習時は使う） | test | 不要 |
| `A5_no_er_proprio` | ER 時のみ固有感覚予測誤差（★学習時は使う） | test | 不要 |
| `A6_no_online_er` | オンライン ER 自体（`n_itr=0`、開ループ） | test | 不要 |

`--id` は `A1` のような短い形でも、`A1_uniform_tau` のようなフル ID でも受ける。

**種別を混同すると無駄に 30,000 step 回す。** A4〜A6 は本線 ckpt に評価フラグを
足すだけである（`--id test` でその 3 つだけを選べる）。

★**A1〜A3 と A4〜A6 で見に行くシードが違う。** A1〜A3 は専用 ckpt
（`A2_no_pb_spatial_s0` など）、A4〜A6 は本線 ckpt（`predvla_spatial_s0` など）を探す。
同梱してあるのは A1〜A3 が**シード 0**、本線が**シード 1** なので、同梱 ckpt だけで
動かすなら次のように分けて呼ぶ。

```bash
# A1〜A3（同梱の専用 ckpt はシード 0）
.venv/bin/python scripts/run.py --track ablation --id A1,A2,A3 \
    --suite libero_spatial --seeds 0 --stage eval

# A4〜A6（同梱の本線 ckpt はシード 1）
.venv/bin/python scripts/run.py --track ablation --id test \
    --suite libero_spatial --seeds 1 --stage eval
```

自分で学習して 7 シード揃えたあとは、`--seeds 0-6` でまとめて回せばよい。

τ を 1 に固定するのではなく**平均値 8 に統一する**のが A1 の要点。τ=1 だと
「階層はあるが全層が最速」なので、多時定数の必要性と「遅い層があること」を
分離できない。

**★A1〜A5 の参考値は確定プロトコル（er_w=1.0）では未取得である。**
既存の値は er_w=0.9 で取ったもので、`protocol.py` では
`ABLATION_ERW09_NOT_FOR_CITATION` に「引用禁止」と明示して隔離してある。
A6 だけは `n_itr=0` のとき `er_batch` が er_w も最適化器も参照しないため
**プロトコル不変で有効**（spatial 72.36 ± 6.00 / goal 81.50 ± 3.67 /
object 77.93 ± 4.94 / Long 28.00 ± 6.62）。

**★アブレーションに旧看板プロトコル（`--protocol sgd10`）を使わない。**
無傷のモデルなら主表と区別できないが、A3（efference なし）は SGD10 でだけ
1.00 まで崩壊する。設定とアブレーションの交互作用である。

---

## 7. 集計と数値の読み方

```bash
.venv/bin/python scripts/aggregate.py                                        # 全系列
.venv/bin/python scripts/aggregate.py --match predvla_spatial --per-task
.venv/bin/python scripts/aggregate.py --match predvla_spatial --reference libero_spatial
.venv/bin/python scripts/aggregate.py --compare predvla_spatial A2_no_pb__predvla_spatial
```

`run.py --stage all` と `--stage eval` は最後に自動で集計を出す。`--stage aggregate` で
集計だけを回せる。

**読むときの約束**

- 平均だけでなく必ず **±SD と n** を出す。シード分散は σ≈4〜6pt（3 スイート）/
  6.4pt（Long）ある
- **n<7 は「候補」と印を付ける。** この系列は n=3 の ±5pt が n=7 で消えた実例が 4 件ある
- Long は 1 シードが 10 個のログに分かれる。**10 個揃っていないシードは平均から外す**
  （部分値は成功エピソードが先に終わるので上向きに偏る。実測で 3 回踏んだ:
  6/10 で 50.0 → 確定 36.0 など）。`aggregate.py` は未完了セルを検出して除外する
- **同梱 ckpt だけで出した数値は 1 シードの値**。表の平均と比べない（§3）

論文の数値は `predvla/protocol.py` の `MAIN_TABLE`（平均・SD・n）と
`MAIN_TABLE_SEEDS`（シード別）にある。**引用はここから引く。**

---

## 8. ディレクトリ

```
scripts/
  run.py                ★★ 唯一の入口。トラックを渡すだけで学習〜集計まで
  train.py              本線の学習
  evaluate.py           評価の入口。protocol.py から引数を組み立てる
  eval_batch.py         閉ループ評価の本体（つまみが多い。直に叩かない）
  aggregate.py          ログ → 表
  run_ladder.py         はしごの入口（run.py から呼ばれる）
  run_ablation.py       アブレーションの入口（run.py から呼ばれる）

predvla/                本体
  model.py              階層生成モデル（T → V → A_up → A_low、675,732 パラメータ）
  er.py / er_batch.py   テスト時のオンライン誤差回帰（ER）。er_batch が評価用
  er_train.py           ER をループに通した学習（次論文向け。既定は無効）
  data.py               キャッシュから系列を切り出す
  frontend.py           凍結前段（ResNet18 + MiniLM + PCA）の読み込み
  protocol.py           ★プロトコル定数・命名規則・論文の参考値。数値はここから引く
  ladder.py             ★はしごの段の定義（何を落として d をいくつにするか）

src/                    凍結前段と対照モデル
  models/encoders.py    凍結 ResNet18 と MiniLM
  models/registry.py    cfg の model.type からモデルを組む（旧表記も受ける）
  models/baseline_bc_lstm.py         BC-LSTM
  models/baseline_bc_transformer.py  BC-Transformer
  models/baseline_mtrnn.py           MT-RNN（追加の対照）
  models/pc_cells.py    予測符号化 RNN の部品（ベースラインの head を共有する）
  models/legacy_agent.py 旧世代の予測符号化エージェント（benchmark/ 側にだけ残る）
  data/pca.py           凍結 PCA
  data/preprocess.py    デモ hdf5 → 凍結特徴キャッシュ
  data/dataset.py       ベースライン用のチャンク切り出し
  eval/rollout.py       ベースラインの閉ループ評価
  utils/                device 判定・MUJOCO_GL・config 合成・シード

benchmark/              ベースラインの学習（train.py）と評価（eval_suite.py）

tools/
  doctor.py             環境の健診
  prepare_data.py       デモの取得と凍結特徴キャッシュの作成
  install_checkpoints.py ★ckpt を小さくして checkpoints/ に置く
  configs_from_ckpt.py  ckpt の cfg から yaml を復元
  freeze_env.py         requirements.lock.txt の生成

configs/                学習設定（はしごの段は configs/ladder/ に生成される）
checkpoints/            ★同梱の学習済み ckpt（23 本、61MB）
data/                   デモと凍結特徴キャッシュ（gitignore）
third_party/LIBERO/     setup.sh が clone（gitignore）
results/                学習と評価の出力（gitignore）
  logs/ev_<タグ>.log    評価ログ。aggregate.py が読む
tests/smoke.sh          clone 直後の 7 項目確認
```

### 命名規則

学習の run 名・ckpt のファイル名・評価ログのタグは 1 つの規則に従う。
決めているのは `predvla/protocol.py` の `run_name()` / `series_name()` だけである。

```
<系列>_<スイート短縮名>_s<シード>

  系列   predvla                       本線
         bc_lstm / bc_transformer      ベースライン
         ladder_<段>                   はしご（L2〜L6, L2b, …）
         <ID>_<何を落とすか>           アブレーション（A1_uniform_tau など）

  例     predvla_spatial_s1
         bc_lstm_goal_s3
         ladder_L3_spatial_s1
         A2_no_pb_spatial_s0
```

`run.py` と `evaluate.py` は、配布 ckpt の平置き（`checkpoints/<run 名>.pt`）と
学習の出力（`results/<run 名>/step_<N>.pt`）の両方を探す。

---

## 9. 再現するときの落とし穴

実際に踏んだものだけを挙げる。

1. **PCA 基底の取り違え** — spatial/goal/object は共通基底、Long だけ張り直し（§2）
2. **Long を 1 プロセスに複数タスク詰める** — 描画コンテキストの共有で別シーンの
   カメラ姿勢になる（§6.1）
3. **部分値から向きを読む** — 成功エピソードが先に終わるので上向きに偏る。
   10 セル揃うまで平均を出さない（§7）
4. **n=3 で結論を出す** — σ≈4〜11pt。n=3 の ±5pt が n=7 で消えた実例が 4 件ある
5. **同梱 ckpt の 1 シードを表の値と比べる** — 表は 7〜14 シードの平均である（§3）
6. **アブレーションの種別を混同する** — A4〜A6 は再学習が不要（§6.4）
7. **er_w を動かす** — 1.0 が頂点。0.01 まで下げると −26.7pt、1.2 は n=3 で +2.7 に
   見えて n=6 で −0.4 に転じた
8. **lr スケジュールを cosine にする** — `total_steps` で lr が 0 になるので
   「飽和した」という誤った結論が出る。`lr_schedule: none` を変えない
9. **`numpy>=2` を入れる** — robosuite 1.4 が壊れる。`doctor.py` が検出する
10. **はしごの段を `n_itr>0` で評価する** — L2 以降は `c` を持たないので ER が定義
    できない。段ごとのプロトコルは `ladder.py` が決める（§6.3）
11. **旧番号のはしごの結果と新番号を混ぜる** — 2026-09-01 に L3 以降を繰り上げた。
    旧 `L3`(no-prediction) は新 `L4` である（§6.3 の対応表）
12. **小さくした ckpt から学習を再開しようとする** — `C` も `opt` も落としてある（§3）
13. **Ubuntu 24.04 の既定 Python 3.12 を使う** — LIBERO の依存のホイールが無い。
    `setup.sh` が 3.10 を取ってくる
