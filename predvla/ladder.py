"""「はしご」の定義: PredVLA から BC-LSTM まで、機構を 1 段ずつ外していく対照列。

表③のアブレーション（`predvla/protocol.py` の `ABLATIONS`）が「1 つの部品を落とす」
のに対し、こちらは **上の段の設定をそのまま引き継いで次の部品を落とす**。
最後の段は独立実装のベースライン（BC-LSTM）と同じ構成になる。

段の一覧（上から順に落としていく）

    L0   無傷の PredVLA                       本線 ckpt（再学習は不要）
    L1   − オンライン ER                     本線 ckpt を n_itr=0 で評価するだけ
    L2   − 自由変数 c と自由エネルギー        train.no_c で再学習
    L3   + 生の観測を前向きに入れる（予測は保つ）  model.feedforward + ff_keep_pred
    L4   − 予測（視覚予測 head と予測損失を外す）  ff_keep_pred を落とす
    L5   − 多時定数（全層 τ=1）              + model.ab_tau_flat
    L6   − PV-RNN セル（LSTM にする）        + model.cell=lstm
    L7   独立実装の BC-LSTM                  benchmark/ 側で学習・評価する

    側枝（本線から 1 手だけ動かして、経路と密度を分けて見るための段）
    L2b   L2 + 生の視覚を A_low に前向きに入れる（V には入れない）
    L2-s1 / L2b-s1  L2 / L2b の視覚レートだけを 4 → 1 に変えたもの

★段番号は 2026-09-01 に付け替えた（ユーザー指示）。
  旧 L3'（予測を保ったまま直入力）が本線に上がって **新 L3** になり、
  旧 L3 以降が 1 つずつ繰り上がった。対応表:

      旧 L3'（prediction-kept）→ 新 L3
      旧 L3  （no-prediction） → 新 L4
      旧 L4  （tau-flat）      → 新 L5
      旧 L5  （lstm-cell）     → 新 L6
      旧 L6  （bc-lstm）       → 新 L7
      L0 / L1 / L2 / L2b / L2-s1 / L2b-s1 は変わらない

  ★これより前に取った評価ログ（`results/logs/ev_ladder_L*.log`）は旧番号で
    書かれている。混ぜて集計しない。

★2 つの約束

  1. **パラメータ数を揃える**（675,732 ± 1%）。段ごとに層幅 `d` を変えて合わせてある。
     揃えないと「簡単なモデルは params が少ないから負けた」という別解釈が残る。
  2. **L2 以降の評価は必ず `n_itr=0`**。自由変数 c を持たないので ER が定義できない。
     段ごとの評価プロトコルは下の `protocol` 欄に書いてあり、`scripts/run_ladder.py`
     はそれを読むだけ（手で `--protocol` を渡す必要はない）。
     L0 だけが確定プロトコル（ER あり）で、L1 以降はすべて `ni0`。

★`ff_delay`（L3 以降）について
  前向きに入れる観測は 1 step 遅らせる。遅らせないと、視覚予測 head は同じ step の
  観測を入力として受け取ったうえでそれを出力することになり、予測ではなく再構成に
  なってしまう（`predvla/model.py` の `one_step` / `generate` を参照）。
"""
from __future__ import annotations

import os
from typing import Any

# 本線（L0）のパラメータ数。全段をこれに合わせる。
PARAMS_TARGET = 675_732
PARAMS_TOL = 0.01           # ±1%

# id -> 段の定義
#   name   : リリースフォルダと同じ表記（release_arxiv/runs/ の名前と対応）
#   kind   : test  = 本線 ckpt に評価フラグを足すだけ（再学習しない）
#            train = 再学習が要る
#            baseline = 別実装（benchmark/ 側）。ここでは手順だけ示す
#   d        : 層幅。パラメータ数を本線に合わせるための値
#   protocol : 評価プロトコル（predvla/protocol.py の PROTOCOLS の名前）
#              main = 確定プロトコル（ER あり）/ ni0 = n_itr=0
#   model / train / data : 本線 config への上書き
RUNGS: dict[str, dict[str, Any]] = {
    "L0": dict(
        name="L0_intact", kind="test", d=None, protocol="main",
        desc="無傷の PredVLA（確定プロトコル = Adam n_itr10 / er_lr 0.05 / er_w 1.0）",
        model={}, train={}, data={}),

    "L1": dict(
        name="L1_no-ER", kind="test", d=None, protocol="ni0",
        desc="− オンライン ER（n_itr=0 の開ループ生成）。重みは本線と同一",
        model={}, train={}, data={}),

    "L2": dict(
        name="L2_no-free-energy", kind="train", d=256, protocol="ni0",
        desc="− 自由変数 c と自由エネルギー（KL）。観測は予測誤差としてのみ効く",
        model={}, train={"no_c": True}, data={}),

    "L2b": dict(
        name="L2b_vision-to-action", kind="train", d=256, protocol="ni0",
        desc="側枝: L2 + 生の視覚を A_low に前向きに入れる（V には入れない）",
        model={"obs_to_low": True}, train={"no_c": True}, data={},
        params_note="d を L2 と同じ 256 に固定したので +7.3%（724,884）。"
                    "A_low へ入る 192 次元のブリッジ分。1 変数だけ動かす方を優先した"),

    "L2-s1": dict(
        name="L2_no-free-energy_stride1", kind="train", d=256, protocol="ni0",
        desc="側枝: L2 の視覚レートだけを 4 → 1（前向き経路は無いまま）",
        model={}, train={"no_c": True}, data={"vision_stride": 1}),

    "L2b-s1": dict(
        name="L2b_vision-to-action_stride1", kind="train", d=256, protocol="ni0",
        desc="側枝: L2b の視覚レートだけを 4 → 1",
        model={"obs_to_low": True}, train={"no_c": True},
        data={"vision_stride": 1},
        params_note="L2b と同じ 724,884（+7.3%）"),

    "L3": dict(
        name="L3_prediction-kept", kind="train", d=265, protocol="ni0",
        desc="+ 生の観測を毎 step V に前向きに入れる（★視覚・固有感覚の予測 head と "
             "予測損失は保つ）。前向きの観測は 1 step 遅らせる",
        model={"feedforward": True, "ff_keep_pred": True, "ff_delay": True},
        train={"no_c": True}, data={"vision_stride": 1}),

    "L4": dict(
        name="L4_no-prediction", kind="train", d=301, protocol="ni0",
        desc="− 予測。視覚予測 head と予測損失を外す（直入力は L3 のまま残る）",
        model={"feedforward": True}, train={"no_c": True},
        data={"vision_stride": 1}),

    "L5": dict(
        name="L5_tau-flat", kind="train", d=301, protocol="ni0",
        desc="− 多時定数（全層 τ=1）。層数は保つ",
        model={"feedforward": True, "ab_tau_flat": True},
        train={"no_c": True}, data={"vision_stride": 1}),

    "L6": dict(
        name="L6_lstm-cell", kind="train", d=142, protocol="ni0",
        desc="− PV-RNN セル（LSTM に置き換える）",
        model={"feedforward": True, "ab_tau_flat": True, "cell": "lstm"},
        train={"no_c": True}, data={"vision_stride": 1}),

    "L7": dict(
        name="L7_bc-lstm", kind="baseline", d=None, protocol=None,
        desc="独立実装の BC-LSTM（1 層 LSTM に [v; q; lang] を連結）。benchmark/ 側で学習する",
        model={}, train={}, data={}),
}

# 別名（論文・リリースの表記から引けるようにする）
ALIASES = {
    "l0": "L0", "intact": "L0", "main": "L0",
    "l1": "L1", "no-er": "L1", "no_er": "L1",
    "l2": "L2", "no-free-energy": "L2",
    "l2b": "L2b", "vision-to-action": "L2b",
    "l2s1": "L2-s1", "l2-s1": "L2-s1",
    "l2bs1": "L2b-s1", "l2b-s1": "L2b-s1",
    # ★中身で引ける名前。番号を付け替えても意味が動かないのでこちらを推奨する。
    "l3": "L3", "prediction-kept": "L3",
    "l4": "L4", "no-prediction": "L4",
    "l5": "L5", "tau-flat": "L5",
    "l6": "L6", "lstm-cell": "L6",
    "l7": "L7", "bc-lstm": "L7", "bclstm": "L7",
    # 旧 L3'（= いまの L3）の呼び名。中身は同じなので受ける。
    "l3p": "L3", "l3'": "L3",
}

# 既定で回す順（側枝を含まない本線のはしご）
MAIN_LINE = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
SIDE = ["L2b", "L2-s1", "L2b-s1"]
ALL = MAIN_LINE + SIDE


def resolve(x: str) -> str:
    """CLI で来た名前を段の id に直す。"""
    if x in RUNGS:
        return x
    k = ALIASES.get(x.strip().lower())
    if k is None:
        raise KeyError(x)
    return k


def make_cfg(rung: str, base_cfg: dict, seed: int, run_name: str,
             total_steps: int) -> dict:
    """本線 config に段の上書きを当てた config を返す（元の dict は壊さない）。"""
    import copy
    spec = RUNGS[resolve(rung)]
    cfg = copy.deepcopy(base_cfg)
    if spec["d"] is not None:
        cfg["model"]["d"] = spec["d"]
    cfg["model"].update(spec["model"])
    cfg["train"].update(spec["train"])
    cfg["data"].update(spec["data"])
    cfg["train"]["run_name"] = run_name
    cfg["train"]["seed"] = seed
    cfg["train"]["total_steps"] = total_steps
    return cfg


def count_params(cfg: dict) -> int:
    """その config でモデルを 1 度組んでパラメータ数を数える（データは要らない）。"""
    from predvla.model import PredVLA
    m = PredVLA(cfg)
    return sum(p.numel() for p in m.parameters())


def config_path(root: str, rung: str, suite_short: str, seed: int) -> str:
    return os.path.join(root, "configs", "ladder",
                        f"{resolve(rung)}_{suite_short}_s{seed}.yaml")
