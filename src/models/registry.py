"""`cfg["model"]["type"]` からモデルを組む 1 か所。

★旧名を受けるのが要点。同梱している学習済み ckpt の `cfg` には
`type: bcrnn` / `type: bctf` と書かれている（それらを学習した当時の表記）。
ckpt を作り直さずに新しい呼び名へ移るため、ここで正規化する。
state_dict のキーはクラス名に依らないので、クラスを改名しても既存 ckpt は読める。
"""
from __future__ import annotations

# 表記ゆれ -> 正規名
ALIASES = {
    # BC-LSTM（1 層 LSTM に [v; q; lang] を連結する robomimic 型のベースライン）
    "bc_lstm": "bc_lstm", "bclstm": "bc_lstm", "bc-lstm": "bc_lstm",
    "bcrnn": "bc_lstm", "bc_rnn": "bc_lstm", "bc-rnn": "bc_lstm",
    # BC-Transformer（因果 Transformer のベースライン）
    "bc_transformer": "bc_transformer", "bctransformer": "bc_transformer",
    "bc-transformer": "bc_transformer", "bctf": "bc_transformer",
    "bc_tf": "bc_transformer", "bc-tf": "bc_transformer",
    # MT-RNN（BC-LSTM の LSTM を 2 層 leaky RNN に替えた追加対照）
    "mtrnn": "mtrnn", "mt_rnn": "mtrnn", "mt-rnn": "mtrnn",
    # 旧世代の予測符号化エージェント（benchmark/ 側にだけ残っている）
    "legacy_pc": "legacy_pc", "legacy": "legacy_pc", "predvla": "legacy_pc",
}

# 正規名 -> 人間向けの表示名
DISPLAY = {
    "bc_lstm": "BC-LSTM",
    "bc_transformer": "BC-Transformer",
    "mtrnn": "MT-RNN",
    "legacy_pc": "LegacyPCAgent",
}


def normalize(model_type: str) -> str:
    """cfg に書かれた型名を正規名に直す。未知なら例外。"""
    key = str(model_type).strip().lower()
    if key not in ALIASES:
        raise KeyError(
            f"未知の model.type: {model_type!r}。"
            f"使えるのは {sorted(set(ALIASES.values()))}")
    return ALIASES[key]


def build(cfg: dict):
    """cfg からモデルを 1 つ組んで返す（device への移動は呼び手の責任）。"""
    kind = normalize(cfg["model"].get("type", "legacy_pc"))
    if kind == "bc_lstm":
        from src.models.baseline_bc_lstm import BCLSTM
        return BCLSTM(cfg)
    if kind == "bc_transformer":
        from src.models.baseline_bc_transformer import BCTransformer
        return BCTransformer(cfg)
    if kind == "mtrnn":
        from src.models.baseline_mtrnn import BaselineMTRNN
        return BaselineMTRNN(cfg)
    from src.models.legacy_agent import LegacyPCAgent
    return LegacyPCAgent(cfg)
