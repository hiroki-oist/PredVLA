"""評価プロトコルの定数を 1 か所に集めたもの。

元リポジトリではこれらがキュースクリプト 100 本以上に散っていて、
「どの数値がどの設定で出たか」を追うのに毎回ログを掘る必要があった。
論文の数値はすべてここに書いてある組み合わせで出ている。

★スイートごとに違う点が 3 つあるので、共通化できるのはここまで。
  1. PCA 基底（cache_root）  3 スイートは共通、long だけ張り直し（下の注記を読む）
  2. 時定数 tau              long だけ広い（16/8/5/2 → 30/14/8/2）
  3. ロールアウト数と打ち切り  long だけ 5/タスク・可変打ち切り（3 スイートは 50/タスク・600 固定）
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# スイート
# --------------------------------------------------------------------------
SUITES_MAIN = ("libero_spatial", "libero_goal", "libero_object")
SUITE_LONG = "libero_10"
ALL_SUITES = SUITES_MAIN + (SUITE_LONG,)

SHORT = {"libero_spatial": "spatial", "libero_goal": "goal",
         "libero_object": "object", "libero_10": "long"}

# ★PCA 基底はスイート群で違う。取り違えると数値が再現しない。
#   2026-08-22 に実キャッシュを突き合わせて確かめた実態（以前の注記は不正確だった）:
#
#   cache_l64     視覚 PCA と q の正規化統計を **libero_spatial のデモだけで** 当て、
#                 それを spatial / goal / object の符号化に使い回している
#                 （= 3 スイート共通の基底。言語 PCA は全ベンチマークの指示文で当てる）
#   cache_ps_sp   上と **同じ手順をもう一度** 回したもの。視覚基底と正規化統計は
#                 cache_l64 とバイト単位で同一、言語 PCA だけ再当てはめのぶん 5e-5
#                 ずれる。spatial の特徴の差は最大 2e-6（float32 の丸め相当）。
#                 → 実質は同じもの。論文が spatial にこれを使っているので残す。
#   cache_l64_lg  視覚 PCA と正規化統計を **libero_10 のデモで張り直した** もの。
#                 これは本当に違う（基底の最大差 0.14）。Long 専用。
#
#   つまり「張り直しが要るのは Long だけ」。作り方は tools/prepare_data.py にある。
CACHE_ROOT = {
    "libero_spatial": "data/cache_ps_sp",
    "libero_goal":    "data/cache_l64",
    "libero_object":  "data/cache_l64",
    "libero_10":      "data/cache_l64_lg",
}

# --------------------------------------------------------------------------
# 学習
# --------------------------------------------------------------------------
TRAIN_STEPS = 30_000
TRAIN_BATCH = 32
LR_W = 6.0e-5          # 重み。3e-5 と 1.2e-4 の両側で悪化することを確認済み
LR_C = 1.0e-2          # 自由変数 c
LR_SCHEDULE = "none"   # ★定数。cosine だと「飽和した」という誤った結論が出る
SEQ_LEN = {"libero_spatial": 200, "libero_goal": 200,
           "libero_object": 200, "libero_10": 500}

# 論文のシード。s0 は歴史的経緯で除外し s1 から数える（元リポジトリの慣行）。
SEEDS_MAIN = tuple(range(1, 15))   # 表①: 14 シード
SEEDS_ABLATION = tuple(range(0, 7))  # 表③: 7 シード
SEEDS_LONG = tuple(range(1, 8))    # Long: 7 シード

# --------------------------------------------------------------------------
# 評価（ER）
# --------------------------------------------------------------------------
# ★確定プロトコル（2026-08-21 に 182 セル全部で確定した主表 = 表①）:
#     Adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / 窓 40
#   これが論文の本線。以下の MAIN が同じものを指す。
WINDOW = 40

MAIN = dict(n_itr=10, er_opt="adam", er_lr=0.05, er_w=1.0)

# 参考: 旧看板（SGD / n_itr=10 / er_lr=0.1）。推論計算が半分で、主表と統計的に
#   区別できない（42 シードペアで Δ = −0.40 ± 3.75, t = −0.70）。
#   ただし ★アブレーションと組み合わせると壊れる例がある: A3（efference なし）は
#   SGD10 でだけ 1.00 まで崩壊する（無傷のモデルはどの設定でも 82〜83）。
#   したがってアブレーションでこれを使ってはいけない。
SGD10 = dict(n_itr=10, er_opt="sgd", er_lr=0.1, er_w=1.0)

# ER を回さない（A6 = 開ループ）。n_itr=0 のときは er_batch が er_w も最適化器も
#   参照しないので、★この行の値はプロトコル変更に対して不変である。
NI0 = dict(n_itr=0, er_opt="adam", er_lr=0.05, er_w=1.0)

# 評価プロトコルの名前 -> 中身。CLI の --protocol はこの名前で受ける。
PROTOCOLS = {"main": MAIN, "sgd10": SGD10, "ni0": NI0}

# 旧名（後方互換。新しいコードは MAIN / SGD10 / NI0 を使う）
ER_TABLE1 = MAIN
ER_TABLE1_PRIME = SGD10
ER_NI0 = NI0

# ER の Complexity 重み。学習時の model.w（層別 0.05/0.02/0.02/0.01）ではなく
# テスト時は全層一律の値で上書きする。★確定プロトコルは全スイート 1.0。
#   掃引の履歴: 一時期 Long は 0.8〜0.9 が良いと見えたが、n を増やすと消えた。
#   2026-08-22 に Long で er_w を 0.01〜1.5 まで振り直した結果、1.0 が頂点で
#   下げると単調に悪化（0.01 で −26.7pt）、1.2 は n=3 で +2.7 に見えたが n=6 で
#   −0.4 に転じた。したがって 1.0 を動かす根拠は無い。
ER_W = 1.0
ER_W_DEFAULT_MAIN = ER_W    # 旧名
ER_W_BEST_LONG = ER_W       # 旧名（★0.8 ではない。上の注記を読む）

# ロールアウト
ROLLOUTS_MAIN = 50        # 1 タスクあたり → 10 タスクで 500 エピソード
ROLLOUTS_LONG = 5         # ★Long は 1/10（50 エピソード）。表に注記が要る
MAX_STEPS_MAIN = 600
# Long は固定 600 ではなく「そのタスクの最長デモ × 1.2」で打ち切る。
MAX_STEPS_AUTO_LONG = 1.2
# その実測値（data/cache_l64_lg の ep_bounds から算出）。ベースライン側の
# benchmark/eval_suite.py は --max-steps-auto を持たないので、この表を直接渡す。
MAX_STEPS_LONG_PER_TASK = {0: 466, 1: 387, 2: 408, 3: 381, 4: 398,
                           5: 311, 6: 411, 7: 401, 8: 621, 9: 539}

# ★libero_10 は 1 プロセス 1 タスクが必須。
#   複数の OffScreenRenderEnv を 1 プロセスに持つと描画コンテキストを共有し、
#   最後に reset した env 以外が別シーンのカメラ姿勢で描かれる。
#   libero_10 は 10 タスクが 9 シーンにまたがるので影響が出る。
#   単一シーンの spatial/goal/object は影響しない。
#   これを踏むと 2026-08-10 に撤回した結果と同じ間違いをする。
ONE_PROCESS_PER_TASK = ("libero_10",)


def requires_task_split(suite: str) -> bool:
    return suite in ONE_PROCESS_PER_TASK


# --------------------------------------------------------------------------
# アブレーション（表③）
# --------------------------------------------------------------------------
# 学習時 (train): 該当フラグを付けて再学習が要る
# テスト時 (test): 既存 ckpt に評価フラグを足すだけ。再学習は不要
#
# ★ID は論文の表③ に合わせる。A1〜A3 が機構を引く（再学習）、A4〜A6 は ER の
#   目的関数を触るだけ（再学習不要）。
ABLATIONS = {
    # A1: 全層の τ を平均値 8 に統一する。τ=1 固定（ab_tau_flat）だと「階層は
    #     あるが全層が最速」なので、多時定数の必要性と「遅い層があること」を
    #     分離できない。16/8/5/2 の平均 7.75 → 8 に統一すれば「遅いが均一」との
    #     対照になり、多時定数そのものを問える。
    "A1_uniform_tau": dict(
        kind="train", set={"model.ab_tau_uniform": 8},
        desc="− 時間階層（全層の τ を 8 に統一）"),
    "A2_no_pb": dict(
        kind="train", set={"model.ab_no_pb": True},
        desc="− PB（V→A、W_pb·d^V 256→32）"),
    "A3_no_efference": dict(
        kind="train", set={"model.ab_no_av_bridge": True},
        desc="− efference copy（A→V、前 step の â と q̂ を V に渡す経路）"),
    "A4_no_er_vision": dict(
        kind="test", eval_flags={"--er-lambda-v": "0"},
        desc="ER 時のみ視覚予測誤差を切る（★学習時は使う）"),
    "A5_no_er_proprio": dict(
        kind="test", eval_flags={"--er-lambda-q": "0"},
        desc="ER 時のみ固有感覚予測誤差を切る（★学習時は使う）"),
    "A6_no_online_er": dict(
        kind="test", eval_flags={"--n-itr": "0"},
        desc="− オンライン ER（n_itr=0、開ループ生成のみ）"),
}

# 旧名（2026-08-16 版の名前。既存のログやスクリプトのため残す）
ABLATION_ALIASES = {
    "uniform_tau": "A1_uniform_tau",
    "no_pb_v2a": "A2_no_pb",
    "no_efference_a2v": "A3_no_efference",
    "no_er_vision": "A4_no_er_vision",
    "no_er_action": "A5_no_er_proprio",
    "no_online_er": "A6_no_online_er",
}


def resolve_ablation(name: str) -> str:
    """旧名でも新名でも、`A2` のような ID だけでも受ける。未知なら例外。

    論文の表③は `A1`〜`A6` で呼ぶので、CLI ではその短い形が自然に来る。
    """
    if name in ABLATIONS:
        return name
    if name in ABLATION_ALIASES:
        return ABLATION_ALIASES[name]
    key = name.strip().upper()
    hits = [k for k in ABLATIONS if k.upper().split("_")[0] == key]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(
        f"未知のアブレーション: {name}（使えるのは "
        f"{', '.join(k.split('_')[0] for k in ABLATIONS)} / {', '.join(ABLATIONS)}）")


# --------------------------------------------------------------------------
# 論文の参考値（再現できたかの照合用）
# --------------------------------------------------------------------------
# ★表① 主表（確定プロトコル MAIN = Adam n_itr10 / er_lr 0.05 / er_w 1.0 / 窓 40）
#   2026-08-21 に 182 セル全部で確定。全スイート n=14。
#   引用するときはここから引く。(平均, SD, n) の組。
MAIN_TABLE = {
    "libero_spatial": (83.19, 6.10, 14),
    "libero_goal":    (88.40, 3.83, 14),
    "libero_object":  (89.24, 6.13, 14),
    "libero_10":      (40.57, 6.44, 14),   # ★Long は 50 エピソード/シード（1/10）
}
# 4 スイート平均 75.35

# シード別（s1〜s14）。ロールアウトのラン間ノイズを見積もるのに使う。
MAIN_TABLE_SEEDS = {
    "libero_spatial": (81.6, 84.6, 83.8, 66.6, 84.2, 79.0, 81.8,
                       84.6, 86.8, 80.4, 83.4, 90.4, 93.6, 83.8),
    "libero_goal":    (84.8, 90.2, 88.2, 89.8, 90.6, 94.0, 78.8,
                       87.8, 84.0, 90.2, 92.4, 87.2, 90.0, 89.6),
    "libero_object":  (86.0, 91.2, 96.0, 95.6, 87.0, 95.6, 80.8,
                       96.4, 88.2, 89.0, 84.0, 76.0, 90.6, 93.0),
    "libero_10":      (46, 36, 28, 40, 52, 48, 36, 40, 38, 48, 38, 34, 40, 44),
}

# 旧名（平均だけ）
MAIN_REFERENCE = {k: v[0] for k, v in MAIN_TABLE.items()}

# ★旧看板（SGD10）での主表。プロトコルが違うので上と混ぜない。
SGD10_TABLE = {"libero_spatial": 82.49, "libero_goal": 87.20,
               "libero_object": 89.54, "libero_10": 41.43}

# ★アブレーションの参考値
#   A1〜A5 は **er_w=0.9 で取った旧値しか無く、確定プロトコル（er_w=1.0）では
#   未取得** である（2026-08-19 に全面やり直しを決め、主表の完成待ちで止まっている）。
#   したがって A1〜A5 に照合できる数値は無い。None を置いてそれを明示する。
#   A6 だけは n_itr=0 で er_w も最適化器も参照しないので★プロトコル不変で有効。
ABLATION_TABLE = {
    "A1_uniform_tau":   {"libero_spatial": None, "libero_goal": None,
                         "libero_object": None, "libero_10": None},
    "A2_no_pb":         {"libero_spatial": None, "libero_goal": None,
                         "libero_object": None, "libero_10": None},
    "A3_no_efference":  {"libero_spatial": None, "libero_goal": None,
                         "libero_object": None, "libero_10": None},
    "A4_no_er_vision":  {"libero_spatial": None, "libero_goal": None,
                         "libero_object": None, "libero_10": None},
    "A5_no_er_proprio": {"libero_spatial": None, "libero_goal": None,
                         "libero_object": None, "libero_10": None},
    # A6 のみ確定（プロトコル不変）。(平均, SD, n)
    "A6_no_online_er":  {"libero_spatial": (72.36, 6.00, 7),
                         "libero_goal":    (81.50, 3.67, 7),
                         "libero_object":  (77.93, 4.94, 7),
                         "libero_10":      (28.00, 6.62, 14)},
}

# 参考: er_w=0.9 で取った A1〜A5 の値（★論文には載せない。引用禁止）。
#   向きの目安としてだけ残す。Δ は同プロトコルの主表に対する差。
ABLATION_ERW09_NOT_FOR_CITATION = {
    "A1_uniform_tau":   {"libero_spatial": 61.25, "libero_goal": 73.08,
                         "libero_object": 60.08},
    "A2_no_pb":         {"libero_spatial": 57.58, "libero_goal": 74.70,
                         "libero_object": 92.08},
    "A3_no_efference":  {"libero_spatial": 74.08, "libero_goal": 76.71,
                         "libero_object": 89.14},
    "A4_no_er_vision":  {"libero_spatial": 79.10, "libero_goal": 81.14,
                         "libero_object": 88.71},
    "A5_no_er_proprio": {"libero_spatial": 78.50, "libero_goal": 87.14,
                         "libero_object": 74.29},
}

# 旧名（spatial だけ、旧看板 SGD10 プロトコルの値。★引用禁止）
ABLATION_REFERENCE_SPATIAL = {
    "A1_uniform_tau": None,
    "A3_no_efference": 70.21,
    "A2_no_pb": 52.00,
    "A4_no_er_vision": 79.97,
    "A5_no_er_proprio": 78.03,
    "A6_no_online_er": 72.36,
}


# --------------------------------------------------------------------------
# ベースライン
# --------------------------------------------------------------------------
# ★ベースラインは 50,000 step（PredVLA 本線は 30,000）。学習曲線が寝るまでの
#   step 数が違うだけで、比較の条件（凍結前段・特徴・スイート）は同一である。
BASELINE_STEPS = 50_000

# 正規名 -> 表示名。config は configs/baseline_<正規名>_<スイート短縮名>.yaml
BASELINES = {
    "bc_lstm": "BC-LSTM",
    "bc_transformer": "BC-Transformer",
}


# --------------------------------------------------------------------------
# 名前の付け方（★ここが唯一の出所）
# --------------------------------------------------------------------------
# 学習の run 名・ckpt のファイル名・評価ログのタグを、この 1 か所で決める。
#
#     <系列>_<スイート短縮名>_s<シード>
#
#   系列   predvla                    本線
#          bc_lstm / bc_transformer   ベースライン
#          ladder_<段>                はしご（L2〜L6, L2b, …）
#          <ID>_<何を落とすか>        アブレーション（A1〜A3。A4〜A6 は再学習しない）
#
# 例   predvla_spatial_s1 / bc_lstm_goal_s3 / ladder_L3_spatial_s1 / A2_no_pb_spatial_s0
#
# 学習の出力は results/<run 名>/step_<N>.pt、配布 ckpt は checkpoints/<run 名>.pt。
# scripts/evaluate.py はこの両方を探す。
def run_name(series: str, suite: str, seed: int) -> str:
    """系列・スイート・シードから run 名（= ckpt のファイル名の幹）を作る。"""
    if suite not in SHORT:
        raise KeyError(f"未知のスイート: {suite}")
    return f"{series}_{SHORT[suite]}_s{seed}"


def series_name(track: str, variant: str | None = None) -> str:
    """トラックと変種から系列名を作る。

    track: main / baseline / ladder / ablation
    variant: baseline なら bc_lstm|bc_transformer、ladder なら段 id、
             ablation なら A1〜A6。main では使わない。
    """
    if track == "main":
        return "predvla"
    if track == "baseline":
        if variant not in BASELINES:
            raise KeyError(f"未知のベースライン: {variant}（{list(BASELINES)}）")
        return variant
    if track == "ladder":
        return f"ladder_{variant}"
    if track == "ablation":
        return resolve_ablation(variant)
    raise KeyError(f"未知のトラック: {track}")


def train_steps(track: str) -> int:
    """そのトラックの学習 step 数。ベースラインだけ 50,000。"""
    return BASELINE_STEPS if track == "baseline" else TRAIN_STEPS
