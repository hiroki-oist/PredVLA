"""バッチ ER での評価。B 本のロールアウトを 1 プロセスで同時に回す。2026-08-03。

既存 eval.py との違いは **ERBatch を使うこと**だけ。判定も出力も揃えてある:
  成功判定    env.step の rew > 0（eval.py と同じ）
  行動        clip せずそのまま送る（eval.py と同じ）
  初期化      set_init_state → 無操作 5 step で settle（LIBERO 標準）
  出力        "  task N success S/T = P%" と "全体" の行（既存の集計が読める）

★結果は eval.py と一致しない
  B が変わると BLAS が別のカーネルを選び加算順序が変わる。fp64 でも step 0 で
  1e-15、10 step で 1e-6 まで育つ（閉ループ + ER の内側最適化は正の Lyapunov
  指数を持つ）。**軌道は原理的に一致しない。**
  一致するのは「同じ状態を与えたときの勾配」で、B=4 で 1.2e-14 を確認済み
  。
  したがって **B は実験条件の一部**であり、参照側も同じ B で測り直す必要がある。

枠の割り当て
  1 枠 = 1 タスク（env を作りっぱなしにできる。言語ベクトルも変わらない）。
  B > タスク数のときは 1 タスクに複数枠を割り当て、ロールアウトを分担する。
  枠は自分の担当ぶんを消化したら遊ぶ（末尾だけ無駄計算が出る）。

使い方:
  OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_batch.py \
      --ckpt results/predvla_spatial_s1/step_30000.pt --n 5 --B 16 \
      --window 40 --n-itr 20 --er-w 1.0 --er-lr 0.1 --compile --tag b16_sp_s0
"""
import argparse
import glob
import os
import sys

# torch.compile(inductor) のメモリ計画が各ノードを再帰で辿るため、誤差/観測を
# 前向きに流す変種(err_to_low / obs_to_low)では既定の上限 1000 を超えて
# RecursionError になる。train.py と同じ対処(2026-08-18 追加)。
sys.setrecursionlimit(50000)
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # リポジトリ直下
_ROOT = HERE
sys.path.insert(0, _ROOT)
sys.path.insert(0, HERE)

from src.utils import compat  # noqa: E402
compat.apply()

from predvla.er_batch import ERBatch    # noqa: E402
from predvla.frontend import Frontend   # noqa: E402
from predvla.model import PredVLA        # noqa: E402


def _cache_root(cfg, args):
    """ckpt に保存された cache_root を本リポジトリの階層に合わせて解決する。

    元研究リポジトリでは学習スクリプトが 1 段下にあったので、公開 ckpt の cfg には
    "../data/cache_l64" のように 1 段上る相対パスが保存されている。本リポジトリでは
    リポジトリ直下に data/ を置くので、先頭の "../" を剥がす。
    --cache-root を渡せば cfg を無視して差し替えられる（別基底での再現用）。

    ★論文の本線は PCA 基底がスイートで違う（取り違えると数値が再現しない）:
        spatial       data/cache_ps_sp    スイート個別 PCA
        goal, object  data/cache_l64      3 スイート共有 PCA
        long          data/cache_l64_lg   Long で張り直した PCA
    """
    cr = getattr(args, "cache_root", None) or cfg["paths"]["cache_root"]
    while cr.startswith("../"):
        cr = cr[3:]
    return cr if os.path.isabs(cr) else os.path.join(_ROOT, cr)



def demo_inits(data_root, suite_name, task):
    """学習に使ったデモの初期状態を返す。

    ★初期状態だけを抜いた npz（`data/{suite}_init_states.npz`、187KB）があれば
      そちらを使う。生デモ hdf5 は 13GB あって共用機に置いていないことがあり、
      必要なのは init_state（50 × 45〜123 の配列）だけなので 7 万分の 1 で済む
      （2026-08-11、ユーザー指摘）。生成は `scripts/dump_init_states.py`。
    """
    stem = task.bddl_file.replace(".bddl", "")
    npz = os.path.join(_ROOT, "data", f"{suite_name}_init_states.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        hit = [k for k in z.files if stem in k]
        assert hit, f"init npz に {stem} が無い（{npz}）"
        return z[hit[0]]
    import h5py
    cand = [p for p in sorted(glob.glob(os.path.join(data_root, suite_name,
                                                     "*.hdf5")))
            if stem in os.path.basename(p)]
    assert cand, (f"デモ hdf5 も init npz も無い: {stem}\n"
                  f"  → scripts/dump_init_states.py で {npz} を作る")
    with h5py.File(cand[0], "r") as h:
        keys = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))
        return np.stack([np.asarray(h[f"data/{k}"].attrs["init_state"])
                         for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", default=None,
                    help="ckpt の cfg にある cache_root を上書きする"
                         "（既定は ckpt の値。先頭の ../ は自動で剥がす）")
    ap.add_argument("--n", type=int, default=5, help="1 タスクあたりのロールアウト数")
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--B", type=int, default=16, help="同時に走らせる枠の数")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--n-itr", type=int, default=20)
    ap.add_argument("--er-lr", type=float, default=0.1)
    ap.add_argument("--er-w", type=float, default=1.0)
    # ★対照 A（2026-08-13）: ER の Complexity 重みを**学習時と同じ** model.w に戻す。
    #   既定の --er-w 1.0 は全層を 1.0 で上書きするので、学習時の
    #   {t 0.05, v 0.02, up 0.02, low 0.01} の 20〜100 倍の罰則がかかっている。
    #   本線ではそれで +10.1pt 出ているが、ER をループに通して学習したモデル
    #   （ertwarm / ert）では学習時の罰則スケールに特化している可能性があり、
    #   ER 成功率 0% の原因候補。既定を変えると既存結果の再現性が崩れるので
    #   明示フラグにする（付けないときの挙動は完全に従来と同一）。
    ap.add_argument("--er-w-train", action="store_true",
                    help="ER の Complexity 重みを学習時の model.w にする")
    # ★2026-08-15（ユーザー指示）: 学習時 w の**比率を保ったまま**大きさだけ変える。
    #   --er-w はスカラで全層を同じ値に潰すので、層ごとの重み付け
    #   （t 0.05 : v 0.02 : up 0.02 : low 0.01 = 5:2:2:1）が消えていた。
    #   学習時 w の平均は 0.025 なので --er-w-scale 40 で★平均 1.0・比率そのまま
    #   （t 2.0 / v 0.8 / up 0.8 / low 0.4）になる。--er-w-train は scale 1.0 と同じ。
    ap.add_argument("--er-w-scale", type=float, default=None,
                    help="ER の Complexity 重み = 学習時 model.w × この倍率（比率保存）")
    ap.add_argument("--er-lambda-v", type=float, default=1.0)
    # ★P0a（2026-08-12）: 視覚誤差から固有感覚と冗長な部分空間を落とす。
    #   scripts/build_q_subspace.py が作る npz（U を含む）を渡す。
    ap.add_argument("--er-v-proj", default=None,
                    help="腕部分空間 U の npz。誤差から U 成分を落として ER する")
    ap.add_argument("--er-lambda-q", type=float, default=1.0)
    ap.add_argument("--a-init", default="prior", choices=["prior", "zero"])
    ap.add_argument("--fresh-c", action="store_true",
                    help="記憶なし ER: 毎制御 step、窓内全 slot の c を prior に"
                         "貼り直してから n_itr 回推論する")
    ap.add_argument("--er-freeze", nargs="*", default=[],
                    choices=["t", "v", "up", "low"],
                    help="ER で更新しない層（例: --er-freeze v で視覚の c を凍結）")
    ap.add_argument("--er-opt", default="adam", choices=["adam", "sgd"],
                    help="ER の最適化器。sgd = 生勾配（J_o 行空間に閉じる）")
    ap.add_argument("--er-eps", type=float, default=1e-4,
                    help="Adam の eps（大きいほど比例領域 = 符号SGDから遠ざかる）")
    ap.add_argument("--er-obs-roll", type=int, default=0, metavar="K",
                    help="★観測差し替え: ER に渡す (v, q, mv) を稼働中スロット間で K だけ "
                         "巡回させる。摂動の大きさ・視覚更新の回数は保ったまま中身だけ "
                         "別エピソード（別タスク・別シーン）のものになる。0 で無効。"
                         "A6/A7/A5 の 2x2 に対する対抗仮説『利得は誤差の中身ではなく "
                         "c が prior から動くこと自体』を潰すための対照（2026-08-14）")
    ap.add_argument("--er-obs-roll-what", default="both", choices=["both", "v", "q"],
                    help="差し替える対象。both = v と q の両方（既定）／v = 視覚だけ嘘に "
                         "する（q は正しい。★実機シナリオの proxy: エンコーダは正確で "
                         "凍結 PCA 基底の v だけが もっともらしく外れる）／q = 固有感覚だけ。"
                         "A6/A7 の『黙らせる』に対し、これは『嘘をつかせる』条件")
    ap.add_argument("--er-obs-roll-alpha", type=float, default=1.0, metavar="A",
                    help="★段階的な嘘。差し替えを混合率 A で入れる: x ← A·x_other + (1−A)·x_self。"
                         "A=0 で無効（表①と同一）、A=1 で完全置換。基底ずれのような『もっともらしい "
                         "嘘』を作って、成功率が落ちるのに eps が上がらない危険領域を探す。"
                         "★この経路では mv（視覚の更新タイミング）はスロット自身のものを保つので、"
                         "A=0 が厳密に表①に一致する（--er-obs-roll-what の非 alpha 経路とは違う）")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-steps-auto", type=float, default=None, metavar="FACTOR",
                    help="タスクごとの打ち切りを『そのタスクの最長デモ × FACTOR』に"
                         "する(例 1.2)。--max-steps より優先。デモ長はキャッシュの "
                         "ep_bounds から読む")
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda / mps / auto。★既定 cpu のまま論文の数値が出る"
                         "（B=16 の rollout は CPU が速く、GPU は描画に使う）")
    ap.add_argument("--tag", default="")
    # ★学習に使ったデモの初期状態から始める（汎化か記憶かの切り分け。
    #   eval_single.py と同じ実装を 2026-08-11 に移植）
    ap.add_argument("--init-source", default="eval", choices=["eval", "demo"],
                    help="eval=LIBERO 標準の 50 個 / demo=学習デモの初期状態")
    ap.add_argument("--data-root", default="data/libero",
                    help="--init-source demo のときに読むデモ hdf5 の場所")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--record-e", default=None, metavar="PATH",
                    help="★ER の内側反復ごとの自由エネルギーを PATH(.npz) に保存する。\n"
                         "診断用。既定 None のときは従来と完全に同一の挙動。")
    args = ap.parse_args()
    args.device = compat.pick_device(args.device)   # ★本リポジトリ固有
    if args.compile and args.device != "cuda":
        # inductor の reduce-overhead は CUDA graphs 前提。CPU では利得が無く
        # 環境によっては落ちるので黙って切る（★数値は変わらない）。
        print("[eval] device が cuda でないので --compile を無視する", flush=True)
        args.compile = False

    torch.set_num_threads(1)
    if args.compile:
        # grad_mode（no_grad か否か）× want_states で 4 通りに特殊化される。
        # 既定上限 8 は余裕が無く、超えると黙って eager に落ちる（実測で踏んだ）。
        import torch._dynamo as _dyn
        _dyn.config.cache_size_limit = 32

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PredVLA(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stride = cfg["data"]["vision_stride"]
    fe = Frontend(_cache_root(cfg, args), device=args.device)
    # ★オラクル v のモデル(v = 物体位置)は、評価時も v を画像ではなく env の状態から作る。
    #   cache_root に oracle_v_proj.npz があれば自動でその経路に切り替える(2026-08-11)。
    _ovp = os.path.join(_cache_root(cfg, args), "oracle_v_proj.npz")
    ORACLE_V = None
    if os.path.exists(_ovp):
        sys.path.insert(0, os.path.join(_ROOT, "scripts"))
        from build_oracle_v_cache import slot_positions as _slotpos
        _z = np.load(_ovp, allow_pickle=True)
        ORACLE_V = {"A": _z["A"], "mu": _z["mu"], "sd": _z["sd"], "fn": _slotpos,
                    "slots": [str(x) for x in _z["slots"]]}
        print("  ★オラクル v: v を env の物体位置から作る(画像は使わない)")

    def oracle_v(env, obs):
        x = ORACLE_V["fn"](env, obs, ORACLE_V["slots"])                       # (28, 3)
        pres = (np.abs(x).sum(-1) > 1e-9)[..., None]
        z = np.where(pres, (x - ORACLE_V["mu"]) / ORACLE_V["sd"], 0.0).reshape(-1)
        return torch.from_numpy((z @ ORACLE_V["A"]).astype(np.float32))
    suite = benchmark.get_benchmark_dict()[cfg["data"]["suite"]]()
    tids = args.tasks if args.tasks else list(range(suite.n_tasks))
    B = args.B

    print(f"ckpt {args.ckpt}  step {ck['step']}  suite {cfg['data']['suite']}")
    print(f"  バッチ ER  B {B}  窓 {args.window}  n_itr {args.n_itr}  "
          f"er_lr {args.er_lr}  "
          f"er_w {f'★学習時 model.w ×{args.er_w_scale:g} = ' + ' '.join(f'{k}{v * args.er_w_scale:.3g}' for k, v in model.w.items()) if args.er_w_scale is not None else ('★学習時の model.w' if args.er_w_train else args.er_w)}  "
          f"λv {args.er_lambda_v}  λq {args.er_lambda_q}  "
          f"compile {'入' if args.compile else '切'}"
          f"{'  ★fresh_c(記憶なしER)' if args.fresh_c else ''}"
          f"{'  ★er_freeze=' + ','.join(args.er_freeze) if args.er_freeze else ''}"
          f"{f'  ★観測差し替え roll={args.er_obs_roll} 対象={args.er_obs_roll_what}' if args.er_obs_roll else ''}")
    print(f"  タスク {len(tids)} × ロールアウト {args.n} = {len(tids)*args.n} エピソード")

    # ★n を初期状態の数で制限する（2026-08-03 14:00 修正）
    #   LIBERO は 1 タスク 50 個の初期状態しか用意していない（4 スイートすべて）。
    #   eval.py には min(args.n, len(inits)) があるが eval_batch.py に無く、
    #   n=100 を渡すと inits[i][50] で IndexError になっていた。
    #   50 × 10 タスク = 500 エピソード / スイートが文献の標準規模。
    # ★タスクごとの打ち切り(2026-08-07、ユーザー指示)
    #   固定 600 step だと短いタスクに無駄な猶予を与え、長いタスクを切ってしまう。
    #   キャッシュの ep_bounds から「そのタスクの最長デモ × FACTOR」を上限にする。
    max_steps_of = {t: args.max_steps for t in tids}
    if args.max_steps_auto:
        import glob as _glob
        import h5py as _h5
        croot = os.path.join(_cache_root(cfg, args), cfg["data"]["suite"])
        for fp in sorted(_glob.glob(os.path.join(croot, "*.h5"))):
            with _h5.File(fp, "r") as h:
                tid = int(h.attrs["task_id"])
                if tid not in max_steps_of:
                    continue
                ln = np.diff(h["ep_bounds"][:])
            max_steps_of[tid] = int(np.ceil(ln.max() * args.max_steps_auto))
        print("  タスク別の打ち切り: "
              + " ".join(f"t{t}:{max_steps_of[t]}" for t in tids))

    if args.init_source == "demo":
        n_avail = min(len(demo_inits(os.path.join(_ROOT, args.data_root),
                                     cfg["data"]["suite"], suite.get_task(t)))
                      for t in tids)
        print("  ★初期状態は学習デモのもの（--init-source demo）")
    else:
        n_avail = min(len(suite.get_task_init_states(t)) for t in tids)
    if args.n > n_avail:
        print(f"  ★--n {args.n} は初期状態の数 {n_avail} を超えるので "
              f"{n_avail} に切り下げる")
        args.n = n_avail

    # ---- 枠へのタスク割り当て（1 枠 = 1 タスク。B > タスク数なら分担）----
    slot_task = [tids[i % len(tids)] for i in range(B)]
    slot_eps = [[] for _ in range(B)]
    for tid in tids:
        ss = [i for i in range(B) if slot_task[i] == tid]
        for r in range(args.n):
            slot_eps[ss[r % len(ss)]].append(r)

    envs, inits, lvecs = [], [], []
    t_setup = time.time()
    for i in range(B):
        task = suite.get_task(slot_task[i])
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                            task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128,
                               camera_widths=128)
        e.seed(0)
        envs.append(e)
        if args.init_source == "demo":
            inits.append(demo_inits(os.path.join(_ROOT, args.data_root),
                                    cfg["data"]["suite"], task))
        else:
            inits.append(suite.get_task_init_states(slot_task[i]))
        lvecs.append(fe.lang(task.language))
    print(f"  env {B} 個の構築 {time.time()-t_setup:.1f} 秒")

    _VU = None
    if args.er_v_proj:
        _z = np.load(args.er_v_proj)
        _VU = torch.from_numpy(_z["U"].astype(np.float32))
        print(f"  ★P0a: 視覚誤差から腕部分空間 {int(_z['rank'])} 次元を落とす"
              f"（v の分散の {float(_z['var_frac']):.1%}）", flush=True)

    er = ERBatch(model, torch.stack(lvecs), window=args.window,
                 n_itr=args.n_itr, lr=args.er_lr, device=args.device,
                 a_init=args.a_init, vision_stride=stride,
                 er_w=({k: v * args.er_w_scale for k, v in model.w.items()}
                       if args.er_w_scale is not None
                       else (None if args.er_w_train else args.er_w)),
                 er_lambda_v=args.er_lambda_v, er_lambda_q=args.er_lambda_q,
                 fresh_c=args.fresh_c, er_freeze=args.er_freeze,
                 er_opt=args.er_opt, er_eps=args.er_eps,
                 er_v_proj=_VU)
    if args.record_e:
        er.record_E = True
        print(f"  ★ER の内側反復ごとの自由エネルギーを記録する → {args.record_e}", flush=True)
    if args.compile:
        er._roll = types.MethodType(
            torch.compile(ERBatch._roll, dynamic=False), er)

    zero7 = np.zeros(model.a_dim, np.float32)
    obs = [None] * B
    ptr = [0] * B
    slot_t = [0] * B
    active = [False] * B
    succ = {t: 0 for t in tids}
    ran = {t: 0 for t in tids}

    def start(i):
        if ptr[i] >= len(slot_eps[i]):
            return False
        r = slot_eps[i][ptr[i]]; ptr[i] += 1
        envs[i].reset()
        o = envs[i].set_init_state(inits[i][r])
        for _ in range(5):
            o, _, _, _ = envs[i].step(zero7)
        obs[i] = o; slot_t[i] = 0
        er.reset_slot(i)
        return True

    for i in range(B):
        active[i] = start(i)

    t0 = time.time()
    n_ctrl = 0
    # ★予測誤差の記録（2026-08-14）。「成功率が落ちるのに eps が上がらない」危険領域を
    #   探すために、成功率と同じ実行で eps を取る。ERBatch.step が毎 step 更新している。
    #   稼働中スロットだけを平均する（停止枠の残留値を混ぜない）。
    eps_sum = {"v": 0.0, "q": 0.0}; eps_n = 0
    while any(active):
        v = torch.zeros(B, model.v_dim, device=args.device)
        q = torch.zeros(B, model.q_dim, device=args.device)
        mv = torch.zeros(B, device=args.device)
        for i in range(B):
            if not active[i]:
                continue
            q[i] = fe.q(obs[i])
            if slot_t[i] % stride == 0:
                if ORACLE_V is not None:
                    v[i] = oracle_v(envs[i], obs[i]).to(args.device)
                else:
                    v[i] = fe.v(obs[i]["agentview_image"],
                            obs[i]["robot0_eye_in_hand_image"])
                mv[i] = 1.0
        if args.er_obs_roll:
            # ★観測差し替え。稼働中スロットだけを巡回させる（停止枠のゼロを混ぜない）。
            #   (v, q, mv) を三つ組で動かすので、受け取る側から見れば「別エピソードの
            #   観測ストリームが丸ごと来ている」状態になる。置換なので視覚更新の総回数は不変。
            act = [i for i in range(B) if active[i]]
            if len(act) >= 2:
                k = args.er_obs_roll % len(act)
                if k:
                    src = act[-k:] + act[:-k]          # dst act[j] <- src[j]
                    w8 = args.er_obs_roll_what
                    al = args.er_obs_roll_alpha
                    if al >= 1.0:
                        # 完全置換。mv も視覚チャネルの一部として一緒に動かす
                        if w8 in ("both", "v"):
                            vs, ms = v[src].clone(), mv[src].clone()
                            v = v.clone(); mv = mv.clone()
                            v[act] = vs; mv[act] = ms
                        if w8 in ("both", "q"):
                            qs = q[src].clone()
                            q = q.clone(); q[act] = qs
                    elif al > 0.0:
                        # ★段階的。値だけ混ぜ、mv は自分のものを保つ（A=0 が厳密に表①になる）
                        if w8 in ("both", "v"):
                            vs = v[src].clone()
                            v = v.clone(); v[act] = al * vs + (1.0 - al) * v[act]
                        if w8 in ("both", "q"):
                            qs = q[src].clone()
                            q = q.clone(); q[act] = al * qs + (1.0 - al) * q[act]
        a = er.step(v, q, mv)
        n_ctrl += 1
        if getattr(er, "last_eps_v", None) is not None:
            am = torch.tensor(active, device=er.last_eps_v.device)
            na = int(am.sum())
            if na:
                eps_sum["v"] += float(er.last_eps_v[am].sum())
                eps_sum["q"] += float(er.last_eps_q[am].sum())
                eps_n += na
        for i in range(B):
            if not active[i]:
                continue
            o, rew, done, _ = envs[i].step(np.asarray(a[i], np.float32))
            obs[i] = o; slot_t[i] += 1
            fin = (rew > 0) or (slot_t[i] >= max_steps_of[slot_task[i]])
            if fin:
                tid = slot_task[i]
                ran[tid] += 1
                succ[tid] += int(rew > 0)
                active[i] = start(i)
        if n_ctrl % 50 == 0:
            # 途中経過に成功数も出す（2026-08-03 12:53、ユーザー要望）。
            # 表示だけの変更で、成功判定にも行動にも影響しない。
            done_n = sum(ran.values()); tot = len(tids) * args.n
            sc = sum(succ.values())
            rate = 100.0 * sc / done_n if done_n else 0.0
            # タスクごとの最新成功率も出す（2026-08-03 12:55、ユーザー要望）。
            # 評価中のモデルがどのタスクで詰まっているかを走行中に見るため。
            per = " ".join(f"t{t}:{succ[t]}/{ran[t]}" for t in tids if ran[t] > 0)
            print(f"  [{time.time()-t0:7.0f}s] 制御 step {n_ctrl}  "
                  f"完了 {done_n}/{tot}  成功 {sc}/{done_n} = {rate:5.1f}%  "
                  f"稼働枠 {sum(active)}/{B}", flush=True)
            if per:
                print(f"      タスク別 {per}", flush=True)

    el = time.time() - t0
    print()
    for tid in tids:
        print(f"  task {tid:2d} success {succ[tid]:2d}/{ran[tid]:<2d} = "
              f"{100.0*succ[tid]/max(ran[tid],1):5.1f}%  "
              f"[{suite.get_task(tid).language[:40]}]")
    S = sum(succ.values()); N = sum(ran.values())
    print(f"\n  全体 {100.0*S/max(N,1):.2f}%  ({S}/{N})")
    if args.record_e and er.E_trace:
        import numpy as _np
        _tr = _np.array(er.E_trace, dtype=_np.float64)   # (制御step, 反復i, E)
        _np.savez_compressed(args.record_e, trace=_tr, n_itr=args.n_itr,
                             er_opt=args.er_opt, er_lr=args.er_lr,
                             ckpt=args.ckpt)
        print(f"  ★E の軌跡 {len(er.E_trace)} 点を {args.record_e} に保存した"
              f"（制御 step {int(_tr[:,0].max())+1} 本 × 反復 {args.n_itr}+1）", flush=True)
    if eps_n:
        ev, eq = eps_sum["v"] / eps_n, eps_sum["q"] / eps_n
        print(f"  ★eps  v {ev:.4f}  q {eq:.4f}  比 v/q {ev/max(eq,1e-9):.1f}  "
              f"（稼働スロット平均、標本 {eps_n}）")
    print(f"  実時間 {el:.0f} 秒  制御 step {n_ctrl}  "
          f"1 エピソードあたり {el/max(N,1):.1f} 秒")


if __name__ == "__main__":
    main()
