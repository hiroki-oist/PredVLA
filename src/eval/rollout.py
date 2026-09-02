"""Minimal closed-loop rollout (Phase 1, §6.1 core) — the true M1 gate.

Drives the LIBERO simulator with the trained agent:
  * every `vision_stride` steps: agentview + eye_in_hand -> frozen ResNet-18 ->
    frozen PCA -> v_t (192). Held between vision updates.
  * each step: proprio q_t (z-scored) fed in, PB interpolated, action head emits
    a_hat which is sent to the env.
  * first `warmup` steps: send zero action (observe only) to initialize state.

Image/observation consistency with the training cache (critical, see M1.md):
  - hdf5 and live env both use robosuite IMAGE_CONVENTION 'opengl' -> no flip.
  - joint_states  <- obs['robot0_joint_pos'] (7)
    gripper_states <- obs['robot0_gripper_qpos'] (2)   (same order as cache q)
"""
import os
import sys

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from src.utils import compat
from src.data.pca import FrozenPCA
from src.models.encoders import FrozenResNet18, LanguageEncoder
from src.models import registry

compat.apply()


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PROJ, p)


class Rollout:
    """Closed-loop rollout. Optional test-time latent updates (predictive-coding
    inference; works on existing checkpoints, no retraining):

    Phase A (lu_k > 0): at each vision tick, refine the pre-step vision state
      h_v by K gradient steps on ||v_t - v_hat(h_v)||^2 + beta ||h_v - h_fwd||^2
      (trust region), i.e. filter the latent toward states that explain the
      current observation, then re-derive eps_v / PB from the refined state.

    Phase B (lu_pb_eta > 0): error regression of an additive PB correction
      delta: within each vision-stride window the action steps carry a graph
      w.r.t. delta; at the window end delta is updated by one gradient step on
      the accumulated proprio prediction error, then clamped (classic Tani-
      style parametric-bias regression, on the PB bottleneck only).
    """

    def __init__(self, ckpt_path: str, device: str = "cpu",
                 lu_k: int = 0, lu_eta: float = 0.05, lu_beta: float = 0.1,
                 lu_pb_eta: float = 0.0, lu_pb_clip: float = 1.0,
                 pc_rollout_k: int = -1,
                 pc_eta: float = -1.0,
                 action_scale: float = 1.0, scale_gripper: bool = False,
                 perturb: str = "none", perturb_at: int = 60, perturb_len: int = 10,
                 obs_noise: float = 0.0, act_delay: int = 0, perturb_seed: int = 0,
                 img_perturb: str = "none", img_strength: float = 0.0,
                 fe_terms: str = "both", pc_vars: str = "",
                 pc_delta_warm=None,
                 oracle_grip_at: float = 0.0, oracle_grip_key: str = ""):
        """pc_rollout_k: rollout 時の窓 PC 推論の反復数。-1 = 学習時設定のまま(既定)、
        0 = 切る、K>0 = 強制。「学習時に推論を unroll した重み」と「テスト時の推論手続き」
        のどちらが効いているかを分離する ablation 用(学習不要)。

        fe_terms: rollout 時の内部推論が最小化する Accuracy の構成。
          both(既定・従来と同一) / v = 視覚のみ / q = 固有感覚のみ。
        閉ループでの固有感覚予測は「動かない」と予測するより数十倍悪い(skill -5〜-75)
        にもかかわらず窓20 の Accuracy の 22.7% を占めるので、それを落として
        成功率が上がるかを見る(再学習不要)。

        pc_vars: 推論で動かす自由変数の上書き("" = ckpt の設定のまま)。
          both  = h_v0 と delta / hv = h_v0 のみ / delta = delta のみ
        delta は PB(視覚->行動のインターフェース)に足される自由オフセットで、内部状態
        ではない。どの h_v からも生成され得ない PB を作れてしまうため、
        「内部状態のみを推論する」条件(hv)と分けて測る必要がある。
        fe_terms=v も delta の勾配を 0 にするので delta=0 になるが、そちらは同時に
        h_v0 から固有感覚項も外してしまう。pc_vars=hv + fe_terms=both なら
        「delta を殺すこと」だけを単独で評価できる。"""
        assert fe_terms in ("both", "v", "q"), fe_terms
        assert pc_vars in ("", "both", "hv", "delta"), pc_vars
        self.fe_terms = fe_terms
        self._pc_vars_override = pc_vars
        # pc_delta_warm: 窓推論の delta を前 tick の値から始め、Complexity もそこからの
        # 変位を罰する。None なら ckpt の model.pc_delta_warm に従う(pcv2 系は true)。
        # 学習では推論が 1 窓 1 回なので効かず、rollout 専用の設定になる。
        self._pc_delta_warm_arg = pc_delta_warm
        self.lu_k = lu_k
        self.lu_eta = lu_eta
        self.lu_beta = lu_beta
        self.lu_pb_eta = lu_pb_eta
        self.lu_pb_clip = lu_pb_clip
        # map_location は必須。Mac(MPS) で保存した ckpt はテンソルのデバイスが mps に
        # なっており、指定しないと Ubuntu 側で
        #   RuntimeError: Storage device not recognized: mps
        # で落ちる（2026-08-01 に実際に踏んだ）。CPU に落としてから device へ移す。
        ck = torch.load(_abs(ckpt_path), map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        self.device = device
        self.agent = registry.build(self.cfg).to(device)
        self.agent.load_state_dict(ck["model"])
        self.agent.eval()
        # rollout 側の窓 PC 推論を上書き(-1 は学習時設定のまま)。pc_k は
        # rollout のゲート判定と pc_infer の既定反復数の両方を兼ねる。
        if pc_rollout_k >= 0 and hasattr(self.agent, "pc_k"):
            self.agent.pc_k = pc_rollout_k
        self.pc_rollout_k = pc_rollout_k
        # rollout 側の内側推論のステップ幅を上書き(-1 は学習時設定のまま)。
        # 学習時の既定 eta=0.1 では窓自由エネルギーが 5 step で 0.17% しか下がらず
        # (scripts/probe_pc_inference.py)、推論が実質何もしていない。摂動下で
        # 「状態を訂正できるか」を検定するには eta を実効域(1-10)に上げる必要がある。
        if pc_eta >= 0 and hasattr(self.agent, "pc_eta"):
            self.agent.pc_eta = pc_eta
        self.pc_eta_override = pc_eta
        # 推論で動かす自由変数の上書き。delta は PB(視覚->行動)への自由オフセットで
        # 内部状態ではないので、"hv" にすると「内部状態のみ推論」になる。
        if self._pc_vars_override and hasattr(self.agent, "pc_vars"):
            self.agent.pc_vars = self._pc_vars_override
        self.pc_delta_warm = bool(self.cfg["model"].get("pc_delta_warm", False)) \
            if self._pc_delta_warm_arg is None else bool(self._pc_delta_warm_arg)
        # 行動振幅のスケール。学習ログで |a_hat| が教師の ~72% しかないため
        # (GMM の最大重み成分を代表点に使う縮み + tanh)、到達動作が足りていない
        # 疑いを検証する。既定 1.0 = 従来と完全に同一。グリッパ次元(6)は開閉指令で
        # 意味が違うので既定ではスケールしない。
        self.action_scale = float(action_scale)
        self.scale_gripper = bool(scale_gripper)
        # --- 摂動条件(ロバストネス評価用。既定 none で従来と完全に同一) ---
        #   intervene : perturb_at から perturb_len step だけランダム行動を送る(人の介入/外乱)
        #   obs_noise : 視覚特徴 v にガウスノイズを乗せる(センサ劣化)
        #   act_delay : 行動を d step 遅らせて送る(制御遅延・ハードウェア制約)
        # オラクルグリッパ(2026-07-31、診断専用)。
        # ロールアウト中にシミュレータの真値で eef と対象物体の距離を見て、
        # oracle_grip_at [cm] 未満になったらグリッパ次元を +1(閉じる)に上書きし、
        # 以降は閉じたままにする。他の 6 次元はモデルの出力のまま。
        # 問い: object 10タスクは d_min 4.8cm(xy 3.8 / z 2.3)まで到達するのに
        # 最接近時のグリッパ指令が例外なく -1.00(開く)で、閉率は 4%(デモは 47-58%)。
        # 「閉じるトリガが出ないだけ」なのか「その距離では届いていない」のかを分ける。
        # 真値を使うので成功率として報告してはいけない(診断専用)。
        self.oracle_grip_at = float(oracle_grip_at)
        self.oracle_grip_key = oracle_grip_key
        self._og_closed = False
        self.perturb = perturb
        self.perturb_at = int(perturb_at)
        self.perturb_len = int(perturb_len)
        self.obs_noise = float(obs_noise)
        self.act_delay = int(act_delay)
        self._prng = np.random.default_rng(perturb_seed)
        # --- 画像空間の摂動(実機のカメラ劣化に対応。特徴空間ノイズより現実的) ---
        #   noise   : 画素ガウスノイズ (strength = σ, 0-255 スケール)  センサノイズ/低照度
        #   blur    : ガウシアンブラー (strength = カーネルサイズ)      露光/振動
        #   bright  : 明度ゲイン (strength = 相対変化, 例 0.3 で +30%)  照明変化
        #   occlude : 矩形遮蔽 (strength = 面積比, 例 0.1 で 10%)      遮蔽/汚れ
        #   shift   : 画像の平行移動 (strength = 画素)                 キャリブレーションずれ
        # 摂動の実現値は「エピソード内の tick 番号」で決定論的に決まるので、
        # アーム間で同一の劣化条件になる(公平な比較のため)。
        self.img_perturb = img_perturb
        self.img_strength = float(img_strength)
        self.perturb_seed = int(perturb_seed)
        self._tick = 0
        self._occ = None

        cache = _abs(self.cfg["paths"]["cache_root"])
        self.av_pca = FrozenPCA.load(os.path.join(cache, "pca_agentview.npz"))
        self.eye_pca = FrozenPCA.load(os.path.join(cache, "pca_eye.npz"))
        pp_ = self.cfg["preprocess"]
        self.lang_source = pp_.get("language_source", "minilm")
        self.lang_pca = None
        if self.lang_source == "minilm" and pp_["language_pca_dim"] != 0:
            self.lang_pca = FrozenPCA.load(os.path.join(cache, "pca_language.npz"))
        st = np.load(os.path.join(cache, "norm_stats.npz"))
        self.q_mean, self.q_std = st["q_mean"], st["q_std"]

        pp = self.cfg["preprocess"]
        self.encoder = FrozenResNet18(agentview_grid=pp["agentview_grid"]).to(device)
        self.lang_enc = LanguageEncoder(device="cpu")
        self.stride = self.cfg["model"]["vision_stride"]
        self.vh = self.cfg["model"]["vision_hidden"]
        self.ah = self.cfg["model"]["action_hidden"]
        self.pb_dim = self.cfg["model"]["pb_dim"]
        self.v_dim = self.agent.v_dim
        self.q_dim = self.agent.q_dim

    def _scale_action(self, a: np.ndarray) -> np.ndarray:
        if self.action_scale == 1.0:
            return a.astype(np.float32)
        out = a.astype(np.float32).copy()
        hi = 7 if self.scale_gripper else 6
        out[:hi] = np.clip(out[:hi] * self.action_scale, -1.0, 1.0)
        return out

    @torch.no_grad()
    def v_from_obs(self, agent_img: np.ndarray, eye_img: np.ndarray) -> torch.Tensor:
        if self.img_perturb != "none":
            agent_img = self._perturb_image(np.asarray(agent_img), 0)
            eye_img = self._perturb_image(np.asarray(eye_img), 1)
            self._tick += 1
        av = torch.from_numpy(np.ascontiguousarray(agent_img[None])).to(self.device)
        eye = torch.from_numpy(np.ascontiguousarray(eye_img[None])).to(self.device)
        av_tok = self.encoder.agentview_tokens(av).cpu().numpy().reshape(-1, 512)  # (4,512)
        eye_tok = self.encoder.eye_token(eye).cpu().numpy()                        # (1,512)
        av_p = self.av_pca.transform(av_tok).reshape(-1)                           # (128,)
        eye_p = self.eye_pca.transform(eye_tok).reshape(-1)                        # (64,)
        v = np.concatenate([av_p, eye_p]).astype(np.float32)
        if self.obs_noise > 0.0:                      # センサ劣化の模擬
            v = v + self.obs_noise * self._prng.standard_normal(v.shape).astype(np.float32)
        return torch.from_numpy(v)[None].to(self.device)

    def language_latent(self, text: str) -> torch.Tensor:
        # オラクル条件で obs キーを決めるため、指示文を覚えておく
        self._tgt_lang = text
        if self.lang_source == "task_id":
            # one-hot over the suite's task list (matched by instruction text)
            from libero.libero import benchmark
            suite = benchmark.get_benchmark_dict()[self.cfg["data"]["suite"]]()
            langs = [suite.get_task(i).language for i in range(suite.n_tasks)]
            l = np.zeros(suite.n_tasks, np.float32)
            l[langs.index(text)] = 1.0
        elif self.lang_pca is None:
            l = self.lang_enc.encode([text])[0].astype(np.float32)      # raw 384d
        else:
            l = self.lang_pca.transform(self.lang_enc.encode([text]))[0].astype(np.float32)
        return torch.from_numpy(l)[None].to(self.device)

    def _resolve_target_keys(self, obs, lang: str):
        """オラクル条件で使う obs キー(対象物体・置き場所)を言語から決める。
        学習側 preprocess.target_basket_positions と同じ規則にする。"""
        t = lang.lower()
        tgt = None
        if "pick up the " in t:
            rest = t.split("pick up the ", 1)[1]
            for b in (" and place", " and put"):
                if b in rest:
                    tgt = rest.split(b, 1)[0].strip(); break
        assert tgt, f"対象物体を言語から取れない: {lang}"
        tk = f"{tgt.replace(' ', '_')}_1_pos"
        assert tk in obs, f"{tk} が obs に無い"
        gk = next((k for k in ("basket_1_pos", "plate_1_pos") if k in obs), None)
        assert gk, "置き場所が見つからない"
        self._tgt_keys = (tk, gk)

    def q_from_obs(self, obs) -> torch.Tensor:
        parts = [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        src = self.cfg["preprocess"].get("proprio_source", "joint_gripper")
        if src == "joint_gripper_eef":
            import robosuite.utils.transform_utils as T
            # dataset ee_ori is axis-angle; live env gives a quaternion
            parts += [obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])]
        elif src == "joint_gripper_target":
            # オラクル: 対象物体(3) + 置き場所(3)。学習キャッシュと同じ順序・同じキー。
            # キーはエピソード開始時に一度決めて self._tgt_keys に持つ。
            tk, gk = self._tgt_keys
            parts += [np.asarray(obs[tk], np.float32), np.asarray(obs[gk], np.float32)]
        elif src == "joint_gripper_obj":
            # オラクル物体位置(5物体 x 3)。学習時は sim state から復元した同じ量を使う
            # (src/data/preprocess.py の OBJ_KEYS と順序を揃えること)。
            from src.data.preprocess import OBJ_KEYS
            parts += [np.asarray(obs[k], np.float32) for k in OBJ_KEYS]
        q = np.concatenate(parts).astype(np.float32)
        q = (q - self.q_mean) / self.q_std
        return torch.from_numpy(q.astype(np.float32))[None].to(self.device)

    def _begin_episode(self):
        """エピソード開始時に摂動の状態をリセットする(tick カウンタ・遮蔽位置)。"""
        self._tick = 0
        self._occ = None
        if self.img_perturb == "occlude" and self.img_strength > 0:
            rng = np.random.default_rng(self.perturb_seed + 12345)
            side = float(np.sqrt(self.img_strength))
            self._occ = (rng.random(), rng.random(), side)   # (y比, x比, 一辺の比)

    def _perturb_image(self, img, cam: int):
        """uint8 (H,W,3) の画像に摂動を適用して返す。cam: 0=agentview, 1=eye_in_hand"""
        k = self.img_perturb
        if k == "none" or self.img_strength == 0:
            return img
        x = img.astype(np.float32)
        if k == "noise":
            rng = np.random.default_rng((self.perturb_seed * 1000003 + self._tick * 1009 + cam)
                                        % (2 ** 32))
            x = x + rng.standard_normal(x.shape).astype(np.float32) * self.img_strength
        elif k == "blur":
            import cv2
            ks = max(3, int(self.img_strength) | 1)
            x = cv2.GaussianBlur(x, (ks, ks), 0)
        elif k == "bright":
            x = x * (1.0 + self.img_strength)
        elif k == "occlude":
            h, w = x.shape[:2]
            fy, fx, side = self._occ
            ah, aw = int(h * side), int(w * side)
            y0 = int(fy * max(1, h - ah)); x0 = int(fx * max(1, w - aw))
            x[y0:y0 + ah, x0:x0 + aw] = 0.0
        elif k == "shift":
            d = int(self.img_strength)
            x = np.roll(np.roll(x, d, axis=0), d, axis=1)
        else:
            raise ValueError(f"unknown img_perturb: {k}")
        return np.clip(x, 0, 255).astype(np.uint8)

    def _perturb_action(self, a_send, s, buf):
        """介入(ランダム行動)と遅延を適用して実際に送る行動を返す。"""
        if self.perturb == "intervene" and self.perturb_at <= s < self.perturb_at + self.perturb_len:
            a_send = self._prng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        if self.act_delay > 0:
            buf.append(a_send.copy())
            a_send = buf[0] if len(buf) > self.act_delay else np.zeros(7, np.float32)
            if len(buf) > self.act_delay:
                buf.pop(0)
        return a_send

    @torch.no_grad()
    def run_episode(self, env, l_vec, init_state, warmup=10, max_steps=400, settle=5,
                    collect=False, frames=None, obs_rec=None, diag=None,
                    imagine_at=(), imagine_len=100, force_actions=None):
        """collect=True records the on-policy (v, q, a) stream of non-warmup steps
        (fresh per-frame v to match the demo cache), returned as result['traj'] for
        self-imitation. The policy is unchanged, so success rates are identical.
        frames: リストを渡すと agentview フレーム(表示用に上下反転済み)を毎 step 追記する。
        obs_rec: リストを渡すと毎 step の `*_pos` 系 obs(手先・物体位置)を追記する。
                 「最初にどこへ向かっているか」の解析用(方策には影響しない)。
        diag: dict を渡すと診断量を記録する(方策には影響しない)。
              eps_v  : 視覚予測誤差 ||v_t - v_hat||^2/D_v (視覚 tick ごと、tick step 番号つき)
              eps_q  : 固有感覚予測誤差 ||q_t - q_hat_{t-1}||^2/D_q (毎 step)
              q_obs  : 観測した q (毎 step, 9次元)
              q_pred : 1 step 先予測 q_hat (毎 step, 9次元。q_obs[t] と q_pred[t-1] が対応)
        imagine_at: この step で「観測を使わない開ループ予測」を imagine_len step ぶん
              走らせ、予測 q の軌跡を diag["imagine"] に (start_step, array) で記録する。
              自分の予測を入力に戻すだけで env は進めないので、方策・環境には影響しない。
              「最初の予測からズレているのか、誤差の蓄積でズレるのか」を見るため。
        force_actions: (n, a_dim) を渡すと最初の n step はこの行動を環境に送る
              (モデルは通常どおり観測を受けて内部状態を更新し、行動だけ捨てる)。
              デモの行動列を前半に入れて「正しい軌道で内部状態を育ててから引き渡す」
              実験に使う。warmup の零行動より優先される。"""
        A = self.agent
        if hasattr(A, "rollout_step"):    # generic interface (baselines)
            return self._run_episode_generic(env, l_vec, init_state, warmup, max_steps, settle,
                                             frames=frames, obs_rec=obs_rec,
                                             force_actions=force_actions)
        ctx = A.lang_ctx(l_vec)           # static language context (episode-constant)
        # 学習可能な初期状態(learn_init_state)があればそれを使う。既定はゼロ。
        # init_encoder=True のときは最初の観測が必要なので、settle 後に作り直す
        # (下の「初期状態エンコーダ」ブロック)。
        if getattr(A, "learn_init", False):
            h_v, h_low, h_up = (x.detach() for x in A.initial_states(1, ctx))
        else:
            h_v = torch.zeros(1, self.vh, device=self.device)
            h_low = torch.zeros(1, self.ah, device=self.device)
            h_up = torch.zeros(1, self.ah, device=self.device)
        pb_prev = torch.zeros(1, self.pb_dim, device=self.device)
        pb_curr = torch.zeros(1, self.pb_dim, device=self.device)
        v_pred_prev = None
        win_start = 0

        self._begin_episode()
        self._og_closed = False           # オラクルグリッパの状態をエピソードごとに戻す
        env.reset()                       # clear any terminated flag from prior episode
        obs = env.set_init_state(init_state)
        if self.cfg["preprocess"].get("proprio_source") == "joint_gripper_target" \
                and getattr(self, "_tgt_lang", None):
            self._resolve_target_keys(obs, self._tgt_lang)
        # physics settle with zero action (LIBERO-canonical; not fed to the agent)
        for _ in range(settle):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        # 初期状態エンコーダ(init_encoder): h_0 = f(v_0, q_0, ctx) から
        # (h_v, h_low, h_up) を作る。学習側は窓の先頭 v_in[:,0]/q_in[:,0] を渡すので、
        # ここでもエピソード先頭(settle 後・最初の行動を出す前)の観測を使う。
        if getattr(A, "init_enc", False):
            with torch.no_grad():
                _v0 = self.v_from_obs(obs["agentview_image"],
                                      obs["robot0_eye_in_hand_image"])
                _q0 = self.q_from_obs(obs)
                h_v, h_low, h_up = (x.detach() for x in
                                    A.initial_states(1, _v0, _v0, _q0, ctx))
        success = False
        reward_max = 0.0
        eef_path = 0.0
        act_abs = 0.0
        eef_prev = np.asarray(obs["robot0_eef_pos"])
        n = 0
        # --- train-matched windowed PC inference (models trained with pc_infer_k>0):
        # sliding window of the last `win` steps; at each vision tick the window
        # is re-inferred (initial h_v + PB delta) and the refined states replace
        # the streaming ones. Mirrors training exactly.
        pc_on = getattr(A, "pc_k", 0) > 0
        win = self.cfg["train"]["burn_in"] if pc_on else 0
        v_buf, q_buf, snap_buf = [], [], []   # per-step v (tick steps), q, h_v snapshots at ticks
        a_buf = []                            # 実行した行動(視覚 RNN に渡す efference copy 用)
        pc_delta = torch.zeros(1, self.pb_dim, device=self.device)
        chunk = getattr(A, "chunk", 1)
        plan = None                            # (C, a_dim) planned actions (chunk mode)
        lu_pb = self.lu_pb_eta > 0.0
        delta_pb = torch.zeros(1, self.pb_dim, device=self.device)
        delta = None                       # per-window leaf (requires_grad)
        E_win = None                       # accumulated proprio error (graph)
        q_hat_prev = None
        rec_v, rec_q, rec_a = [], [], []   # self-imitation trajectory (collect=True)
        _abuf = []                          # 行動遅延用バッファ
        if diag is not None:
            diag.setdefault("eps_v", []); diag.setdefault("eps_q", [])
            diag.setdefault("q_obs", []); diag.setdefault("q_pred", [])
            diag.setdefault("imagine", [])
            # win_rec: 視覚 tick ごとの窓推論のあとで、窓内(直近 win step)の各 step の
            # eps_v / eps_q / q_hat を記録する。窓推論は tick ごとに直近 20 step を
            # 再推論するので、同じ絶対 step の値が tick をまたいで更新される。
            # 「窓の中は毎 tick 更新され、窓から出たら最後の値で凍る」を描くのに使う。
            diag.setdefault("win_rec", [])
        _q_hat_diag = None                  # 直前 step の 1step 先予測(診断用)
        for s in range(max_steps):
            # q_t は視覚 tick より前に計算する（視覚 RNN に固有感覚を渡す構成で、
            # 学習側の q_in[:, s] と同じタイミングに揃えるため。2026-07-30）
            q_t = self.q_from_obs(obs)
            if s % self.stride == 0:
                # --- Phase B: close out the previous window (delta update) ---
                if lu_pb and delta is not None and E_win is not None:
                    with torch.enable_grad():
                        (g,) = torch.autograd.grad(E_win, delta, allow_unused=True)
                    if g is not None:
                        delta_pb = (delta_pb - self.lu_pb_eta * g).detach()
                        nrm = float(delta_pb.norm())
                        if nrm > self.lu_pb_clip:
                            delta_pb = delta_pb * (self.lu_pb_clip / nrm)
                    h_low = h_low.detach()
                    h_up = h_up.detach()
                    if q_hat_prev is not None:
                        q_hat_prev = q_hat_prev.detach()
                    E_win = None

                v_t = self.v_from_obs(obs["agentview_image"], obs["robot0_eye_in_hand_image"])
                # --- windowed PC inference over the past `win` steps ---
                if pc_on and len(q_buf) >= win and len(snap_buf) > win // self.stride:
                    vv = torch.stack([x if x is not None else torch.zeros(1, self.v_dim, device=self.device)
                                      for x in v_buf[-win:]], dim=1)
                    qq = torch.stack(q_buf[-win:], dim=1)
                    _wrec = ({"eps_v": [], "eps_q": [], "q_hat": []}
                             if diag is not None else None)
                    with torch.enable_grad():
                        pc_delta, (h_v, h_low, h_up, pb_prev, pb_curr,
                                   v_pred_prev, _, _) = A.pc_infer(
                            vv, qq, qq, vv, ctx, win, create_graph=False,
                            h0=snap_buf[-(win // self.stride) - 1],
                            fe_terms=self.fe_terms,
                            # 前 tick の推論結果を delta の事前値にする(warm start)。
                            # 従来は毎 tick 0 からで、k=5・eta=0.1 では 1e-2 しか動けなかった。
                            delta0=pc_delta if self.pc_delta_warm else None,
                            rec=_wrec,
                            a_in=(torch.stack(a_buf[-win:], dim=1)
                                  if (a_buf and len(a_buf) >= win) else None))
                    pc_delta = pc_delta.detach()
                    if _wrec is not None:
                        # 窓の先頭に対応する絶対 step。pc_infer は現 step を v_buf に
                        # append する前に走るので、窓が覆うのは s-win .. s-1。
                        w0 = s - win
                        diag["win_rec"].append((
                            w0,
                            [(w0 + i, e) for i, e in _wrec["eps_v"]],
                            [(w0 + i, e) for i, e in _wrec["eps_q"]],
                            [(w0 + i, q) for i, q in _wrec["q_hat"]]))
                    h_v = h_v.detach(); h_low = h_low.detach(); h_up = h_up.detach()
                    pb_prev = pb_prev.detach(); pb_curr = pb_curr.detach()
                    v_pred_prev = v_pred_prev.detach() if v_pred_prev is not None else None
                if pc_on:
                    snap_buf.append(h_v.detach())
                    snap_buf = snap_buf[-(win // self.stride + 2):]
                # --- Phase A: refine pre-step h_v against the current observation ---
                if self.lu_k > 0 and v_pred_prev is not None:
                    h0 = h_v.detach()
                    with torch.enable_grad():
                        h = h0.clone().requires_grad_(True)
                        for _ in range(self.lu_k):
                            pred = A.vision.predict(h)
                            E = ((v_t - pred) ** 2).mean() \
                                + self.lu_beta * ((h - h0) ** 2).mean()
                            (gh,) = torch.autograd.grad(E, h)
                            h = (h - self.lu_eta * gh).detach().requires_grad_(True)
                    h_v = h.detach()
                    v_pred_prev = A.vision.predict(h_v)

                eps_v = torch.zeros(1, self.v_dim, device=self.device) \
                    if v_pred_prev is None else (v_t - v_pred_prev)
                if diag is not None and v_pred_prev is not None:
                    diag["eps_v"].append((s, float((eps_v ** 2).sum() / self.v_dim)))
                _ac = _qc = None
                if getattr(A, 'vis_use_a', False) and a_buf:
                    _ac = torch.stack(a_buf[-self.stride:], dim=0).mean(dim=0)
                if getattr(A, 'vis_use_q', False):
                    _qc = q_t
                h_v = A.vision.step(v_t, ctx, h_v, a_ctx=_ac, q_ctx=_qc)
                v_pred_prev = A.vision.predict(h_v)
                pb_prev = pb_curr
                pb_curr = A.pb_gen(h_v, eps_v)
                win_start = s
                if lu_pb:
                    with torch.enable_grad():
                        delta = delta_pb.clone().requires_grad_(True)
            i = s - win_start
            frac = (i + 1) / self.stride
            pb_t = (1.0 - frac) * pb_prev + frac * pb_curr
            if pc_on:
                pb_t = pb_t + pc_delta

            if lu_pb:
                with torch.enable_grad():
                    h_low, h_up = A.action.step(q_t, ctx, pb_t + delta, h_low, h_up)
                    a_hat, q_head = A.action.heads(h_low)
                    q_hat = A.q_pred(h_low, q_t, a_hat)[0] \
                        if getattr(A, "proprio_from_action", False) else q_head
                    if q_hat_prev is not None:
                        e = ((q_t - q_hat_prev) ** 2).mean()
                        E_win = e if E_win is None else E_win + e
                    q_hat_prev = q_hat
                a_hat = a_hat.detach()
            else:
                h_low, h_up = A.action.step(q_t, ctx, pb_t, h_low, h_up)
                a_hat, q_head = A.action.heads(h_low)
                q_hat_d = A.q_pred(h_low, q_t, a_hat)[0] \
                    if getattr(A, "proprio_from_action", False) else q_head
                if diag is not None:
                    qn = q_t[0].detach().cpu().numpy().copy()
                    diag["q_obs"].append(qn)
                    if _q_hat_diag is not None:
                        e = float(((q_t - _q_hat_diag) ** 2).sum() / self.q_dim)
                        diag["eps_q"].append((s, e))
                        diag["q_pred"].append(_q_hat_diag[0].detach().cpu().numpy().copy())
                    else:
                        diag["q_pred"].append(qn * np.nan)
                    _q_hat_diag = q_hat_d.detach()
                # --- 観測を使わない開ループ予測(想像)。方策・環境には影響しない ---
                if diag is not None and s in tuple(imagine_at):
                    diag["imagine"].append((s, self._imagine(A, ctx, h_v, h_low, h_up,
                                                             pb_prev, pb_curr, v_pred_prev,
                                                             q_t, s, win_start, pc_delta if pc_on else None,
                                                             imagine_len)))
            if pc_on:
                v_buf.append(v_t if s % self.stride == 0 else None)
                q_buf.append(q_t)
                v_buf = v_buf[-win:]
                q_buf = q_buf[-win:]
            if chunk > 1:
                # plan a chunk at each vision tick, execute open-loop within it
                if i == 0 or plan is None:
                    plan = a_hat.detach().reshape(chunk, -1)
                a_np = np.clip(plan[min(i, chunk - 1)].cpu().numpy(), -1.0, 1.0)
            else:
                a_np = np.clip(a_hat[0].cpu().numpy(), -1.0, 1.0)

            if collect and s >= warmup:
                # record the HELD v_t the model actually consumed (no extra ResNet),
                # q as fed, and the executed policy action -> demo-cache format
                rec_v.append(v_t[0].detach().cpu().numpy())
                rec_q.append(q_t[0].detach().cpu().numpy())
                rec_a.append(a_np.astype(np.float32))

            if force_actions is not None and s < len(force_actions):
                a_send = np.asarray(force_actions[s], dtype=np.float32)
            elif s < warmup:
                a_send = np.zeros(7, dtype=np.float32)
            else:
                a_send = self._scale_action(a_np)
            a_send = self._perturb_action(a_send, s, _abuf)
            if self.oracle_grip_at > 0.0 and self.oracle_grip_key:
                # 真値で eef-対象距離を見て、閾値を切ったら閉じる(以降維持)
                if self.oracle_grip_key in obs:
                    _d = float(np.linalg.norm(
                        np.asarray(obs["robot0_eef_pos"])
                        - np.asarray(obs[self.oracle_grip_key])[:3])) * 100.0
                    if _d < self.oracle_grip_at:
                        self._og_closed = True
                if self._og_closed:
                    a_send = a_send.copy()
                    a_send[6] = 1.0
            if getattr(A, 'vis_use_a', False):
                a_buf.append(torch.from_numpy(np.asarray(a_send[:A.a_dim], np.float32))
                             [None].to(self.device))
                a_buf = a_buf[-max(win, self.stride) - 2:]
            obs, reward, done, info = env.step(a_send)
            if frames is not None:                 # 可視化用の録画(方策には影響しない)
                frames.append(np.asarray(obs["agentview_image"])[::-1].copy())
            if obs_rec is not None:                # 手先・物体位置の記録(解析用)
                _r = {k: np.asarray(v, np.float32).copy()
                      for k, v in obs.items()
                      if k.endswith("_pos") and not k.startswith("robot0_joint")}
                # 実際に env へ送った行動も残す(2026-07-31)。閉ループで
                # 「対象に最も近づいた瞬間にグリッパが閉じているか」を測るため。
                # 記録するだけなので方策・環境には影響しない。
                _r["_a_send"] = np.asarray(a_send, np.float32).copy()
                obs_rec.append(_r)

            eef = np.asarray(obs["robot0_eef_pos"])
            eef_path += float(np.linalg.norm(eef - eef_prev))
            eef_prev = eef
            act_abs += float(np.abs(a_np).mean())
            reward_max = max(reward_max, float(reward))
            n = s + 1
            if float(reward) >= 1.0 or bool(info.get("success", False)):
                success = True
                break
            if done:
                break

        out = {
            "success": success,
            "steps": n,
            "reward_max": reward_max,
            "eef_path": eef_path,
            "act_absmean": act_abs / max(n, 1),
        }
        if collect:
            # only expose traj under collect=True; eval_suite's CSV DictWriter
            # rejects unknown keys, so the default eval path must stay clean
            out["traj"] = None
            if rec_v:
                out["traj"] = {"v": np.stack(rec_v).astype(np.float32),
                               "q": np.stack(rec_q).astype(np.float32),
                               "a": np.stack(rec_a).astype(np.float32)}
        return out

    @torch.no_grad()
    @torch.no_grad()
    def _imagine(self, A, ctx, h_v, h_low, h_up, pb_prev, pb_curr, v_pred_prev,
                 q_t, s0, win_start, pc_delta, steps):
        """現在の内部状態から、観測を一切使わずに `steps` step 先まで自己予測で走らせ、
        予測した固有感覚の軌跡を返す (steps, q_dim)。

        なぜ必要か: rollout の破綻が「最初の予測から既にズレている」のか
        「1step 予測は合っているが誤差が蓄積してズレる」のかを分離したい。
        観測を入力に使わず、視覚は自分の予測 v_hat を、固有感覚は自分の予測 q_hat を
        次の入力に戻す(= 学習時の scheduled sampling を確率 1 にした状態)。
        env は進めないので方策にも環境にも影響しない。"""
        hv = h_v.clone(); hl = h_low.clone(); hu = h_up.clone()
        pp = pb_prev.clone(); pc = pb_curr.clone()
        vp = None if v_pred_prev is None else v_pred_prev.clone()
        q = q_t.clone()
        ws = win_start
        out = []
        for k in range(steps):
            s = s0 + k
            if s % self.stride == 0 and k > 0:
                # 観測が無いので自分の視覚予測を「観測」として使う
                v_in = vp if vp is not None else torch.zeros(1, self.v_dim, device=self.device)
                eps = torch.zeros_like(v_in)      # 予測と一致するので誤差ゼロ扱い
                hv = A.vision.step(v_in, ctx, hv)
                vp = A.vision.predict(hv)
                pp = pc
                pc = A.pb_gen(hv, eps)
                ws = s
            frac = ((s - ws) + 1) / self.stride
            pb = (1.0 - frac) * pp + frac * pc
            if pc_delta is not None:
                pb = pb + pc_delta
            hl, hu = A.action.step(q, ctx, pb, hl, hu)
            a_h, q_head = A.action.heads(hl)
            q_hat = A.q_pred(hl, q, a_h)[0] \
                if getattr(A, "proprio_from_action", False) else q_head
            q = q_hat                            # 自分の予測を次の入力に戻す
            out.append(q_hat[0].detach().cpu().numpy().copy())
        return np.stack(out)

    def _run_episode_generic(self, env, l_vec, init_state, warmup, max_steps, settle,
                             force_actions=None,
                             frames=None, obs_rec=None):
        """Rollout for models exposing rollout_reset/rollout_step (baselines).
        Same protocol as the LegacyPCAgent path: settle, warmup, vision every stride.

        開ループ評価（2026-08-01 追加）
          self.ol_freeze_vision : v を最初の 1 回だけ観測し、以降更新しない
          self.ol_proprio_pred  : q を観測ではなく自分の予測 q̂ に置き換えて自走する
        両方を立てると「初期観測と言語だけから行動列を生成する」= PredVLA の n_itr=0
        （観測がモデルに一切入らない開ループ）に対応する条件になる。
        重みは同じで評価時だけの ablation なので、PredVLA 側と完全に対応する。
        既定は両方 False で従来と同一。"""
        A = self.agent
        ctx = A.lang_ctx(l_vec)
        state = A.rollout_reset(1, self.device)
        self._begin_episode()
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(settle):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        success = False
        reward_max = 0.0
        eef_path = 0.0
        act_abs = 0.0
        eef_prev = np.asarray(obs["robot0_eef_pos"])
        v_t = None
        n = 0
        _abuf = []                          # 行動遅延用バッファ
        ol_fv = bool(getattr(self, "ol_freeze_vision", False))
        ol_qp = bool(getattr(self, "ol_proprio_pred", False))
        q_fed = None                        # 開ループ時に次 step へ戻す q̂
        # action chunking（2026-08-01）: chunk step 分をまとめて出し、開ループで実行して
        # から再観測する。chunk=1 なら毎 step 再観測＝従来と完全に同一。
        chunk = int(getattr(A, "chunk", 1))
        cache = None                        # (chunk, a_dim) の実行待ち行動
        # --- ACT 式の temporal ensembling（2026-08-02 追加）---
        # 素朴なチャンク実行（k step 分を出して開ループで消化 → 再観測）は ACT の
        # レシピではない。ACT は **毎 step モデルを呼び**、過去 k 回分の予測のうち
        # 現在 step を覆うものを指数重みで平均する。論文自身が ensembling 無しだと
        # 動作が不連続になって性能が落ちると報告している。
        #
        # さらに素朴版は学習時との不一致もある。学習では毎 step 文脈が更新されるが、
        # 素朴版は k step に 1 回しかモデルを呼ばないので文脈の入力分布がずれる。
        # 毎 step 呼ぶ ensembling はこれも同時に直す。
        #
        # 重みは ACT の実装に合わせ exp(-m*i)（i=0 が最も古い予測、m=0.01 既定）。
        ens = bool(getattr(self, "chunk_ensemble", False)) and chunk > 1
        ens_m = float(getattr(self, "chunk_ensemble_m", 0.01))
        plans = []                          # [(発行 step, (chunk, a_dim))] 古い順
        for s in range(max_steps):
            use_model = (chunk == 1) or ens or (s % chunk == 0)
            if use_model:
                if s % self.stride == 0 and (v_t is None or not ol_fv):
                    v_t = self.v_from_obs(obs["agentview_image"],
                                          obs["robot0_eye_in_hand_image"])
                q_t = q_fed if (ol_qp and q_fed is not None) else self.q_from_obs(obs)
                if ol_qp or chunk > 1:
                    a_out, q_hat, state = A.rollout_step_q(v_t, q_t, ctx, state)
                    if ol_qp:
                        q_fed = q_hat.detach()
                else:
                    a_out, state = A.rollout_step(v_t, q_t, ctx, state)
                cache = a_out if chunk > 1 else None
                a_hat = a_out[:, 0] if chunk > 1 else a_out
            else:
                a_hat = cache[:, s % chunk]
            if ens:
                plans.append((s, a_out[0].detach().cpu().numpy()))     # (chunk, a_dim)
                plans = [(s0, p) for (s0, p) in plans if 0 <= s - s0 < chunk]
                cands = np.stack([p[s - s0] for (s0, p) in plans])     # 古い順
                w = np.exp(-ens_m * np.arange(len(cands)))
                w = w / w.sum()
                a_np = np.clip((cands * w[:, None]).sum(0), -1.0, 1.0)
            else:
                a_np = np.clip(a_hat[0].cpu().numpy(), -1.0, 1.0)
            if force_actions is not None and s < len(force_actions):
                a_send = np.asarray(force_actions[s], dtype=np.float32)
            elif s < warmup:
                a_send = np.zeros(7, dtype=np.float32)
            else:
                a_send = self._scale_action(a_np)
            a_send = self._perturb_action(a_send, s, _abuf)
            obs, reward, done, info = env.step(a_send)
            if frames is not None:                 # 可視化用の録画(方策には影響しない)
                frames.append(np.asarray(obs["agentview_image"])[::-1].copy())
            if obs_rec is not None:                # 手先・物体位置の記録(解析用)
                _r = {k: np.asarray(v, np.float32).copy()
                      for k, v in obs.items()
                      if k.endswith("_pos") and not k.startswith("robot0_joint")}
                # 実際に env へ送った行動も残す(2026-07-31)。閉ループで
                # 「対象に最も近づいた瞬間にグリッパが閉じているか」を測るため。
                # 記録するだけなので方策・環境には影響しない。
                _r["_a_send"] = np.asarray(a_send, np.float32).copy()
                obs_rec.append(_r)
            eef = np.asarray(obs["robot0_eef_pos"])
            eef_path += float(np.linalg.norm(eef - eef_prev))
            eef_prev = eef
            act_abs += float(np.abs(a_np).mean())
            reward_max = max(reward_max, float(reward))
            n = s + 1
            if float(reward) >= 1.0 or bool(info.get("success", False)):
                success = True
                break
            if done:
                break
        return {
            "success": success,
            "steps": n,
            "reward_max": reward_max,
            "eef_path": eef_path,
            "act_absmean": act_abs / max(n, 1),
        }
