"""Preprocessing / feature caching (Phase 1, §3).

For each task hdf5 we extract FROZEN features once and cache them; training
reads only the cache.

Per frame:
  v (192) = 4 agentview tokens x 32 (PCA)  +  1 eye_in_hand token x 64 (PCA)
  q (9)   = [joint(7), gripper(2)]  z-scored (stats from training data)
  a (7)   = raw LIBERO actions in [-1, 1]
  l (32)  = PCA(MiniLM(instruction))   (constant over the episode)

Frozen artifacts saved under cache_root:
  pca_agentview.npz, pca_eye.npz, pca_language.npz, norm_stats.npz
Per-task cache: cache_root/<suite>/<task_name>.h5 with v,q,a,l,ep_bounds.
"""
import os
import sys
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)

from src.utils import compat  # noqa: E402
from src.data.pca import FrozenPCA  # noqa: E402
from src.models.encoders import FrozenResNet18, LanguageEncoder  # noqa: E402

compat.apply()


def pick_device(requested: str = "auto") -> str:
    return compat.pick_device(requested)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJ, p)


# --------------------------------------------------------------------------- #
# LIBERO task enumeration
# --------------------------------------------------------------------------- #
def get_suite_tasks(suite_name: str, data_root: str) -> List[Tuple[int, object, str, str]]:
    """-> list of (task_id, task, hdf5_path, language) for tasks with data present."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    out = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        path = os.path.join(_abs(data_root), suite_name, task.name + "_demo.hdf5")
        if os.path.exists(path):
            out.append((i, task, path, task.language))
    return out


def all_benchmark_languages() -> List[str]:
    """Every task instruction across all suites (for fitting the language PCA)."""
    from libero.libero import benchmark

    langs = []
    for name, ctor in benchmark.get_benchmark_dict().items():
        try:
            suite = ctor()
        except Exception:
            continue
        for i in range(suite.n_tasks):
            langs.append(suite.get_task(i).language)
    # dedup preserving order
    seen, uniq = set(), []
    for s in langs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# --- オラクル物体位置(対照実験用) -------------------------------------------
# spatial の全タスクで共通の 5 物体。手先相対ではなく絶対座標を使う
# (手先位置は proprio から既に分かるので、必要なのは物体の絶対位置)。
OBJ_KEYS = ["akita_black_bowl_1_pos", "akita_black_bowl_2_pos", "plate_1_pos",
            "glazed_rim_porcelain_ramekin_1_pos", "cookies_1_pos"]
_OBJ_ENV_CACHE = {}


def oracle_obj_positions(hdf5_path: str, states: np.ndarray) -> np.ndarray:
    """各フレームの sim state を env に流し込み、5 物体の絶対位置 (T,15) を返す。"""
    import os as _os
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    name = _os.path.basename(hdf5_path).replace("_demo.hdf5", "")
    if name not in _OBJ_ENV_CACHE:
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        tid = [i for i in range(suite.n_tasks) if suite.get_task(i).name == name]
        assert tid, f"task not found for {name}"
        t = suite.get_task(tid[0])
        bddl = _os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        env.reset()
        _OBJ_ENV_CACHE[name] = env
    env = _OBJ_ENV_CACHE[name]
    out = np.empty((len(states), 3 * len(OBJ_KEYS)), np.float32)
    for i, st in enumerate(states):
        obs = env.set_init_state(st)
        out[i] = np.concatenate([np.asarray(obs[k], np.float32) for k in OBJ_KEYS])
    return out


_TGT_ENV_CACHE = {}


def _find_task(name: str):
    """タスク名から (suite_name, task) を引く。スイート横断で探す。"""
    from libero.libero import benchmark
    for su in ("libero_object", "libero_spatial", "libero_goal", "libero_90", "libero_10"):
        try:
            b = benchmark.get_benchmark_dict()[su]()
        except Exception:
            continue
        for i in range(b.n_tasks):
            if b.get_task(i).name == name:
                return su, b.get_task(i)
    raise AssertionError(f"task not found: {name}")


def target_basket_positions(hdf5_path: str, states: np.ndarray) -> np.ndarray:
    """対象物体の位置(3) + 置き場所(バスケット/皿)の位置(3) を各フレームで返す -> (T,6)。

    動機(2026-07-30): libero_object は全条件で 0.00%。物体位置は視覚特徴から
    リッジ回帰で 1.2cm 誤差・R² 0.988 で復元できるのに、把持器が対象物体に一度も
    27cm 以内に近づかない(d_min 27.7-35.2cm、lift 0.0cm)。単一タスク学習でも同じ。
    「位置が分かっていれば到達できるのか」を切り分けるため、位置を直接与える上限性能を測る。
    オラクル入力なので主結果には使えない。

    対象物体は言語から特定する("pick up the X and place it in the basket" -> X)。
    置き場所は basket_1_pos / plate_1_pos のうちシーンにあるものを使う。
    """
    import os as _os
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    name = _os.path.basename(hdf5_path).replace("_demo.hdf5", "")
    if name not in _TGT_ENV_CACHE:
        su, t = _find_task(name)
        bddl = _os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        o0 = env.reset()
        lang = t.language.lower()
        tgt = None
        if "pick up the " in lang:
            rest = lang.split("pick up the ", 1)[1]
            for b_ in (" and place", " and put"):
                if b_ in rest:
                    tgt = rest.split(b_, 1)[0].strip(); break
        assert tgt, f"対象物体を言語から取れない: {t.language}"
        tkey = f"{tgt.replace(' ', '_')}_1_pos"
        assert tkey in o0, f"{tkey} が obs に無い（候補: " \
            f"{[k for k in o0 if k.endswith('_pos')]}）"
        gkey = next((k for k in ("basket_1_pos", "plate_1_pos") if k in o0), None)
        assert gkey, f"置き場所が見つからない: {[k for k in o0 if k.endswith('_pos')]}"
        _TGT_ENV_CACHE[name] = (env, tkey, gkey)
        print(f"[preprocess]   oracle: target={tkey} goal={gkey}")
    env, tkey, gkey = _TGT_ENV_CACHE[name]
    out = np.empty((len(states), 6), np.float32)
    for i, st in enumerate(states):
        obs = env.set_init_state(st)
        out[i, :3] = np.asarray(obs[tkey], np.float32)
        out[i, 3:] = np.asarray(obs[gkey], np.float32)
    return out


# --------------------------------------------------------------------------- #
# Vision feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_task_features(path: str, encoder: FrozenResNet18, device: str,
                          batch: int, max_demos: int = 0,
                          proprio_source: str = "joint_gripper") -> Dict[str, np.ndarray]:
    """Concatenate all demos of a task -> agentview tokens, eye token, q, a, ep_bounds."""
    av_list, eye_list, q_list, a_list, bounds = [], [], [], [], [0]
    with h5py.File(path, "r") as h:
        demos = sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1]))
        if max_demos:
            demos = demos[:max_demos]
        for dk in demos:
            d = h["data"][dk]
            obs = d["obs"]
            agent = np.asarray(obs["agentview_rgb"])      # (T,128,128,3) uint8
            eye = np.asarray(obs["eye_in_hand_rgb"])
            joint = np.asarray(obs["joint_states"], dtype=np.float32)   # (T,7)
            grip = np.asarray(obs["gripper_states"], dtype=np.float32)  # (T,2)
            act = np.asarray(d["actions"], dtype=np.float32)           # (T,7)
            T = agent.shape[0]

            av_tokens = np.empty((T, encoder.agentview_grid ** 2, 512), np.float32)
            eye_tokens = np.empty((T, 512), np.float32)
            for s in range(0, T, batch):
                e = min(s + batch, T)
                av_b = torch.from_numpy(agent[s:e]).to(device)
                eye_b = torch.from_numpy(eye[s:e]).to(device)
                av_tokens[s:e] = encoder.agentview_tokens(av_b).cpu().numpy()
                eye_tokens[s:e] = encoder.eye_token(eye_b).cpu().numpy()

            av_list.append(av_tokens)
            eye_list.append(eye_tokens)
            parts = [joint, grip]                                      # (T,9)
            if proprio_source == "joint_gripper_eef":
                # ee_pos (3) + ee_ori (3, axis-angle) -> (T,15)
                parts += [np.asarray(d["obs"]["ee_pos"], dtype=np.float32),
                          np.asarray(d["obs"]["ee_ori"], dtype=np.float32)]
            elif proprio_source == "joint_gripper_target":
                # 対象物体(3) + 置き場所(3) を足して (T,15)。オラクル入力。
                parts.append(target_basket_positions(path, np.asarray(d["states"])))
            elif proprio_source == "joint_gripper_obj":
                # オラクル物体位置(5物体 x 3 = 15) を足して (T,24)。
                # 「視覚の空間圧縮(2x2 グリッド)が置き動作の律速か」を切り分けるための
                # 上限性能の対照実験用。hdf5 の obs には物体位置が無いので、毎フレームの
                # sim state を env に流し込んで body の位置を取り出す(手先位置の誤差 <1cm で整合)。
                parts.append(oracle_obj_positions(path, np.asarray(d["states"])))
            q_list.append(np.concatenate(parts, axis=1))
            a_list.append(act)
            bounds.append(bounds[-1] + T)

    return {
        "agentview": np.concatenate(av_list, 0),   # (N,4,512)
        "eye": np.concatenate(eye_list, 0),         # (N,512)
        "q": np.concatenate(q_list, 0),             # (N,9)
        "a": np.concatenate(a_list, 0),             # (N,7)
        "ep_bounds": np.asarray(bounds, np.int64),  # (n_demos+1,)
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(cfg: dict, force: bool = False, max_demos: int = 0, device: str = "auto"):
    pp = cfg["preprocess"]
    suite = pp["suite"]
    data_root = cfg["paths"]["data_root"]
    cache_root = _abs(cfg["paths"]["cache_root"])
    suite_cache = os.path.join(cache_root, suite)
    os.makedirs(suite_cache, exist_ok=True)
    device = pick_device(device if device != "auto" else cfg.get("device", "auto"))
    # ResNet is conv-heavy -> prefer an accelerator (cuda/mps) even if the RNN
    # device is cpu.
    vis_device = compat.accelerator()
    print(f"[preprocess] suite={suite} vision_device={vis_device} force={force} "
          f"max_demos={max_demos or 'all'}")

    tasks = get_suite_tasks(suite, data_root)
    assert tasks, f"No hdf5 found for suite {suite} under {data_root}"
    print(f"[preprocess] {len(tasks)} tasks with data")

    av_pca_p = os.path.join(cache_root, "pca_agentview.npz")
    eye_pca_p = os.path.join(cache_root, "pca_eye.npz")
    lang_pca_p = os.path.join(cache_root, "pca_language.npz")
    stats_p = os.path.join(cache_root, "norm_stats.npz")

    encoder = FrozenResNet18(agentview_grid=pp["agentview_grid"]).to(vis_device)
    lang_enc = LanguageEncoder(device="cpu")

    # --- language representation ---
    # language_source: minilm (default) | task_id (one-hot; "task conditioning"
    # baseline). With minilm, language_pca_dim 0 stores the raw 384-d embedding
    # (no PCA; pair with model.lang_proj=learned). PCA caveat: the instruction
    # corpus has effective rank ~15, so whitened dims beyond that amplify noise.
    lang_source = pp.get("language_source", "minilm")
    n_tasks_suite = max(t[0] for t in tasks) + 1
    lang_pca = None
    if lang_source == "task_id":
        print(f"[preprocess] language = one-hot task id ({n_tasks_suite}d)")
    elif pp["language_pca_dim"] == 0:
        print("[preprocess] language = raw MiniLM embedding (384d, no PCA)")
    elif force or not os.path.exists(lang_pca_p):
        langs = all_benchmark_languages() if pp["language_fit_all_suites"] \
            else [t[3] for t in tasks]
        emb = lang_enc.encode(langs)
        lang_pca = FrozenPCA.fit(emb, pp["language_pca_dim"], pp["pca_whiten"], cfg["seed"])
        lang_pca.save(lang_pca_p)
        print(f"[preprocess] fit language PCA on {len(langs)} instructions "
              f"-> {lang_pca.dim_out}d  (saved)")
    else:
        lang_pca = FrozenPCA.load(lang_pca_p)
        print("[preprocess] loaded frozen language PCA")

    # PCA と proprio 統計の再フィットを分離する(2026-07-30)。
    # オラクル条件では q の次元が変わるので統計だけ取り直したいが、視覚 PCA は
    # 既存のものを再利用しないと v が変わって過去の結果と比較できない。
    have_pca = all(os.path.exists(p) for p in (av_pca_p, eye_pca_p))
    have_stats = os.path.exists(stats_p)
    fit_visual = force or not (have_pca and have_stats)   # 特徴を全部メモリに載せるか
    fit_pca = force or not have_pca
    fit_stats = force or not have_stats

    # ------------------------------------------------------------------ #
    # Extract features for all tasks (kept in memory for the fit pass).
    # ------------------------------------------------------------------ #
    todo = tasks
    if not fit_visual:
        todo = [t for t in tasks
                if force or not os.path.exists(os.path.join(suite_cache, t[1].name + ".h5"))]
        if not todo:
            print("[preprocess] all task caches present & PCA frozen -> nothing to do")
            return
        print(f"[preprocess] frozen visual PCA exists; processing {len(todo)} new task(s)")

    proprio_source = pp.get("proprio_source", "joint_gripper")
    expected_q = {"joint_gripper": 9, "joint_gripper_eef": 15,
                  "joint_gripper_obj": 24,
                  # オラクル: 関節7 + 指2 + 対象物体3 + 置き場所3
                  "joint_gripper_target": 15}[proprio_source]
    assert pp["proprio_dim"] == expected_q, \
        f"proprio_dim={pp['proprio_dim']} but {proprio_source} gives {expected_q}"

    t0 = time.time()
    feats: Dict[int, Dict[str, np.ndarray]] = {}
    for tid, task, path, lang in todo:
        f = extract_task_features(path, encoder, vis_device, pp["image_batch"], max_demos,
                                  proprio_source=proprio_source)
        feats[tid] = f
        print(f"  [{task.name[:48]:48s}] frames={f['agentview'].shape[0]:5d} "
              f"demos={len(f['ep_bounds'])-1}")
    print(f"[preprocess] ResNet extraction done in {time.time()-t0:.1f}s")

    # --- visual PCA + proprio stats (frozen) ---
    if fit_visual:
        av_all = np.concatenate([f["agentview"].reshape(-1, 512) for f in feats.values()], 0)
        eye_all = np.concatenate([f["eye"] for f in feats.values()], 0)
        q_all = np.concatenate([f["q"] for f in feats.values()], 0)
        a_all = np.concatenate([f["a"] for f in feats.values()], 0)

        rng = np.random.default_rng(cfg["seed"])
        cap = pp["pca_fit_max_samples"]
        if av_all.shape[0] > cap:
            av_fit = av_all[rng.choice(av_all.shape[0], cap, replace=False)]
        else:
            av_fit = av_all
        eye_fit = eye_all if eye_all.shape[0] <= cap else \
            eye_all[rng.choice(eye_all.shape[0], cap, replace=False)]

        if fit_pca:
            av_pca = FrozenPCA.fit(av_fit, pp["agentview_pca_dim"], pp["pca_whiten"], cfg["seed"])
            eye_pca = FrozenPCA.fit(eye_fit, pp["eye_pca_dim"], pp["pca_whiten"], cfg["seed"])
            av_pca.save(av_pca_p)
            eye_pca.save(eye_pca_p)
            print(f"[preprocess] fit visual PCA (agentview->{av_pca.dim_out}, "
                  f"eye->{eye_pca.dim_out}) (saved)")
        else:
            av_pca = FrozenPCA.load(av_pca_p)
            eye_pca = FrozenPCA.load(eye_pca_p)
            print("[preprocess] loaded frozen visual PCA (再利用)")
        if fit_stats:
            q_mean = q_all.mean(0).astype(np.float32)
            q_std = q_all.std(0).astype(np.float32)
            q_std = np.where(q_std > 1e-6, q_std, 1e-6)
            np.savez(stats_p, q_mean=q_mean, q_std=q_std)
            print(f"[preprocess] fit proprio stats ({len(q_mean)}d, saved)")
        else:
            st_ = np.load(stats_p); q_mean, q_std = st_["q_mean"], st_["q_std"]
            print("[preprocess] loaded frozen proprio stats")
        print(f"[preprocess] action range: [{a_all.min():.3f}, {a_all.max():.3f}] "
              f"mean|a|={np.abs(a_all).mean():.3f}")
    else:
        av_pca = FrozenPCA.load(av_pca_p)
        eye_pca = FrozenPCA.load(eye_pca_p)
        st = np.load(stats_p)
        q_mean, q_std = st["q_mean"], st["q_std"]

    av_dim = pp["agentview_pca_dim"] * (pp["agentview_grid"] ** 2)
    v_dim = av_dim + pp["eye_pca_dim"]

    # --- transform & write per-task cache ---
    for tid, task, path, lang in todo:
        f = feats[tid]
        N = f["agentview"].shape[0]
        av = av_pca.transform(f["agentview"].reshape(-1, 512)).reshape(N, -1)  # (N,128)
        eye = eye_pca.transform(f["eye"])                                       # (N,64)
        v = np.concatenate([av, eye], axis=1).astype(np.float32)               # (N,192)
        assert v.shape[1] == v_dim, (v.shape, v_dim)
        q = ((f["q"] - q_mean) / q_std).astype(np.float32)
        a = f["a"].astype(np.float32)
        if lang_source == "task_id":
            l = np.zeros(n_tasks_suite, np.float32); l[tid] = 1.0
        elif lang_pca is None:
            l = lang_enc.encode([lang])[0].astype(np.float32)                  # (384,)
        else:
            l = lang_pca.transform(lang_enc.encode([lang]))[0].astype(np.float32)

        out = os.path.join(suite_cache, task.name + ".h5")
        with h5py.File(out, "w") as hf:
            hf.create_dataset("v", data=v, compression="gzip")
            hf.create_dataset("q", data=q, compression="gzip")
            hf.create_dataset("a", data=a, compression="gzip")
            hf.create_dataset("l", data=l)
            hf.create_dataset("ep_bounds", data=f["ep_bounds"])
            hf.attrs["language"] = lang
            hf.attrs["task_name"] = task.name
            hf.attrs["task_id"] = tid
            hf.attrs["v_dim"] = v_dim
        print(f"  wrote {os.path.basename(out)}  v={v.shape} q={q.shape} a={a.shape} l={l.shape}")

    print(f"[preprocess] DONE. cache -> {suite_cache}")
