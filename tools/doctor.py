#!/usr/bin/env python
"""環境の健診。学習・評価を回す前にこれを通す。

    python tools/doctor.py            全部見る
    python tools/doctor.py --quick    シミュレータの起動確認を飛ばす（速い）

見るもの
  1. Python / OS / CPU / RAM
  2. torch と device（CUDA / MPS / CPU）
  3. 依存パッケージの版（★版が違うと成功判定が変わるものを別扱いで警告する）
  4. MuJoCo のオフスクリーン描画（EGL / OSMesa のどちらが使えるか）
  5. LIBERO の導入とデモ hdf5 の有無
  6. 凍結特徴キャッシュ（PCA 基底）の有無 — スイートごとに違う点を明示する
  7. LIBERO 環境を 1 つ実際に起動して 1 枚描画できるか

終了コードは「致命的な問題の数」。警告だけなら 0。
"""
from __future__ import annotations

import argparse
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OK, WARN, BAD = "  OK  ", " 警告 ", "★致命"
n_bad = 0
n_warn = 0


def line(mark: str, label: str, detail: str = "") -> None:
    global n_bad, n_warn
    if mark == BAD:
        n_bad += 1
    elif mark == WARN:
        n_warn += 1
    print(f"[{mark}] {label:34s} {detail}")


def section(t: str) -> None:
    print(f"\n--- {t} " + "-" * max(0, 58 - len(t)))


# 版が違うと **成功判定や物理が変わる** ので厳密に合わせるべきもの
STRICT = {"robosuite": "1.4.0", "mujoco": "2.3.7", "robomimic": "0.2.0",
          "bddl": "3.6.0"}
# 版が多少違っても数値が変わらないもの（下限だけ見る）
LOOSE = ["torch", "torchvision", "numpy", "h5py", "PyYAML", "scikit-learn",
         "sentence-transformers", "transformers", "opencv-python", "einops"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="LIBERO 環境の起動確認を飛ばす")
    a = ap.parse_args()

    print("=" * 68)
    print(" PredVLA 再現環境の健診")
    print("=" * 68)

    # -------------------------------------------------- 1. 基盤
    section("1. 基盤")
    py = sys.version.split()[0]
    line(OK if py.startswith("3.10") else WARN, "Python", py +
         ("" if py.startswith("3.10") else "  ★3.10 で検証している"))
    line(OK, "OS", f"{platform.system()} {platform.release()}")
    try:
        import multiprocessing
        ncpu = multiprocessing.cpu_count()
    except Exception:
        ncpu = -1
    line(OK, "CPU コア", str(ncpu))
    try:
        with open("/proc/meminfo") as f:
            tot = int(f.readline().split()[1]) / 2 ** 20
        line(OK if tot >= 16 else WARN, "RAM", f"{tot:.0f} GB" +
             ("" if tot >= 16 else "  ★Long(T=500) の学習は 16GB 以上を推奨"))
    except Exception:
        pass

    # -------------------------------------------------- 2. torch と device
    section("2. torch と device")
    try:
        import torch
    except Exception as e:
        line(BAD, "torch", f"import できない: {e}")
        return n_bad
    line(OK, "torch", f"{torch.__version__}  (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            line(OK, f"CUDA device {i}", f"{p.name}  {p.total_memory/2**30:.1f} GB")
    else:
        line(WARN, "CUDA", "使えない → CPU で回る（学習は 20〜40 倍遅い）")
    try:
        if torch.backends.mps.is_available():
            line(OK, "MPS", "使える")
    except Exception:
        pass
    from src.utils import compat
    line(OK, "既定 device (auto)", compat.pick_device("auto"))

    # -------------------------------------------------- 3. 依存の版
    section("3. 依存の版")
    import importlib.metadata as md

    def ver(name: str):
        try:
            return md.version(name)
        except Exception:
            return None

    for name, want in sorted(STRICT.items()):
        got = ver(name)
        if got is None:
            line(BAD, name, f"入っていない（要 {want}）")
        elif got == want:
            line(OK, name, got)
        else:
            line(BAD, name, f"{got}  ★{want} でなければ成功判定が変わりうる")
    for name in LOOSE:
        got = ver(name)
        line(OK if got else BAD, name, got or "入っていない")
    nv = ver("numpy")
    if nv and int(nv.split(".")[0]) >= 2:
        line(BAD, "numpy のメジャー版", f"{nv}  ★robosuite 1.4 は numpy<2 が必要")

    # -------------------------------------------------- 4. 描画
    section("4. MuJoCo のオフスクリーン描画")
    line(OK, "MUJOCO_GL", os.environ.get("MUJOCO_GL", "(未設定 → compat が egl にする)"))
    backends = []
    for be in ("egl", "osmesa", "glfw"):
        os.environ["MUJOCO_GL"] = be
        try:
            import importlib
            import mujoco
            importlib.reload(mujoco)
            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><body><geom size='.1'/></body></worldbody></mujoco>")
            d = mujoco.MjData(m)
            r = mujoco.Renderer(m, 64, 64)
            r.update_scene(d)
            r.render()
            backends.append(be)
        except Exception:
            pass
    if backends:
        line(OK, "使える描画バックエンド", ", ".join(backends) +
             ("  ← egl 推奨（GPU・ヘッドレス可）" if "egl" in backends
              else "  ★egl が無いので MUJOCO_GL=osmesa を設定して使う"))
        os.environ["MUJOCO_GL"] = "egl" if "egl" in backends else backends[0]
    else:
        line(BAD, "描画バックエンド", "どれも動かない。"
                  "sudo apt-get install libegl1 libgl1 libosmesa6-dev を試す")

    # -------------------------------------------------- 5. LIBERO
    section("5. LIBERO")
    try:
        import libero
        from libero.libero import benchmark, get_libero_path
        line(OK, "libero", os.path.dirname(libero.__file__))
        try:
            line(OK, "bddl_files", get_libero_path("bddl_files"))
            line(OK, "init_states", get_libero_path("init_states"))
        except Exception as e:
            line(WARN, "get_libero_path", str(e))
        d = benchmark.get_benchmark_dict()
        line(OK, "ベンチマーク定義", f"{len(d)} 件")
    except Exception as e:
        line(BAD, "libero", f"import できない: {e}\n"
                            f"          bash setup.sh で入れる")
        benchmark = None

    section("6. LIBERO のデモ hdf5（data/libero/<suite>/*.hdf5）")
    from predvla import protocol as P
    for suite in P.ALL_SUITES:
        d = os.path.join(ROOT, "data", "libero", suite)
        if not os.path.isdir(d):
            line(WARN, suite, f"{d} が無い → tools/prepare_data.py --download")
            continue
        h5 = [f for f in os.listdir(d) if f.endswith(".hdf5")]
        line(OK if len(h5) >= 10 else WARN, suite,
             f"{len(h5)} / 10 タスク  ({d})")

    section("7. 凍結特徴キャッシュ（★スイートごとに PCA 基底が違う）")
    for suite in P.ALL_SUITES:
        cr = P.CACHE_ROOT[suite]
        d = os.path.join(ROOT, cr, suite)
        pcas = [f for f in ("pca_agentview.npz", "pca_eye.npz",
                            "pca_language.npz", "norm_stats.npz")
                if os.path.exists(os.path.join(ROOT, cr, f))]
        if not os.path.isdir(d):
            line(WARN, f"{suite} ({cr})",
                 "無い → tools/prepare_data.py --suite " + suite)
            continue
        h5 = [f for f in os.listdir(d) if f.endswith(".h5")]
        mark = OK if (len(h5) >= 10 and len(pcas) == 4) else WARN
        line(mark, f"{suite} ({cr})",
             f"タスク {len(h5)}/10, 基底 {len(pcas)}/4")

    # -------------------------------------------------- 8. 実起動
    if not a.quick and benchmark is not None:
        section("8. LIBERO 環境を 1 つ起動して 1 枚描画する")
        try:
            compat.apply()
            from libero.libero.envs import OffScreenRenderEnv
            suite = benchmark.get_benchmark_dict()["libero_spatial"]()
            task = suite.get_task(0)
            bddl = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl,
                                     camera_heights=128, camera_widths=128)
            env.reset()
            obs, *_ = env.step([0.0] * 7)
            img = obs["agentview_image"]
            env.close()
            line(OK, "rollout 1 step", f"agentview {img.shape} {img.dtype}")
        except Exception as e:
            line(BAD, "rollout 1 step", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 68)
    print(f" 致命 {n_bad} 件 / 警告 {n_warn} 件")
    if n_bad == 0:
        print(" → 学習と評価を回せる。")
    else:
        print(" → 致命の項目を直すこと。README.md の「環境構築」を見る。")
    print("=" * 68)
    return n_bad


if __name__ == "__main__":
    sys.exit(main())
