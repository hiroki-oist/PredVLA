"""predvla 用のデータ。1 デモ = 1 系列を先頭から、固定長 T にゼロ埋め + mask。

PV-RNN と同じく自由変数 c が (系列, 時刻) で索引されるので、
ランダム切り出しはできない（c の添字が絶対時刻に対応する必要がある）。

既存の凍結視覚特徴キャッシュ（../data/cache_l64）をそのまま読む。
比較可能性のため前段は既存と完全に同一（凍結 ResNet18 + PCA、MiniLM + PCA）。

視覚は 5Hz（vision_stride=4 step ごとに更新）なので、
mask_v は「視覚が更新された step」のみ 1 にして視覚の誤差項を落とす
（LibPvrnn の obsMask_ と同じ発想）。
"""
import glob
import os

import h5py
import numpy as np
import torch


class SeqDataset:
    def __init__(self, cache_root: str, suite: str, T: int, vision_stride: int = 4,
                 tasks=None, max_demos: int = 0):
        # ★suite はカンマ区切りで複数指定できる（2026-08-26 追加）。
        #   例: "libero_spatial,libero_goal,libero_object,libero_10"
        #   4 スイートをまとめて 1 モデルに学習させるために足した。
        #   ★同じ cache_root の下にある = PCA 基底と norm_stats が共通、が前提。
        #     基底が違うキャッシュを混ぜてはいけない（data/cache_l64 は 4 スイート
        #     すべてを単一基底で持っているので、そこを使う）。
        #   ★task_id はスイート間で衝突する（どのスイートにも 0..9 がある）ので、
        #     スイートの並び順 s に対し tid + 100*s にずらして一意にする。
        #   単一スイート指定のときは従来と完全に同一の動作（ずらしも 0）。
        suites = [x.strip() for x in str(suite).split(",") if x.strip()]
        files = []
        for si, su in enumerate(suites):
            fs = sorted(glob.glob(os.path.join(cache_root, su, "*.h5")))
            assert fs, f"no cache under {os.path.join(cache_root, su)}"
            files += [(fp, si) for fp in fs]
        self.suites = suites
        V, Q, A, L, M, TID = [], [], [], [], [], []
        n_trunc = 0
        for fp, si in files:
            with h5py.File(fp, "r") as h:
                tid = int(h.attrs["task_id"]) + 100 * si
                if tasks is not None and tid not in tasks:
                    continue
                v, q, a, l = h["v"][:], h["q"][:], h["a"][:], h["l"][:]
                bounds = h["ep_bounds"][:]
            n_demo = len(bounds) - 1
            if max_demos:
                n_demo = min(n_demo, max_demos)
            for d in range(n_demo):
                s, e = int(bounds[d]), int(bounds[d + 1])
                n = min(e - s, T)
                if e - s > T:
                    n_trunc += 1
                pad = T - n
                def z(x):
                    x = x[s:s + n].astype(np.float32)
                    return x if pad == 0 else np.concatenate(
                        [x, np.zeros((pad,) + x.shape[1:], np.float32)], 0)
                V.append(z(v)); Q.append(z(q)); A.append(z(a))
                L.append(l.astype(np.float32))
                M.append(np.concatenate([np.ones(n, np.float32),
                                         np.zeros(pad, np.float32)]))
                TID.append(tid)
        self.v = torch.from_numpy(np.stack(V))          # (N,T,192)
        self.q = torch.from_numpy(np.stack(Q))          # (N,T,9)
        self.a = torch.from_numpy(np.stack(A))          # (N,T,7)
        self.l = torch.from_numpy(np.stack(L))          # (N,64)
        self.mask = torch.from_numpy(np.stack(M))       # (N,T)
        self.tid = np.array(TID)
        self.n_trunc = n_trunc
        self.T = T
        # 視覚が更新される step のみ 1
        mv = np.zeros(T, np.float32)
        mv[::vision_stride] = 1.0
        self.mask_v = torch.from_numpy(mv)[None]        # (1,T)

    def __len__(self):
        return self.v.shape[0]

    def summary(self):
        n = len(self)
        eff = self.mask.sum(1)
        sus = ("  スイート " + "+".join(self.suites)) if len(self.suites) > 1 else ""
        return (f"{n} 系列（1 デモ = 1 系列、先頭から T={self.T}）{sus}  "
                f"有効長 平均 {eff.mean():.1f} / 最小 {int(eff.min())} / 最大 {int(eff.max())}  "
                f"T 超過で切った系列 {self.n_trunc}  "
                f"視覚が有効な step {int(self.mask_v.sum())}/{self.T}")

    def to(self, device):
        self.v = self.v.to(device); self.q = self.q.to(device)
        self.a = self.a.to(device); self.l = self.l.to(device)
        self.mask = self.mask.to(device); self.mask_v = self.mask_v.to(device)
        return self

    def batch(self, idx):
        return (self.v[idx], self.q[idx], self.a[idx], self.l[idx],
                self.mask[idx], self.mask_v.expand(len(idx), self.T))
