"""In-memory TBPTT chunk sampler over cached features (§5.1).

Caches are small (~45MB for libero_spatial) so we load everything into RAM and
sample chunks directly — this sidesteps the h5py + DataLoader/fork issues noted
in §9-6 (no DataLoader needed). Chunks never cross demo boundaries (ep_bounds).
"""
import glob
import os
from typing import List, Optional

import h5py
import numpy as np
import torch


class ChunkDataset:
    def __init__(self, cache_root: str, suite: str, chunk_len: int,
                 tasks: Optional[List[int]] = None, max_demos: int = 0,
                 task_weights: Optional[List[float]] = None,
                 allow_short: bool = False, min_valid: int = 0,
                 fixed_start: bool = False, fixed_start_prob: float = 0.0):
        """allow_short=True にすると chunk_len より短いデモも採用し、末尾を 0 で
        埋めて mask で損失から除外する(2026-07-27 追加)。既定 False は従来と同一
        (短いデモは丸ごと捨てる)。
        なぜ必要か: 窓を rollout の 300 step に近づけようとすると、窓 140 で
        spatial のデモが 114/500 しか残らず「窓を伸ばした効果」と「長いデモだけに
        偏ったデータ」が交絡する。窓 200(= 最長デモ 197 超)なら全デモが
        「先頭から末尾まで丸ごと1窓」になり、学習が rollout と完全に一致する。"""
        self.chunk_len = chunk_len
        self.allow_short = allow_short
        # min_valid > 0: 開始位置をランダムに保ったまま窓を伸ばす(2026-07-28 追加)。
        # 窓長ぶん切り出し、デモ末尾を超えた分は 0 埋め + mask。有効長が min_valid 未満に
        # なる開始位置は使わない。
        # なぜ必要か: 窓200(全デモを1窓に収める)は学習と rollout の不整合を消すが、
        # 1デモ 1 系列になり多様性が 27,750 -> 500 に落ちる(ランダム切り出しがデータ拡張
        # として働いていた)。実測で損失は改善するのに閉ループ成功率は 20k で -13.8pt
        # 悪化した。min_valid=100 なら多様性 13,288・平均有効長 112 で両立できる。
        self.min_valid = int(min_valid)
        # fixed_start=True: 常にデモの先頭から窓を切り出す(末尾は 0 埋め + mask)。
        # 動機(2026-07-30): PC-RNN は内部状態を持つので、軌道の途中から切り出すと
        # 「その時点の正しい内部状態」が与えられないまま学習することになる
        # (バットのスイングを途中 50ms だけ切り出して見せるのと同じ)。
        # 先頭固定なら全デモが同じ初期状態から始まるので、学習可能な初期状態
        # (model.learn_init_state)と組み合わせて「正しい立ち上がり」を学べる。
        # 単独では多様性が落ちて悪化することが分かっている
        # (spatial 窓200 先頭固定 13.5% 対 min_valid=100 33.3%)。
        self.fixed_start = bool(fixed_start)
        # fixed_start_prob: 各サンプルごとに この確率で先頭固定、残りはランダム切り出し。
        # 動機(2026-07-30、実測で確定したトレードオフ):
        #   ランダム切り出しのみ  d_min 32.9cm（15k 学習でも動かない）/ handoff t5@25step 6.0/8
        #   先頭固定のみ          d_min 17.9cm（改善）             / handoff t5@25step 0.0/8
        # 先頭固定は「冒頭の立ち上がり」を学ぶ代わりに「任意の状態から続ける」能力を失う。
        # 両方必要なので混ぜる。fixed_start=True より優先度は低い（両方指定なら fixed_start）。
        self.fixed_start_prob = float(fixed_start_prob)
        self.demos = []           # list of dict(v,q,a,l,length)
        # multi-suite training: "libero_spatial+libero_90" concatenates caches
        files = []
        for su in suite.split("+"):
            fs = sorted(glob.glob(os.path.join(cache_root, su, "*.h5")))
            assert fs, f"no cache under {os.path.join(cache_root, su)} (run preprocess first)"
            files += fs
        n_skipped = 0
        for fp in files:
            with h5py.File(fp, "r") as h:
                tid = int(h.attrs["task_id"])
                if tasks is not None and tid not in tasks:
                    continue
                v, q, a = h["v"][:], h["q"][:], h["a"][:]
                l = h["l"][:]
                bounds = h["ep_bounds"][:]
            n_demo = len(bounds) - 1
            if max_demos:
                n_demo = min(n_demo, max_demos)
            for d in range(n_demo):
                s, e = int(bounds[d]), int(bounds[d + 1])
                # 採用の下限: min_valid 指定時はその長さ、allow_short なら無制限、
                # どちらでもなければ従来どおり窓長ぶん必要
                min_len = self.min_valid if self.min_valid > 0 \
                    else (0 if allow_short else chunk_len)
                if e - s < min_len:
                    n_skipped += 1
                    continue
                self.demos.append({
                    "v": v[s:e].astype(np.float32),
                    "q": q[s:e].astype(np.float32),
                    "a": a[s:e].astype(np.float32),
                    "l": l.astype(np.float32),
                    "length": e - s,
                    "tid": tid,
                })
        assert self.demos, "no demo long enough for chunk_len"
        self.n_skipped = n_skipped
        self._lengths = np.array([d["length"] for d in self.demos])
        # sample demos proportionally to number of valid chunk start positions,
        # optionally reweighted per task (e.g. upweight rarely-solved tasks)
        # 取れる開始位置の数に比例させる(短いデモは 1 通りなので下限 1 でクリップ)
        span = self.min_valid if self.min_valid > 0 else chunk_len
        w = np.maximum(self._lengths - span + 1, 1).astype(np.float64)
        if task_weights:
            tw = np.array([float(task_weights[d["tid"]])
                           if d["tid"] < len(task_weights) else 1.0
                           for d in self.demos])
            w = w * tw
        self._probs = w / w.sum()

    def summary(self) -> str:
        need = self.min_valid if self.min_valid > 0 else (
            0 if self.allow_short else self.chunk_len)
        extra = ""
        if self.min_valid > 0:
            extra = (f", window {self.chunk_len} w/ random start "
                     f"(>= {self.min_valid} valid frames, tail padded)")
        elif self.allow_short:
            extra = f", window {self.chunk_len} from frame 0 (tail padded)"
        return (f"{len(self.demos)} demos (>= {need} steps), "
                f"{self.n_skipped} skipped as too short, "
                f"total frames {int(self._lengths.sum())}{extra}")

    def sample_batch(self, batch_size: int, rng: np.random.Generator, device: str):
        """(v, q, a, l, mask) を返す。mask は (B, L) の 0/1 で、1 が有効フレーム。
        デモが窓より短い場合(allow_short=True のとき起こる)は末尾を 0 で埋め、
        その区間の mask を 0 にする。窓以上の長さのデモでは mask は全 1 になり、
        損失値は従来と一致する(scripts/check_loss_regression.py で確認)。"""
        L = self.chunk_len
        V, Q, A, Lg, M = [], [], [], [], []
        idxs = rng.choice(len(self.demos), size=batch_size, p=self._probs)
        for di in idxs:
            d = self.demos[di]
            n = d["length"]
            use_fixed = self.fixed_start or (
                self.fixed_start_prob > 0.0
                and float(rng.random()) < self.fixed_start_prob)
            if use_fixed:
                valid = min(L, n)
                pad = L - valid
                z = lambda x: (x if pad == 0 else np.concatenate(
                    [x, np.zeros((pad,) + x.shape[1:], np.float32)], axis=0))
                sl = slice(0, valid)
                V.append(z(d["v"][sl])); Q.append(z(d["q"][sl])); A.append(z(d["a"][sl]))
                M.append(np.concatenate([np.ones(valid, np.float32),
                                         np.zeros(pad, np.float32)]))
            elif self.min_valid > 0:
                # 開始位置をランダムに保ったまま窓長ぶん切り出し、デモ末尾を超えた分を
                # 0 埋め + mask。有効長は min_valid 以上になる。
                start = int(rng.integers(0, max(n - self.min_valid, 0) + 1))
                valid = min(L, n - start)
                pad = L - valid
                z = lambda x: (x if pad == 0 else np.concatenate(
                    [x, np.zeros((pad,) + x.shape[1:], np.float32)], axis=0))
                sl = slice(start, start + valid)
                V.append(z(d["v"][sl])); Q.append(z(d["q"][sl])); A.append(z(d["a"][sl]))
                M.append(np.concatenate([np.ones(valid, np.float32),
                                         np.zeros(pad, np.float32)]))
            elif n >= L:
                start = int(rng.integers(0, n - L + 1))
                sl = slice(start, start + L)
                V.append(d["v"][sl]); Q.append(d["q"][sl]); A.append(d["a"][sl])
                M.append(np.ones(L, np.float32))
            else:                                  # 末尾ゼロ埋め + mask
                pad = L - n
                z = lambda x: np.concatenate(
                    [x, np.zeros((pad,) + x.shape[1:], np.float32)], axis=0)
                V.append(z(d["v"])); Q.append(z(d["q"])); A.append(z(d["a"]))
                M.append(np.concatenate([np.ones(n, np.float32),
                                         np.zeros(pad, np.float32)]))
            Lg.append(d["l"])
        to = lambda x: torch.from_numpy(np.stack(x)).to(device)
        return to(V), to(Q), to(A), to(Lg), to(M)
