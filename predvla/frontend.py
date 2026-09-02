"""凍結された前段（視覚・言語・固有感覚の正規化）。既存 src/ の実装をそのまま使う。

比較可能性のため、既存 pcv 系と完全に同一の特徴を作る:
  agentview 128x128 -> 凍結 ResNet18 の 2x2 トークン -> PCA 32 x4 = 128
  手先カメラ        -> 凍結 ResNet18 の全体特徴      -> PCA 64
  合わせて v = 192 次元
  言語 -> MiniLM 384 -> PCA 64
  q = 関節 7 + 指 2 を norm_stats.npz の平均/標準偏差で z-score

学習キャッシュ（data/cache_l64）と同じ PCA・統計を読むので、
学習時の値と評価時の値がズレない。
"""
import os
import sys

import numpy as np
import torch

# 凍結前段は src/ の実装を共有する（作り直さない）
# predvla/frontend.py -> リポジトリ直下
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.pca import FrozenPCA  # noqa: E402
from src.models.encoders import FrozenResNet18, LanguageEncoder  # noqa: E402


class Frontend:
    def __init__(self, cache_root: str, device: str = "cpu", agentview_grid: int = 2):
        cache = cache_root if os.path.isabs(cache_root) \
            else os.path.join(_ROOT, cache_root)
        self.av_pca = FrozenPCA.load(os.path.join(cache, "pca_agentview.npz"))
        self.eye_pca = FrozenPCA.load(os.path.join(cache, "pca_eye.npz"))
        self.lang_pca = FrozenPCA.load(os.path.join(cache, "pca_language.npz"))
        st = np.load(os.path.join(cache, "norm_stats.npz"))
        self.q_mean, self.q_std = st["q_mean"], st["q_std"]
        self.device = device
        self.encoder = FrozenResNet18(agentview_grid=agentview_grid).to(device)
        self.lang_enc = LanguageEncoder(device="cpu")

    @torch.no_grad()
    def v(self, agent_img, eye_img):
        av = torch.from_numpy(np.ascontiguousarray(agent_img[None])).to(self.device)
        eye = torch.from_numpy(np.ascontiguousarray(eye_img[None])).to(self.device)
        av_tok = self.encoder.agentview_tokens(av).cpu().numpy().reshape(-1, 512)
        eye_tok = self.encoder.eye_token(eye).cpu().numpy()
        av_p = self.av_pca.transform(av_tok).reshape(-1)
        eye_p = self.eye_pca.transform(eye_tok).reshape(-1)
        return torch.from_numpy(np.concatenate([av_p, eye_p]).astype(np.float32))

    def q(self, obs):
        x = np.concatenate([obs["robot0_joint_pos"],
                            obs["robot0_gripper_qpos"]]).astype(np.float32)
        return torch.from_numpy(((x - self.q_mean) / self.q_std).astype(np.float32))

    @torch.no_grad()
    def lang(self, text: str):
        e = np.asarray(self.lang_enc.encode([text]))
        return torch.from_numpy(self.lang_pca.transform(e).reshape(-1).astype(np.float32))
