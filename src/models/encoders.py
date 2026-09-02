"""Frozen encoders for PredictiveCoding-VLA (Phase 1, §3.1 / §3.3).

- FrozenResNet18: ImageNet-pretrained ResNet-18 (eval, frozen). Produces
  agentview tokens (2x2 = 4 tokens x 512) and eye_in_hand token (1 x 512).
- LanguageEncoder: sentence-transformers all-MiniLM-L6-v2 (frozen, 384-d).

Both are frozen; nothing here is trained (design principle P2). The vision
encoder is reused at eval time so preprocessing and rollout share it exactly.
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FrozenResNet18(nn.Module):
    """Frozen ResNet-18 truncated at layer4 (spatial feature map).

    For a 128x128 input the layer4 map is 4x4x512. We adaptive-avg-pool it to
    2x2 for agentview (-> 4 tokens) and 1x1 for eye_in_hand (-> 1 token).
    """

    def __init__(self, agentview_grid: int = 2):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        # keep everything up to and including layer4 (drop avgpool + fc)
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.agentview_grid = agentview_grid
        self.feat_dim = 512
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    @torch.no_grad()
    def _prep(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """images_uint8: (B, H, W, 3) uint8 -> normalized (B, 3, H, W) float."""
        x = images_uint8.to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def feature_map(self, images_uint8: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._prep(images_uint8))  # (B, 512, h, w)

    @torch.no_grad()
    def agentview_tokens(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """-> (B, grid*grid, 512), grid=2 by default (4 tokens)."""
        fm = self.feature_map(images_uint8)
        pooled = F.adaptive_avg_pool2d(fm, self.agentview_grid)  # (B,512,g,g)
        b = pooled.shape[0]
        return pooled.reshape(b, self.feat_dim, -1).permute(0, 2, 1).contiguous()

    @torch.no_grad()
    def eye_token(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """-> (B, 512) global average pooled."""
        fm = self.feature_map(images_uint8)
        return F.adaptive_avg_pool2d(fm, 1).reshape(fm.shape[0], self.feat_dim)


class LanguageEncoder:
    """Frozen MiniLM sentence encoder (384-d)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, sentences: List[str]) -> np.ndarray:
        emb = self.model.encode(
            sentences, convert_to_numpy=True, normalize_embeddings=False
        )
        return emb.astype(np.float32)
