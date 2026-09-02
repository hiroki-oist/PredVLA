"""Frozen PCA with whitening (Phase 1, §3.2 / §3.3).

The projection is FIT ONCE on training data and then FROZEN (design principle
P2): predicting a *learned* projection collapses the representation, so the
visual/language latents are produced by a fixed PCA.

We store only what the linear transform needs: y = (x - mean) @ W, where for a
whitening PCA  W = components_.T / sqrt(explained_variance_). This makes the
transform trivially reproducible from a small npz without pickling sklearn.
"""
import numpy as np


class FrozenPCA:
    def __init__(self, mean: np.ndarray, W: np.ndarray):
        self.mean = mean.astype(np.float32)          # (d_in,)
        self.W = W.astype(np.float32)                # (d_in, d_out)

    @property
    def dim_in(self) -> int:
        return int(self.W.shape[0])

    @property
    def dim_out(self) -> int:
        return int(self.W.shape[1])

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int, whiten: bool = True,
            seed: int = 0) -> "FrozenPCA":
        """Fit on X (n_samples, d_in). If fewer usable components than
        n_components are available (n_samples <= d_in), the transform matrix is
        zero-padded on the right so dim_out == n_components stays fixed."""
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=np.float64)
        d_in = X.shape[1]
        k = int(min(n_components, X.shape[0] - 1, d_in))
        k = max(k, 1)
        pca = PCA(n_components=k, whiten=whiten, random_state=seed)
        pca.fit(X)
        comp = pca.components_                        # (k, d_in)
        if whiten:
            scale = np.sqrt(pca.explained_variance_)  # (k,)
            scale = np.where(scale > 1e-12, scale, 1e-12)
            W = (comp / scale[:, None]).T             # (d_in, k)
        else:
            W = comp.T                                # (d_in, k)
        if k < n_components:
            pad = np.zeros((d_in, n_components - k), dtype=W.dtype)
            W = np.concatenate([W, pad], axis=1)      # (d_in, n_components)
        return cls(pca.mean_, W)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) @ self.W

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, W=self.W)

    @classmethod
    def load(cls, path: str) -> "FrozenPCA":
        d = np.load(path)
        return cls(d["mean"], d["W"])
