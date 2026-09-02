"""Compatibility shims for the pinned LIBERO/robosuite stack on a modern
PyTorch (2.13) setup. Supports macOS/Apple Silicon and Linux.

Import and call `apply()` early (before touching LIBERO benchmark APIs).
"""
import functools
import os
import platform

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def _setup_env():
    if IS_MACOS:
        # Let unimplemented MPS ops fall back to CPU instead of hard-failing (P5).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif IS_LINUX:
        # Headless MuJoCo off-screen rendering: EGL works without a display on
        # NVIDIA GPUs. Override with MUJOCO_GL=osmesa/glfw if needed.
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# MUJOCO_GL / MPS fallback must be in the environment before mujoco/torch are
# first imported; importing this module early achieves that even when apply()
# runs later.
_setup_env()


def pick_device(requested: str = "auto") -> str:
    """Resolve a torch device string. 'auto' -> cuda > mps > cpu; an
    unavailable requested device falls back to cpu."""
    import torch

    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[compat] cuda unavailable -> cpu")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[compat] mps unavailable -> cpu")
        return "cpu"
    return requested


def accelerator() -> str:
    """Best available accelerator for conv-heavy work ('cuda'/'mps'), else 'cpu'."""
    return pick_device("auto")


def patch_torch_load():
    """LIBERO stores init-state files as numpy-pickled tensors and loads them
    with a bare ``torch.load(path)``. PyTorch >= 2.6 defaults ``weights_only``
    to True, which rejects those files. The init-state files ship with LIBERO
    and are trusted, so force ``weights_only=False`` for LIBERO's calls.
    """
    import torch

    if getattr(torch.load, "_predvla_patched", False):
        return
    _orig = torch.load

    @functools.wraps(_orig)
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _patched._predvla_patched = True
    torch.load = _patched


def apply():
    _setup_env()
    patch_torch_load()
