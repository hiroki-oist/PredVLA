"""Training entry (§5).

Examples:
  python scripts/train.py --config configs/smoke.yaml
  python scripts/train.py --config configs/base.yaml --set data.suite=libero_spatial train.run_name=spatial_s0
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from src.utils import compat
from src.utils.config import load_config, apply_overrides
from src.utils.seed import set_seed
from src.data.dataset import ChunkDataset
from src.models import registry

compat.apply()


def git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", PROJ, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(PROJ, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROJ, "configs/base.yaml"))
    ap.add_argument("--device", default=None, help="override cfg.device (cpu|mps|cuda|auto)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=val")
    ap.add_argument("--resume", action="store_true",
                    help="results/<run>/resume.pt から再開(重み+optimizer+scheduler+RNG)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.device:
        cfg["device"] = args.device
    device = compat.pick_device(cfg["device"])

    set_seed(cfg["seed"])
    tr = cfg["train"]
    chunk_len = tr["burn_in"] + tr["loss_len"]
    allow_short = bool(tr.get("allow_short", False))
    min_valid = int(tr.get("min_valid", 0))

    ds = ChunkDataset(_abs(cfg["paths"]["cache_root"]), cfg["data"]["suite"], chunk_len,
                      tasks=cfg["data"]["tasks"], max_demos=cfg["data"]["max_demos"],
                      task_weights=cfg["data"].get("task_weights"),
                      allow_short=allow_short, min_valid=min_valid,
                      fixed_start=bool(tr.get("fixed_start", False)),
                      fixed_start_prob=float(tr.get("fixed_start_prob", 0.0)))
    print(f"[train] dataset: {ds.summary()}")

    # モデルの選択は src/models/registry.py に集約（旧表記の型名も受ける）
    agent = registry.build(cfg).to(device)
    print(f"[train] model: {registry.DISPLAY[registry.normalize(cfg['model'].get('type', 'legacy_pc'))]}")
    print(f"[train] trainable params: {agent.num_params()/1e6:.3f}M  device={device}")

    # init_lr_mult: 学習可能初期状態(init_hv/init_hl/init_hu)だけ学習率を倍率で上げる。
    # 動機(2026-07-30): learn_init_state を入れても 5,000 step で |init_hv| が 0.0145 しか
    # 動かない(勾配 6.86e-05、lr 6e-05)。h_v のスケールは概ね ±1 なので「ゼロから少し
    # 動いた」段階に留まる。d_min は 32.9 → 17.9cm に縮んだので方向は正しく、
    # 動く量が足りていない疑いがある。init_* だけ別のパラメータ群にして lr を上げる。
    _im = float(tr.get("init_lr_mult", 1.0))
    if _im != 1.0:
        _init = [p_ for n_, p_ in agent.named_parameters()
                 if n_.startswith("init_") and p_.requires_grad]
        _rest = [p_ for n_, p_ in agent.named_parameters()
                 if not n_.startswith("init_") and p_.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": _rest, "lr": tr["lr"]},
             {"params": _init, "lr": tr["lr"] * _im}],
            weight_decay=tr["weight_decay"])
        print(f"[train] init_* のみ lr x{_im:g} ({len(_init)} 個のパラメータ群)")
    else:
        opt = torch.optim.AdamW(agent.parameters(), lr=tr["lr"],
                                weight_decay=tr["weight_decay"])
    total_steps = tr["total_steps"]
    # lr スケジュール(2026-07-31 追加。既定 cosine は従来と完全に同一)。
    #   cosine(既定) : CosineAnnealingLR(T_max=total_steps)、eta_min=0 なので終端で lr=0
    #   none         : 一定 lr（スケジューラなし）
    # なぜ選べるようにしたか: eta_min=0 の cosine では total_steps 付近で lr が消えるため、
    # 「終端で飽和した」という観察が「lr が 0 になった」と区別できない。
    # プロジェクト最初の学習ループ(5020cda, 2026-07-20)からこの設定なので、
    # 過去の「@50k がピーク」「もう飽和した」系の結論はすべてこの交絡を含む。
    # lr 一定条件と比べれば決着する。
    _ls = str(tr.get("lr_schedule", "cosine")).lower()
    if _ls == "none":
        sched = None
        print(f"[train] lr スケジュールなし（一定 {tr['lr']:g}）")
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=float(tr.get("lr_eta_min", 0.0)))
        print(f"[train] lr cosine T_max={total_steps} eta_min={tr.get('lr_eta_min', 0.0):g}")

    run_dir = _abs(os.path.join("runs", tr["run_name"]))
    ckpt_dir = _abs(os.path.join("results", tr["run_name"]))
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir)
    except Exception as e:
        print(f"[train] tensorboard unavailable ({e}); logging to stdout only")
        writer = None

    # --- 乱数の初期化（2026-08-01 修正）---
    # 従来はここで cfg["seed"]（トップレベル）だけを読んでおり、ジョブが渡す
    # --set train.seed=N（= cfg["train"]["seed"]）が一切効いていなかった。
    # さらに torch の seed をどこでも設定していなかったため、
    #   Ubuntu(CUDA)  … OS エントロピーで毎回違い、結果的にばらついていた
    #   Mac(MPS)      … 既定シードが固定で 3 シードが完全一致した
    #                    （final ema loss が 3 本すべて −3.0335）
    # train.seed を優先し、torch 側も明示的に seed する。
    _seed = int(tr.get("seed", cfg.get("seed", 0)))
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(_seed)
        except Exception:
            pass
    print(f"[train] seed={_seed} (train.seed 優先, torch/numpy とも設定)")
    rng = np.random.default_rng(_seed)
    # --- 再開(--resume): 重み・オプティマイザ・スケジューラ・RNG をすべて復元する。
    # 重みだけ引き継ぐと Adam のモーメントがゼロに戻って学習軌跡が変わるため、
    # 途中でコードを差し替えたい場合はこちらを使う(数値等価な最適化なら軌跡は保たれる)。
    start_step = 0
    if args.resume:
        rp = os.path.join(ckpt_dir, "resume.pt")
        if os.path.exists(rp):
            st = torch.load(rp, weights_only=False, map_location=device)
            agent.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            if sched is not None and st.get("sched") is not None:
                sched.load_state_dict(st["sched"])
            torch.set_rng_state(st["rng_torch"].cpu() if hasattr(st["rng_torch"], "cpu")
                                else st["rng_torch"])
            if st.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([x.cpu() if hasattr(x, "cpu") else x
                                              for x in st["rng_cuda"]])
            rng.bit_generator.state = st["rng_np"]
            start_step = int(st["step"])
            print(f"[train] resumed from {rp} at step {start_step} "
                  f"(saved by git {st.get('git_hash','?')})")
        else:
            print(f"[train] --resume 指定だが {rp} が無いので最初から開始")
    ghash = git_hash()
    ss_max = tr.get("ss_prob_max", 0.0)
    ss_ramp = max(int(tr.get("ss_ramp_steps", 1)), 1)
    agent.train()
    fwd = agent
    compile_requested = bool(tr.get("compile", False))
    if compile_requested and device != "cuda":
        # validated on cuda only; keep eager on cpu/mps (macOS) to be safe
        print(f"[train] compile requested but device={device}; staying eager")
        compile_requested = False
    if compile_requested:
        # CUDA-graph capture of the unrolled time loop kills the kernel-launch
        # overhead that dominates this launch-bound model (~8x solo).
        #
        # 内側推論(pc_infer)を含むモデルを compile するには 2 つ必要(2026-07-29):
        #   ・model.pc_infer_backend: func   torch.func.grad を使い requires_grad_() の
        #     葉テンソルを作らない。これで "requires_grad_() intermediate leaked as
        #     output" の graph break が消える(graph 数 1 / break 0)
        #   ・torch._dynamo.config.trace_autograd_ops = True
        #     torch.autograd.grad をトレース可能にする(既定 False では graph break)
        # 実測(窓200/batch32、pcv2、単独): eager 520.7 ms → default 123.0 ms →
        # reduce-overhead 41.3 ms（12.6倍）。eager との勾配の最大相対差 3.3e-07 で
        # 一致を確認済み(scripts/check_compile_equiv.py)。
        import torch._dynamo as _dyn
        if hasattr(_dyn.config, "trace_autograd_ops"):
            _dyn.config.trace_autograd_ops = True
        elif getattr(agent, "pc_k", 0) > 0:
            print("[train] 警告: trace_autograd_ops が無いので pc_infer で graph break "
                  "する(compile の効果が大きく落ちる)")
        if getattr(agent, "pc_k", 0) > 0 and getattr(agent, "pc_backend", "") != "func":
            print("[train] 警告: pc_k>0 かつ pc_infer_backend != func。graph break が "
                  "残るので model.pc_infer_backend: func を推奨")
        _mode = os.environ.get("PREDVLA_COMPILE_MODE", "reduce-overhead")
        fwd = torch.compile(agent, mode=_mode)
        print(f"[train] torch.compile enabled ({_mode}) "
              f"pc_backend={getattr(agent, 'pc_backend', '-')}")
    t0 = time.time()
    # EMA is kept on-device; float()/isfinite on the loss forces a GPU sync,
    # so all host-side reads happen only every log_every steps.
    ema_t = None
    compiled = compile_requested
    for step in range(start_step + 1, total_steps + 1):
        ss_prob = ss_max * min(1.0, step / ss_ramp)
        v, q, a, l, msk = ds.sample_batch(tr["batch_size"], rng, device)
        if compiled:
            # cudagraphs: mark step boundary; pass ss_prob as tensor so the
            # ramping value doesn't trigger a recompile every step.
            torch.compiler.cudagraph_mark_step_begin()
            ss_arg = torch.full((), ss_prob, device=device)
            out = fwd(v, q, a, l, burn_in=tr["burn_in"],
                      lambda_v=tr["lambda_v"], lambda_q=tr["lambda_q"], lambda_a=tr["lambda_a"],
                      noise_v=tr["noise_v"], noise_q=tr["noise_q"], training=True,
                      ss_prob=ss_arg, use_ss=ss_max > 0, mask=msk)
        else:
            out = fwd(v, q, a, l, burn_in=tr["burn_in"],
                      lambda_v=tr["lambda_v"], lambda_q=tr["lambda_q"], lambda_a=tr["lambda_a"],
                      noise_v=tr["noise_v"], noise_q=tr["noise_q"], training=True,
                      ss_prob=ss_prob, mask=msk)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(agent.parameters(), tr["grad_clip"])
        opt.step()
        if sched is not None:
            sched.step()

        # clone: cudagraph output buffers are reused across steps
        tl_t = out["total"].detach().clone() if compiled else out["total"].detach()
        ema_t = tl_t if ema_t is None else 0.98 * ema_t + 0.02 * tl_t
        if step % tr["log_every"] == 0 or step == 1:
            tl = float(tl_t)
            ema = float(ema_t)
            if not math.isfinite(tl):
                print(f"[train] non-finite loss at step {step}; stopping early.")
                torch.save({"model": agent.state_dict(), "cfg": cfg, "step": step,
                            "git_hash": ghash, "nan": True},
                           os.path.join(ckpt_dir, f"step_{step}_nan.pt"))
                break
            lv, lq, la = (float(out[k].detach()) for k in ("loss_v", "loss_q", "loss_a"))
            msg = (f"step {step:6d}/{total_steps} | total {tl:.4f} (ema {ema:.4f}) "
                   f"| v {lv:.4f} q {lq:.4f} "
                   f"a {la:.4f} | FE {float(out['fe']):.4f} "
                   f"| |a_hat| {float(out['a_absmean']):.3f} vs |a| {float(out['a_target_absmean']):.3f} "
                   f"| PB {float(out['pb_norm']):.3f} | gnorm {float(gnorm):.2f} | ss {ss_prob:.2f}")
            print(msg)
            if writer:
                for k in ("total", "loss_v", "loss_q", "loss_a"):
                    writer.add_scalar(f"loss/{k}", float(out[k].detach()), step)
                writer.add_scalar("aux/free_energy", float(out["fe"]), step)
                writer.add_scalar("aux/action_absmean_pred", float(out["a_absmean"]), step)
                writer.add_scalar("aux/action_absmean_target", float(out["a_target_absmean"]), step)
                writer.add_scalar("aux/pb_norm", float(out["pb_norm"]), step)
                writer.add_scalar("aux/grad_norm", float(gnorm), step)
                writer.add_scalar("aux/lr",
                              sched.get_last_lr()[0] if sched is not None
                              else opt.param_groups[0]["lr"], step)

        if step % tr["ckpt_every"] == 0 or step == total_steps:
            ckpt = os.path.join(ckpt_dir, f"step_{step}.pt")
            torch.save({"model": agent.state_dict(), "cfg": cfg, "step": step,
                        "git_hash": ghash}, ckpt)
            # 再開用の状態は別ファイルに最新のみ保持する(評価用 ckpt を重くしないため)。
            # オプティマイザ状態を持たない再開は Adam のモーメントがゼロに戻り学習軌跡が
            # 変わるので、途中でコードを差し替えたいときはこちらを使う。
            if tr.get("save_resume", True):
                torch.save({"model": agent.state_dict(), "opt": opt.state_dict(),
                            "sched": (sched.state_dict() if sched is not None else None),
                     "step": step, "cfg": cfg,
                            "git_hash": ghash,
                            "rng_torch": torch.get_rng_state(),
                            "rng_cuda": (torch.cuda.get_rng_state_all()
                                         if torch.cuda.is_available() else None),
                            "rng_np": rng.bit_generator.state},
                           os.path.join(ckpt_dir, "resume.pt"))
            print(f"[train] saved {ckpt}")

    if writer:
        writer.close()
    print(f"[train] done in {time.time()-t0:.1f}s  final ema loss {float(ema_t):.4f}")


if __name__ == "__main__":
    main()
