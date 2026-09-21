# PredVLA — Reproduction Repository

This repository provides the code and instructions required to reproduce the training, evaluation, baselines, mechanism ladder, and ablation studies from:

**PredVLA: Predictive Sensorimotor Modeling for Sub-Million-Parameter Robot Manipulation**

PredVLA is a hierarchical predictive-coding policy with 675,732 trainable parameters. It operates on frozen visual and language features (ResNet18 + MiniLM + PCA) and performs online Error Regression (ER) at test time by optimizing the latent variable `c`.

The main evaluation protocol used in the paper is centralized in `predvla/protocol.py`.

## Main Results

| Suite | Success rate |
|---|---:|
| LIBERO-Spatial | **83.19 ± 6.10** |
| LIBERO-Goal | **88.40 ± 3.83** |
| LIBERO-Object | **89.24 ± 6.13** |
| LIBERO-Long (`libero_10`) | **40.57 ± 6.44** |

Main evaluation protocol:

```text
Adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / ER window=40
```

---

## 1. Environment Setup

### Requirements

- Linux
- Python 3.10
- `uv`
- System libraries required for MuJoCo off-screen rendering

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
    libegl1 libgl1 libglew-dev libosmesa6-dev patchelf libglfw3 build-essential
```

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone https://github.com/CyberneticHumanity/PredVLA.git PredVLA_reproduce
cd PredVLA_reproduce

bash setup.sh
```

To force CPU-only PyTorch:

```bash
bash setup.sh --cpu
```

To explicitly specify a CUDA index:

```bash
bash setup.sh --cuda cu124
```

`setup.sh` creates a Python 3.10 virtual environment, installs the required dependencies, downloads LIBERO, and runs basic environment checks.

### Smoke Test

```bash
bash tests/smoke.sh
```

To also test CUDA:

```bash
bash tests/smoke.sh --cuda
```

For a more detailed environment check:

```bash
.venv/bin/python tools/doctor.py
```

The following major dependencies are pinned for reproducibility:

```text
robosuite==1.4.0
mujoco==2.3.7
robomimic==0.2.0
bddl==3.6.0
numpy<2
```

---

## 2. Data Preparation

Check which data are currently available:

```bash
.venv/bin/python tools/prepare_data.py --check
```

### Recommended: Use the Bundled Frozen Feature Cache

The precomputed frozen feature caches (ResNet18 + MiniLM + PCA features of the LIBERO demonstrations, ≈870 MB in total) are **included in this repository** under `data/`, so a plain `git clone` is enough to train and to evaluate the bundled checkpoints:

```text
data/
├── cache_l64/            # shared PCA basis (fit on libero_spatial demos); used for goal / object
│   ├── libero_spatial/
│   ├── libero_goal/
│   ├── libero_object/
│   ├── libero_10/
│   └── libero_90/
├── cache_ps_sp/          # per-suite refit; used for spatial
│   └── libero_spatial/
└── cache_l64_lg/         # PCA refit on libero_10 demos; used for long
    └── libero_10/
```

With the precomputed caches, the full LIBERO demonstration dataset and ResNet/MiniLM/PCA feature extraction are not required for training.

### Rebuilding Features from LIBERO Demonstrations

Download the LIBERO demonstrations:

```bash
.venv/bin/python tools/prepare_data.py --download
```

Build all frozen feature caches:

```bash
.venv/bin/python tools/prepare_data.py --build all
```

---

## 3. Unified Entry Point

Training, evaluation, and aggregation are all launched through `scripts/run.py`.

General usage:

```bash
.venv/bin/python scripts/run.py \
    --track <track> \
    --suite <suite> \
    --seeds <seeds> \
    --stage <stage>
```

### Tracks

| Track | Description |
|---|---|
| `main` | PredVLA |
| `baseline` | BC-LSTM / BC-Transformer |
| `ladder` | Mechanism ladder from PredVLA to BC-LSTM |
| `ablation` | Ablation experiments |
| `all` | Run all tracks |

### Common Arguments

| Argument | Example |
|---|---|
| `--suite` | `libero_spatial`, `libero_goal`, `libero_object`, `libero_10`, `main`, `all` |
| `--seeds` | `1`, `1-7`, `1-14`, `1,3,5` |
| `--stage` | `train`, `eval`, `aggregate`, `all` |
| `--device` | `auto`, `cuda`, `cpu` |
| `--eval-device` | `cpu`, `cuda` |
| `--jobs` | Number of parallel evaluation jobs |
| `--redo` | Re-run completed evaluation cells |
| `--dry-run` | Print the execution plan without running |

Before launching a large experiment, inspect the execution plan with `--dry-run`:

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 1 \
    --stage eval \
    --dry-run
```

---

## 4. Evaluate the Provided Checkpoints

Representative pretrained checkpoints are included under `checkpoints/`, so evaluation can be run without retraining.

Example: evaluate PredVLA on all suites using one seed:

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite all \
    --seeds 1 \
    --stage eval
```

Evaluation logs are written to:

```text
results/logs/
```

Aggregate all available results:

```bash
.venv/bin/python scripts/aggregate.py
```

Aggregate a specific series:

```bash
.venv/bin/python scripts/aggregate.py \
    --match predvla_spatial \
    --reference libero_spatial
```

---

## 5. Retrain PredVLA

### One Suite / One Seed

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 2 \
    --stage train \
    --device auto
```

### Reproduce the Main Results

```bash
.venv/bin/python scripts/run.py \
    --track main \
    --suite all \
    --seeds 1-14 \
    --stage all
```

PredVLA is trained for 30,000 steps by default.

Suite-specific configurations:

| Suite | Config |
|---|---|
| Spatial | `configs/predvla_spatial.yaml` |
| Goal | `configs/predvla_goal.yaml` |
| Object | `configs/predvla_object.yaml` |
| Long | `configs/predvla_long.yaml` |

The Long suite uses longer internal timescales than the three short-horizon suites.

---

## 6. Baselines

BC-LSTM and BC-Transformer use the same frozen front-end features as PredVLA.

Run training and evaluation for both baselines:

```bash
.venv/bin/python scripts/run.py \
    --track baseline \
    --suite main \
    --seeds 0-6 \
    --stage all
```

Evaluate only one baseline:

```bash
.venv/bin/python scripts/run.py \
    --track baseline \
    --model bc_lstm \
    --suite libero_goal \
    --seeds 0 \
    --stage eval
```

| Model | Parameters |
|---|---:|
| BC-LSTM | 674,276 |
| BC-Transformer | 646,628 |

The baselines are trained for 50,000 steps.

---

## 7. Mechanism Ladder

The mechanism ladder progressively removes or replaces components of PredVLA until reaching BC-LSTM.

| ID | Configuration |
|---|---|
| `L0` | PredVLA |
| `L1` | Without online ER |
| `L2` | Without latent variable `c` / free-energy optimization |
| `L3` | Add feed-forward observation input |
| `L4` | Remove the prediction objective |
| `L5` | Remove multi-timescale dynamics |
| `L6` | Replace the PV-RNN cell with an LSTM |
| `L7` | BC-LSTM |

Run the main ladder:

```bash
.venv/bin/python scripts/run.py \
    --track ladder \
    --rung L0-L7 \
    --suite libero_spatial \
    --seeds 1-7 \
    --stage all
```

Run all ladder configurations, including side branches:

```bash
.venv/bin/python scripts/run.py \
    --track ladder \
    --rung all \
    --suite libero_spatial \
    --seeds 1-7 \
    --stage all
```

The exact definition of each rung is provided in `predvla/ladder.py`.

---

## 8. Ablation Studies

| ID | Ablation | Retraining |
|---|---|---|
| `A1` | Uniform timescale | Required |
| `A2` | Remove predictive bottleneck | Required |
| `A3` | Remove efference copy | Required |
| `A4` | Remove visual prediction error from ER | Not required |
| `A5` | Remove proprioceptive prediction error from ER | Not required |
| `A6` | Disable online ER | Not required |

Run ablations that do not require retraining:

```bash
.venv/bin/python scripts/run.py \
    --track ablation \
    --id A4,A5,A6 \
    --suite libero_spatial \
    --seeds 0-6 \
    --stage eval
```

Run ablations that require retraining:

```bash
.venv/bin/python scripts/run.py \
    --track ablation \
    --id A1,A2,A3 \
    --suite libero_spatial \
    --seeds 0-6 \
    --stage all
```

---

## 9. Run All Experiments

First inspect the execution plan:

```bash
.venv/bin/python scripts/run.py \
    --track all \
    --suite all \
    --seeds 1-7 \
    --dry-run
```

Then launch:

```bash
.venv/bin/python scripts/run.py \
    --track all \
    --suite all \
    --seeds 1-7
```

Running all experiments is computationally expensive. In practice, it is recommended to split runs by track, suite, and seed.

---

## 10. Repository Structure

```text
scripts/
  run.py              # Unified entry point for training/evaluation/aggregation
  train.py            # PredVLA training
  evaluate.py         # PredVLA evaluation
  aggregate.py        # Evaluation-log aggregation
  run_ladder.py       # Mechanism ladder
  run_ablation.py     # Ablation experiments

predvla/
  model.py            # PredVLA model
  er.py               # Error Regression
  er_batch.py         # Batched ER
  data.py             # Training data
  frontend.py         # Frozen front end
  protocol.py         # Evaluation protocol
  ladder.py           # Ladder definitions

benchmark/
  train.py            # Baseline training
  eval_suite.py       # Baseline evaluation

configs/
  predvla_*.yaml      # PredVLA configurations

checkpoints/           # Distributed pretrained checkpoints
data/                  # Data and frozen feature caches
results/               # Training and evaluation outputs
tests/smoke.sh         # Minimal smoke test
tools/                 # Environment checks and data preparation
```

---

## 11. Reproducibility Notes

- Use `numpy<2`.
- Keep the LIBERO / robosuite / MuJoCo versions fixed.
- Use the repository's standard runner for Long (`libero_10`) evaluation.
- Use the evaluation settings defined in `predvla/ladder.py` for the mechanism ladder.
- Do not directly compare a single-seed result from a provided checkpoint with the multi-seed average reported in the paper.
- Prefer `scripts/run.py` over calling low-level training or evaluation scripts directly.

---

## Recommended Reproduction Workflow

For a minimal end-to-end check, run:

```bash
# 1. Setup
bash setup.sh

# 2. Smoke test
bash tests/smoke.sh

# 3. Check data
.venv/bin/python tools/prepare_data.py --check

# 4. Evaluate a pretrained checkpoint
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 1 \
    --stage eval

# 5. Retrain one seed
.venv/bin/python scripts/run.py \
    --track main \
    --suite libero_spatial \
    --seeds 2 \
    --stage all

# 6. Extend to all seeds / baselines / ladder / ablations as needed
```

Once this minimal workflow succeeds, expand to the experiments corresponding to the tables in the paper.
