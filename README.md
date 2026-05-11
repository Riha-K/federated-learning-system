# Federated Continual Learning with Knowledge Distillation: Preserving Stability While Enabling Adaptation

##Youtube Video

Link -> https://youtu.be/73MNc-qABog

A reproducible study of **catastrophic forgetting** in federated learning and how **dual-teacher knowledge distillation** (FLwF-2) mitigates it.

Built with **[Flower](https://flower.ai/)** + **PyTorch**. Class-incremental tasks on MNIST, Fashion-MNIST, and CIFAR-10 across 10 / 50 / 100 clients with IID and Dirichlet non-IID partitions (α ∈ {0.01, 0.1, 0.5, 1.0}) — **90 configurations total**.

![Accuracy vs Rounds](images/accuracy_vs_rounds.png)

---

## What this repository does

Each client trains a small CNN on **two sequential class-incremental tasks** (first half of classes, then the second half), without sharing raw data. The server aggregates with FedAvg every round. Two methods are compared:

| Method | Local objective | Behavior on Task 2 |
|---|---|---|
| **FedAvg** | Cross-entropy only | Strong learner, **catastrophically forgets** Task 1 |
| **FLwF-2** | `L = α·CE + β·KL(student ‖ prev_model) + (1−α−β)·KL(student ‖ server_model)` | **Almost no forgetting**; slightly lower plasticity |

`prev_model` (Teacher 1) is the client's snapshot at the end of Task 1; `server_model` (Teacher 2) is the broadcast global model. Distillation uses softmax with temperature `T = 2`; defaults `α = 0.001`, `β = 0.7` (per the FLwF paper).

---

## Headline result: stability–plasticity tradeoff

From `results/summary_table.csv` (final-round, K=10 clients, IID partition):

| Dataset | Method | Final Acc | **Forgetting (Task 1)** |
|---|---|---|---|
| MNIST | FedAvg | 0.483 | **0.998** ← catastrophic |
| MNIST | FLwF-2 | 0.506 | **~0.000** |
| FMNIST | FedAvg | 0.482 | **0.910** |
| FMNIST | FLwF-2 | ≈ FedAvg | **~0.000** |
| CIFAR-10 | FedAvg | 0.362 | **0.668** |
| CIFAR-10 | FLwF-2 | 0.341 | **0.002** |

FLwF-2 reduces forgetting by **2–4 orders of magnitude** at near-equal accuracy — clearly visible in the figures below.

| FedAvg vs FLwF-2 final accuracy | Forgetting over rounds |
|---|---|
| ![](images/fedavg_vs_flwf2_final_accuracy.png) | ![](images/forgetting_vs_rounds.png) |

| IID vs Non-IID final accuracy | Loss vs rounds |
|---|---|
| ![](images/iid_vs_noniid_final_accuracy.png) | ![](images/loss_vs_rounds.png) |

---

## Repository layout

```
.
├── configs/
│   ├── default.yaml          # one-shot experiment config
│   ├── sweep.yaml            # full 90-run grid (methods × datasets × clients × partitions)
│   └── mini_sweep.yaml       # quick smoke sweep
├── src/
│   ├── server.py             # Flower simulation, FedAvg strategy, server-side eval
│   ├── client.py             # local training: FedAvg or FLwF-2 dual-teacher KD
│   ├── model.py              # LightweightCNN (Conv→ReLU→Pool ×2 → FC)
│   ├── data.py               # task split, IID + Dirichlet partitioning
│   ├── utils.py              # KD loss, forgetting, BWT, comm cost
│   ├── experiments.py        # resume-safe sweep runner (subprocess-isolated)
│   └── analyze.py            # plots + summary_table.csv
├── scripts/
│   └── run_sweep_supervised.sh   # restart the sweep on Ray crashes
├── results/
│   ├── metrics.csv               # all rounds × all completed runs
│   ├── runs/*.csv                # one CSV per (method, dataset, K, partition)
│   ├── plots/*.png               # generated figures
│   └── summary_table.csv         # one row per run, headline numbers
├── images/                       # plots used in main.tex (mirrors results/plots/)
├── tables/                       # auto-generated LaTeX tables
├── main.tex                      # IEEE-style paper sources
├── PROJECT.md                    # long-form project guide
├── requirements.txt
└── README.md
```

---

## Installation

Python ≥ 3.9, PyTorch ≥ 2.0. From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

`flwr[simulation]` pulls in Ray, which powers the multi-client simulation.

---

## Quickstart

### 1. Run a single experiment

Edit `configs/default.yaml` (`method`, `dataset`, `federated.num_clients`, `data.partition`, etc.), then:

```bash
.venv/bin/python -m src.server --config configs/default.yaml
```

Per-run metrics are written to `results/runs/<tag>_<hash>.csv` and appended to `results/metrics.csv`.

### 2. Run the full sweep (90 runs)

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 \
  .venv/bin/python -u -m src.experiments --sweep configs/sweep.yaml
```

The sweep is **resume-safe**: each combo (`method × dataset × K × partition`) gets a stable tag, so already-completed runs are skipped on re-launch. Use `--force` to recompute, or `scripts/run_sweep_supervised.sh` to auto-restart on crashes for unattended overnight runs.

Each run executes in its **own subprocess** so Ray and CUDA state are fully released between runs.

### 3. Generate plots and the summary table

```bash
.venv/bin/python -m src.analyze \
  --metrics      results/metrics.csv \
  --out_dir      results/plots \
  --summary_csv  results/summary_table.csv

cp results/plots/*.png images/   # for the LaTeX paper
```

This produces all five figures (`accuracy_vs_rounds.png`, `loss_vs_rounds.png`, `forgetting_vs_rounds.png`, `iid_vs_noniid_final_accuracy.png`, `fedavg_vs_flwf2_final_accuracy.png`) and a CSV with one row per run: Method, Dataset, Clients, Rounds, Accuracy, Convergence Round, Communication Cost, Forgetting.

---

## Default training settings

| Setting | Value |
|---|---|
| Client fraction per round | 0.5 |
| Local epochs | 5 |
| Batch size | 32 |
| Optimizer | SGD, lr 0.01, momentum 0.9 |
| Continual rounds | 10 (Task 1) + 10 (Task 2) |
| FLwF-2 hyperparameters | α<sub>CE</sub>=0.001, β<sub>KD</sub>=0.7, T=2 |
| Seeds | 42 (`random`, `numpy`, `torch`) |

Override any of these via the YAML config — no code changes needed.

---

## Metrics reported

- **Global test accuracy / loss** — every round, on a class-balanced server-side test set
- **Task-wise accuracy** — separate accuracy on Task 1 and Task 2 classes
- **Catastrophic forgetting**: $F_t = \max_{r \le R_t} a_t^{(r)} - a_t^{(R)}$
- **Backward Transfer (BWT)** — average impact of later training on earlier tasks
- **Communication cost** — total bytes broadcast + uploaded across rounds
- **Convergence round** — first round where global accuracy ≥ 80% (often N/A for class-incremental setups)

---

## Reproducibility

- All randomness seeded with `42` (Python `random`, NumPy, PyTorch).
- Each (method, dataset, K, partition) hashes to a stable run-id; results are written deterministically.
- Re-running the sweep on identical hardware reproduces the numbers in `results/metrics.csv` to within standard nondeterministic GPU variance.




