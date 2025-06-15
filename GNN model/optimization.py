#!/usr/bin/env python
"""
optuna_landmark_gnn.py
-------------------------------------------------------
Hyper-parameter search for facial-expression GCN model
using Optuna, with LR scheduler + early-stopping.

Run:
    python optuna_landmark_gnn.py --data_dir /path/to/dataset --n_trials 50
"""

# ── global log/warning suppression ──────────────────────────────
import os, warnings, logging, absl.logging
# 1) TensorFlow / TF-Lite / XNNPACK
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 0=all, 1=info, 2=warning, 3=error
# 2) MediaPipe / absl & GLOG
os.environ["GLOG_minloglevel"] = "3"            # silence GLOG backend used by absl
absl.logging.set_verbosity(absl.logging.ERROR)  # absl Python side
# 3) Torch DataLoader pin-memory warning
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)
# (optional) silence *all* futurewarnings from NumPy, etc.
warnings.simplefilter("ignore", category=FutureWarning)
# ────────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from optuna.trial import Trial
from tqdm import tqdm

from const import *


# >>> import objects from your training script <<<
from GNNmodel import evaluate, train_epoch, LandmarkGCN, make_graph_loaders
from DataLoader import FaceExpressionLandmarksDS


def objective(trial: Trial, data_dir: str) -> float:
    """Optuna objective: maximize weighted-F1 on the validation split."""
    # ---------- sample hyper-parameters ----------
    hidden_dim     = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    # hidden_dim     = 128
    lr             = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay   = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # batch_size     = trial.suggest_categorical("batch_size", [16, 32, 64])
    batch_size     = 64

    scheduler_name = "cosine"
    # scheduler_name = trial.suggest_categorical("scheduler", ["none", "step", "cosine", "plateau"])

    print(f"Trial {trial.number}:  hidden_dim: {hidden_dim}, lr: {lr:.6f}, weight_decay: {weight_decay:.6f}, batch_size: {batch_size}, scheduler: {scheduler_name}")
    # ---------------------------------------------

    # ---------- data loader ----------------------
    train_loader, val_loader = make_graph_loaders(data_dir, batch_size,
                                                  WORKERS_NUM, K_NEAREST_NEIGHBOR)

    # ----------- model config ---------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = LandmarkGCN(in_channels=2, hidden=hidden_dim, num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ---------- scheduler -------------------------
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.5, patience=3)
    else:
        scheduler = None
    # ----------------------------------------------

    best_val_loss   = 100000  # large initial value
    patience   = 7        # early-stop patience
    no_improve = 0

    for epoch in tqdm(range(1, EPOCHS)):
        # ---- train loop ----
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # ---- validation ----
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        trial.report(val_acc, epoch)          # Optuna pruning hook

        # ---- scheduler step ----
        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()

        # ---- early-stopping / pruning ----
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss + 1e-4 < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            print(f"[Epoch {epoch}] New best val_acc: {val_acc:.4f} | ")
        else:
            no_improve += 1
            print(f"[Epoch {epoch}] No improvement in val_acc for {no_improve} epochs")

        if no_improve >= patience:
            break   # early stop

    return val_acc   # Optuna will maximize this


def main():
    # ----------  args ----------
    trials = 20
    study_name = "gnn1"
    # STUDY_DB_PATH = r"C:\Users\matan\Desktop\Code\Face_expression_recognition_model\Trained_GNN_models\gnn_optuna.db"
    STUDY_DB_PATH = "sqlite:///C:/Users/matan/Desktop/Code/Face_expression_recognition_model/Trained_GNN_models/gnn_optuna.db"

    LOAD_TRAINING = True

    study = optuna.create_study(storage=STUDY_DB_PATH, direction="maximize", study_name=study_name, pruner=MedianPruner(n_warmup_steps=5), load_if_exists=LOAD_TRAINING)

    study.optimize(lambda trial: objective(trial, DATA_DIR), n_trials=trials, show_progress_bar=True)

    # -------- summary --------
    print("Best weighted-F1: {:.4f}".format(study.best_value))
    print("Best hyper-params:")
    for k, v in study.best_trial.params.items():
        print(f" {k}: {v}")


if __name__ == "__main__":
    main()
